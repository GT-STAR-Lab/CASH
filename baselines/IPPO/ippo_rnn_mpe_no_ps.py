"""
Based on PureJaxRL Implementation of PPO.

Note, this file will only work for MPE environments with homogenous agents (e.g. Simple Spread).

"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
import chex
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

import jaxmarl
from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State
from jaxmarl.wrappers.baselines import MPELogWrapper
from jaxmarl.utils import snd

import wandb
import functools


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        hidden_size = ins.shape[-1]
        batch_size = ins.shape[:-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *batch_size),
            rnn_state,
        )

        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (*batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    """Stack along agent dimension"""
    x = jnp.stack([x[a] for a in agent_list])
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Unstack along agent dimension and store in dict"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, viz_test_env):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = MPELogWrapper(env)

    # add test env for visualization / greedy metrics
    test_env = MPELogWrapper(viz_test_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK - need to maintain separate parameters for each agent
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (config["NUM_ACTORS"], 1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
            ),
            jnp.zeros((config["NUM_ACTORS"], 1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])

        # vmap initialization over agent dimension
        rngs = jax.random.split(_rng, config["NUM_ACTORS"])
        network_params = jax.vmap(network.init, in_axes=0)(rngs, init_hstate, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, viz_env_state, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                # vmap network forward pass across agent network parameters
                hstate, pi, value = jax.vmap(network.apply, in_axes=(0, 0, 1))(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"],config["NUM_ENVS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], (env.num_agents, 1)),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, viz_env_state, rng)
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, viz_env_state, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            # vmap network forward pass across agent network parameters
            _, _, last_val = jax.vmap(network.apply, in_axes=(0, 0, 1))(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean()

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = jax.vmap(grad_fn)(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(
                    init_hstate, (1, config["NUM_ACTORS"], config["NUM_ENVS"], -1)
                )
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=1), batch
                # )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_ACTORS"], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[3:]),
                        ),
                        2,
                        0,
                    ),
                    batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            metric = jax.tree_map(
                lambda x: x.reshape(
                    (config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents)
                ),
                traj_batch.info,
            )
            ratio_0 = loss_info[1][3].at[0,0].get().mean()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            rng = update_state[-1]

            rng, _rng = jax.random.split(rng)
            test_results = _get_greedy_metrics(_rng, train_state.params)
            test_metrics, viz_env_states = test_results["metrics"], test_results["viz_env_states"]
            metric["test_metrics"] = test_metrics

            def callback(metric, infos):
                # make IO call to wandb.log()
                env_name = config["ENV_NAME"]
                if env_name == "MPE_simple_fire":
                    wandb.log(
                        {
                            "returns": metric["returned_episode_returns"][-1, :].mean(),
                            "timestep": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"], # num of env interactions (formerly "env_step")
                            **metric["loss"],
                            # **info_metrics,
                            **{k:v.mean() for k, v in metric['test_metrics'].items()},
                        }
                    )
                elif env_name == "MPE_simple_transport":
                    info_metrics = {
                        'quota_met': jnp.max(infos['quota_met'], axis=0).mean(),
                        'makespan': jnp.min(infos['makespan'], axis=0).mean(),
                    }
                    wandb.log(
                        {
                            "returns": metric["returned_episode_returns"][-1, :].mean(),
                            "timestep": metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"], # num of env interactions (formerly "env_step")
                            **metric["loss"],
                            **info_metrics,
                            **{k:v.mean() for k, v in metric['test_metrics'].items()}
                        }
                    )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric, traj_batch.info)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, viz_env_state, rng)
            return (runner_state, update_steps), metric

        def _get_greedy_metrics(rng, params):
            """
            Tests greedy policy in test env (which may have different teams).
            """
            # define a step in test_env, then lax.scan over it to rollout the greedy policy in the env, gather viz_env_states
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_done, hstate, rng = step_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                # vmap forward pass for agent networks across params
                hstate, pi, value = jax.vmap(network.apply, in_axes=(0, 0, 1))(params, hstate, ac_in)
                # print(pi.probs.shape)

                # here, instead of sampling from distribution, take mode
                action = pi.mode()
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
               #  print(env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    test_env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # NOTE: this is a bandaid
                # issue stems from HMT's info metrics being team-wise (16,1,1) but other stats being per-agent (16,4)
                # thus, duplicate across all n_agents
                if "makespan" in info:
                    info["makespan"] = info["makespan"].reshape(-1, 1).repeat(test_env.num_agents, axis=1)

                if "quota_met" in info:
                    info["quota_met"] = info["quota_met"].reshape(-1, 1).repeat(test_env.num_agents, axis=1)

                # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                reward_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                step_state = (params, env_state, obsv, done_batch, hstate, rng)
                return step_state, (reward_batch, done_batch, info, env_state.env_state, obs_batch, hstate)

            # reset test env
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            init_obsv, env_state = jax.vmap(test_env.reset, in_axes=(0,))(reset_rng)
            init_dones = jnp.zeros((config["NUM_ACTORS"], config["NUM_ENVS"]), dtype=bool)
            hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])
            rng, _rng = jax.random.split(rng)

            step_state = (params, env_state, init_obsv, init_dones, hstate, _rng)
            step_state, (rewards, dones, infos, viz_env_states, obs, hstate) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )

            snd_value = snd(rollouts=obs, hiddens=hstate, dim_c=test_env.num_agents*2, params=params, alg='ippo', agent=network)

            # define fire_env_metrics (should be attached to env, but is not)
            def fire_env_metrics(final_env_state):
                """
                Return success rate (pct of envs where both fires are put out)
                and percent of fires which are put out, out of all fires.
                """
                p_pos = final_env_state.p_pos
                rads = final_env_state.rad

                num_agents = viz_test_env.num_agents
                num_landmarks = rads.shape[-1] - num_agents
                num_envs = config["NUM_ENVS"]

                def _agent_in_range(agent_i: int, agent_p_pos, landmark_pos, landmark_rad):
                    """
                    Finds all agents in range of a single landmark.
                    """
                    delta_pos = agent_p_pos[agent_i] - landmark_pos
                    dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
                    return (dist < landmark_rad)

                def _fire_put_out(landmark_i: int, agent_p_pos, agent_rads, landmark_p_pos, landmark_rads):
                    """
                    Determines if a single landmark is covered by enough ff power.
                    """
                    landmark_pos = landmark_p_pos[landmark_i, :]
                    landmark_rad = landmark_rads[landmark_i]

                    agents_on_landmark = jax.vmap(_agent_in_range, in_axes=[0, None, None, None])(jnp.arange(num_agents), agent_p_pos, landmark_pos, landmark_rad)
                    firefighting_level = jnp.sum(jnp.where(agents_on_landmark, agent_rads, 0))
                    return firefighting_level > landmark_rad

                def _fires_put_out_per_env(env_i, p_pos, rads):
                    """
                    Determines how many fires are covered in a single parallel env.
                    """
                    agent_p_pos = p_pos[env_i, :num_agents, :]
                    landmark_p_pos = p_pos[env_i, num_agents:, :]

                    agent_rads = rads[env_i, :num_agents]
                    landmark_rads = rads[env_i, num_agents:]

                    landmarks_covered = jax.vmap(_fire_put_out, in_axes=[0, None, None, None, None])(jnp.arange(num_landmarks), agent_p_pos, agent_rads, landmark_p_pos, landmark_rads)

                    return landmarks_covered

                fires_put_out = jax.vmap(_fires_put_out_per_env, in_axes=[0, None, None])(jnp.arange(num_envs), p_pos, rads)
                # envs where num_landmarks fires are put out / total
                success_rate = jnp.count_nonzero(jnp.sum(fires_put_out, axis=1) == num_landmarks) / num_envs
                # sum of all fires put out / total num fires
                pct_fires_put_out = jnp.sum(fires_put_out) / (num_envs * num_landmarks)
                return success_rate, pct_fires_put_out

            # compute metrics for fire env or HMT
            final_env_state = step_state[1].env_state
            fire_env_metrics = fire_env_metrics(final_env_state)
            # rewards are [NUM_STEPS, NUM_ENVS*NUM_AGENTS] by default
            rewards = rewards.reshape(config["NUM_STEPS"], config["NUM_ENVS"], config["ENV_KWARGS"]["num_agents"])
            test_returns = jnp.sum(rewards, axis=[0,2]).mean()

            env_name = config["ENV_NAME"]
            if env_name == "MPE_simple_fire":
                metrics = {
                    'test_returns': test_returns, # episode returns
                    'test_fire_success_rate': fire_env_metrics[0],
                    'test_pct_fires_put_out': fire_env_metrics[1],
                    'test_snd': snd_value,
                    # **{'test_'+k:v for k,v in first_infos.items()},
                }
            elif env_name == "MPE_simple_transport":
                info_metrics = {
                    'quota_met': jnp.max(infos['quota_met'], axis=0),
                    'makespan': jnp.min(infos['makespan'], axis=0)
                }
                metrics = {
                    'test_returns': test_returns, # episode returns
                    'test_snd': snd_value,
                    **{'test_'+k:v for k,v in info_metrics.items()},
                }

            # return metrics & viz_env_states
            return {"metrics": metrics, "viz_env_states": viz_env_states}

        rng, _rng = jax.random.split(rng)
        greedy_ret = _get_greedy_metrics(_rng, train_state.params) # initial greedy metrics
        test_metrics, viz_env_states = greedy_ret["metrics"], greedy_ret["viz_env_states"]
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"], config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            viz_env_states,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_mpe")
def main(config):
    config = OmegaConf.to_container(config)
    config["NUM_STEPS"] = config["ENV_KWARGS"]["max_steps"]

    env_name = config["ENV_NAME"]
    alg_name = "IPPO"

    wandb_tags = [
        alg_name.upper(),
        env_name,
        f"jax_{jax.__version__}",
    ]
    if 'tag' in config:
        wandb_tags.append(config['tag'])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=wandb_tags,
        name=f'{alg_name} / {env_name}',
        config=config,
        mode=config["WANDB_MODE"],
    )

    # for visualization
    viz_test_env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"], test_env_flag=True)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, viz_test_env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    if config['SAVE_PATH'] is not None:

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        # TODO: I have no idea what this object is from
        # print(outs['runner_state'][1])
        actor_state = outs['runner_state'][0][0]
        params = jax.tree.map(lambda x: x[0], actor_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')
        if config["VISUALIZE_FINAL_POLICY"]:

            # TODO: I have no idea what this object is from
            viz_env_states = outs['runner_state'][0][-2]

            # build a list of states manually from vectorized seq returned by
            # make_train() for desired seeds/envs
            for seed in range(config["NUM_SEEDS"]):
                for env in range(config["VIZ_NUM_ENVS"]):
                    state_seq = []
                    for i in range(config["NUM_STEPS"]):
                        if env_name == "MPE_simple_fire":
                            this_step_state = State(
                                p_pos=viz_env_states.p_pos[seed, i, env, ...],
                                p_vel=viz_env_states.p_vel[seed, i, env, ...],
                                c=viz_env_states.c[seed, i, env, ...],
                                accel=viz_env_states.accel[seed, i, env, ...],
                                rad=viz_env_states.rad[seed, i, env, ...],
                                done=viz_env_states.done[seed, i, env, ...],
                                step=i,
                            )
                            state_seq.append(this_step_state)
                        if env_name == "MPE_simple_transport":
                            this_step_state = State(
                                p_pos=viz_env_states.p_pos[seed, i, env, ...],
                                p_vel=viz_env_states.p_vel[seed, i, env, ...],
                                c=viz_env_states.c[seed, i, env, ...],
                                accel=viz_env_states.accel[seed, i, env, ...],
                                rad=viz_env_states.rad[seed, i, env, ...],
                                done=viz_env_states.done[seed, i, env, ...],
                                capacity=viz_env_states.capacity[seed, i, env, ...],
                                site_quota=viz_env_states.site_quota[seed, i, env, ...],
                                step=i,
                            )
                            state_seq.append(this_step_state)

                    # save visualization to GIF for wandb display
                    visualizer = MPEVisualizer(viz_test_env, state_seq, env_name=env_name)
                    video_fpath = f'{save_dir}/{alg_name}-seed-{seed}-rollout.gif'
                    visualizer.animate(video_fpath)
                    wandb.log({f"env-{env}-seed-{seed}-rollout": wandb.Video(video_fpath)})

    # force multiruns to finish correctly
    wandb.finish()

if __name__ == "__main__":
    main()
