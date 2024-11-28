"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import chex
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, JaxMARLWrapper
from jaxmarl.environments.multi_agent_env import MultiAgentEnv # , State

from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State

import wandb
import functools
import matplotlib.pyplot as plt

from jaxmarl.policies import HyperNetwork
from jaxmarl.utils import snd


class MPEWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self,
              key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self,
             key,
             state,
             action):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action
        )
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape for space in spaces])

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


class ActorRNN(nn.Module):
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
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi

class ActorHyperRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float
    dim_capabilities: int # per team
    hypernet_kwargs: dict

    def hyper_forward(self, in_dim, out_dim, target_in, hyper_in, time_steps, batch_size):
        """
        Compute y = xW + b where W/b are created by a hypernetwork.

        in_dim : dimension of target input x
        out_dim : dimension of target output y
        target_in : target input
        hyper_in : input to hypernet

        time_steps/batch_size : parallel dims
        """
        num_weights = (in_dim * out_dim)
        weight_hypernet = HyperNetwork(hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"], output_dim=num_weights, init_scale=self.hypernet_kwargs["INIT_SCALE"], num_layers=self.hypernet_kwargs["NUM_LAYERS"], use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"])
        weights = weight_hypernet(hyper_in).reshape(time_steps, batch_size, in_dim, out_dim)

        num_biases = out_dim
        bias_hypernet = HyperNetwork(hidden_dim=self.hypernet_kwargs["HIDDEN_DIM"], output_dim=num_biases, init_scale=0, num_layers=self.hypernet_kwargs["NUM_LAYERS"], use_layer_norm=self.hypernet_kwargs["USE_LAYER_NORM"])
        biases = bias_hypernet(hyper_in).reshape(time_steps, batch_size, 1, out_dim)

        # compute y = xW + b
        # NOTE: slicing here expands embedding to be (1, in_dim) @ (in_dim, out_dim)
        # with leading dims for time_steps, batch_size
        target_out = jnp.matmul(target_in[:, :, None, :], weights) + biases
        target_out = target_out.squeeze(axis=2) # remove extra dim needed for computation
        return target_out

    @nn.compact
    def __call__(self, hidden, x, train=True):
        orig_obs, dones = x

        # separate obs into capabilities and observations
        # (env gives obs = orig obs+cap)
        # NOTE: this is hardcoded to match simple_spread's computation
        cap = orig_obs[:, :, -self.dim_capabilities:]
        obs = orig_obs[:, :, :-self.dim_capabilities]

        time_steps, batch_size, obs_dim = obs.shape

        # RNN
        embedding = nn.Dense(
            self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        # actor_mean = nn.Dense(self.hidden_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(
        #     embedding
        # )
        # actor_mean = nn.relu(actor_mean)
        actor_mean = self.hyper_forward(self.hidden_dim, self.hidden_dim, embedding, orig_obs, time_steps, batch_size)
        actor_mean = nn.relu(actor_mean)

        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi

class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    """Stack along agent dimension"""
    x = jnp.stack([x[a] for a in agent_list])
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Unstack along agent dimension and store in dict"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def unbatchify_snd(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((x.shape[0], num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, viz_test_env):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

    # add test env for visualization / greedy metrics
    test_env = MPEWorldStateWrapper(viz_test_env)
    test_env = MPELogWrapper(test_env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        if config["AGENT_HYPERAWARE"]:
            actor_network = ActorHyperRNN(
                action_dim=env.action_space(env.agents[0]).n,
                hidden_dim=config["GRU_HIDDEN_DIM"],
                init_scale=config['AGENT_INIT_SCALE'],
                hypernet_kwargs=config['AGENT_HYPERNET_KWARGS'],
                dim_capabilities=env.dim_capabilities
            )
        else:
            actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)

        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros(
                (config["NUM_ACTORS"], 1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)
            ),
            jnp.zeros((config["NUM_ACTORS"], 1, config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])
        # print(ac_init_x[0].shape)
        # print(ac_init_x[1].shape)
        # print(ac_init_hstate.shape)

        # vmap initialization over agent dimension
        rngs = jax.random.split(_rng_actor, config["NUM_ACTORS"])
        actor_network_params = jax.vmap(actor_network.init, in_axes=0)(rngs, ac_init_hstate, ac_init_x)

        # log agent param count
        actor_param_count = sum(x.size for x in jax.tree_util.tree_leaves(actor_network_params))
        wandb.log({"actor_param_count": actor_param_count})
        print("actor_param_count", actor_param_count)
        print("-" * 10)
        print("DETAILED ACTOR PARAM COUNT:")
        for name, param in jax.tree_util.tree_flatten_with_path(actor_network_params)[0]:
            print(f"{name}: {param.shape}")
        print("-" * 10)


        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], env.world_state_size(),)),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ENVS"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        # print(cr_init_x[0].shape)
        # print(cr_init_x[1].shape)
        # print(cr_init_hstate.shape)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ENVS"]*config["NUM_ACTORS"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, viz_env_states, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = jax.vmap(actor_network.apply, in_axes=(0, 0, 1))(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                # print(ac_in[0].shape)
                # print(ac_in[1].shape)
                # print(hstates[0].shape)
                # print(log_prob.shape)

                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0,1)
                world_state = world_state.reshape((config["NUM_ACTORS"] * config["NUM_ENVS"],-1))
                last_done = last_done.reshape(-1)
                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                # print(cr_in[0].shape)
                # print(cr_in[1].shape)
                # print(hstates[1].shape)
                # print(value.shape)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # NOTE: this is a bandaid
                # issue stems from HMT's info metrics being team-wise (16,1,1) but other stats being per-agent (16,4)
                # thus, duplicate across all n_agents
                if "makespan" in info:
                    info["makespan"] = info["makespan"].reshape(-1, 1).repeat(test_env.num_agents, axis=1)

                if "quota_met" in info:
                    info["quota_met"] = info["quota_met"].reshape(-1, 1).repeat(test_env.num_agents, axis=1)

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"] * config["NUM_ENVS"])), info)

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                reward_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    reward_batch,
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), viz_env_states, rng)
                return runner_state, transition

            initial_hstates = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hstates, viz_env_states, rng = runner_state

            last_world_state = last_obs["world_state"].swapaxes(0,1)
            last_world_state = last_world_state.reshape((config["NUM_ACTORS"] * config["NUM_ENVS"],-1))
            last_done = last_done.reshape(-1)
            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()
            last_val = last_val.reshape((config["NUM_ACTORS"], config["NUM_ENVS"]))
            # print(cr_in[0].shape)
            # print(cr_in[1].shape)
            # print(hstates[1].shape)
            # print(last_val.shape)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    done = done.reshape((config["NUM_ACTORS"], config["NUM_ENVS"]))
                    value = value.reshape((config["NUM_ACTORS"], config["NUM_ENVS"]))
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
                values = jnp.reshape(traj_batch.value, (-1, config["NUM_ACTORS"], config["NUM_ENVS"]))
                return advantages, advantages + values

            advantages, targets = _calculate_gae(traj_batch, last_val)
            # print(advantages.shape)
            # print(targets.shape)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        obs, done, log_prob, action = traj_batch
                        # _, pi = jax.vmap(actor_network.apply, in_axes=(0, 0, 1))(
                        #     actor_params,
                        #     init_hstate.squeeze(),
                        #     (obs, done),
                        # )
                        _, pi = actor_network.apply(actor_params, init_hstate.squeeze(), (obs, done))
                        pi_log_prob = pi.log_prob(action)

                        # CALCULATE ACTOR LOSS
                        logratio = pi_log_prob - log_prob
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

                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        # RERUN NETWORK
                        # print(init_hstate.shape)
                        # print(traj_batch.world_state.shape)
                        # print(traj_batch.done.shape)
                        world_state = jnp.reshape(
                            traj_batch.world_state,
                            list(traj_batch.world_state.shape[:-1]) + [config["NUM_ENVS"] // config["NUM_MINIBATCHES"], -1]
                        )
                        init_hstate = jnp.swapaxes(init_hstate, 0, 1)
                        done = jnp.swapaxes(traj_batch.done, 0, 1)
                        world_state = jnp.swapaxes(world_state, 0, 1)

                        init_hstate = jnp.reshape(init_hstate, (init_hstate.shape[0], -1, init_hstate.shape[-1]))
                        done = jnp.reshape(done, (done.shape[0], -1))
                        world_state = jnp.reshape(world_state, (world_state.shape[0], -1, world_state.shape[-1]))

                        # print(init_hstate.shape)
                        # print(done.shape)
                        # print(world_state.shape)
                        _, value = critic_network.apply(critic_params, init_hstate.squeeze(), (world_state, done))

                        # CALCULATE VALUE LOSS
                        traj_batch_value = jnp.swapaxes(traj_batch.value, 0, 1)
                        traj_batch_value = jnp.reshape(traj_batch_value, (traj_batch_value.shape[0], -1))
                        targets = jnp.swapaxes(targets, 0, 1)
                        targets = jnp.reshape(targets, (targets.shape[0], -1))

                        value_pred_clipped = traj_batch_value + (
                            value - traj_batch_value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    # print(ac_init_hstate.shape)
                    # print(traj_batch.obs.shape)
                    # print(traj_batch.done.shape)
                    # print(traj_batch.log_prob.shape)
                    # print(traj_batch.action.shape)
                    # print(advantages.shape)
                    actor_loss, actor_grads = jax.vmap(actor_grad_fn)(
                        actor_train_state.params, ac_init_hstate, (traj_batch.obs, traj_batch.done, traj_batch.log_prob, traj_batch.action), advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # print(init_hstates[0].shape)
                # print(init_hstates[1].shape)
                init_hstates_batch = (
                    init_hstates[0][None, :],
                    jnp.reshape(init_hstates[1], (1, config["NUM_ACTORS"], config["NUM_ENVS"], -1))
                )
                # init_hstates = jax.tree_map(lambda x: jnp.reshape(
                #     x, (1, config["NUM_ACTORS"], config["NUM_ENVS"], -1)
                # ), init_hstates)
                # print(init_hstates[0].shape)

                batch = (
                    init_hstates_batch[0],
                    init_hstates_batch[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                # permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=1), batch
                # )

                # print(traj_batch.world_state.shape)
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
                # print(minibatches[0].shape)
                # print(minibatches[1].shape)
                # print(minibatches[2].world_state.shape)

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            rng = update_state[-1]

            # update the greedy policy rewards/metrics/viz (test policy)
            # NOTE: here I'm doing this after every update, could limit to a set TEST_INTERVAL like QMIX
            rng, _rng = jax.random.split(rng)
            test_results = _get_greedy_metrics(_rng, train_states[0].params)
            test_metrics, viz_env_states = test_results["metrics"], test_results["viz_env_states"]
            metric["test_metrics"] = test_metrics
            # metric["test_metrics"] = {}

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
            runner_state = (train_states, env_state, last_obs, jnp.reshape(last_done, (config["NUM_ACTORS"], -1)), hstates, viz_env_states, rng)
            return (runner_state, update_steps), metric

        def _get_greedy_metrics(rng, actor_params):
            """
            Tests greedy policy in test env (which may have different teams).
            """
            # define a step in test_env, then lax.scan over it to rollout the greedy policy in the env, gather viz_env_states
            def _greedy_env_step(step_state, unused):
                actor_params, env_state, last_obs, last_done, ac_hstate, rng = step_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstate, pi = jax.vmap(actor_network.apply, in_axes=(0, 0, 1))(actor_params, ac_hstate, ac_in)
                # here, instead of sampling from distribution, take mode
                action = pi.mode()
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

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

                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"] * config["NUM_ENVS"])), info)

                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                reward_batch = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()

                step_state = (actor_params, env_state, obsv, done_batch, ac_hstate, rng)
                return step_state, (reward_batch, done_batch, info, env_state.env_state, obs_batch, ac_hstate)

            # reset test env
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            init_obsv, env_state = jax.vmap(test_env.reset, in_axes=(0,))(reset_rng)
            init_dones = jnp.zeros((config["NUM_ACTORS"], config["NUM_ENVS"]), dtype=bool)
            ac_hstate = ScannedRNN.initialize_carry(config["GRU_HIDDEN_DIM"], config["NUM_ACTORS"], config["NUM_ENVS"])
            rng, _rng = jax.random.split(rng)

            step_state = (actor_params, env_state, init_obsv, init_dones, ac_hstate, _rng)
            step_state, (rewards, dones, infos, viz_env_states, obs, hstate) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            # NOTE: because MAPPO is on-policy, the JaxMARL team did not build this actor to collect across multiple episodes, unlike QMIX. Thus, this can just be tied to NUM_ENVS

            # get snd, NOTE: dim_c multiplier is currently hardcoded since it works for both fire and transport
            # TODO: define SND for MAPPO
            snd_obs = obs.reshape(config['ENV_KWARGS']['max_steps'], len(env.agents), config["NUM_ENVS"], -1)
            snd_hstate = hstate.reshape(config['ENV_KWARGS']['max_steps'], len(env.agents), config["NUM_ENVS"], -1)
            snd_value = snd(rollouts=snd_obs, hiddens=snd_hstate, dim_c=test_env.num_agents*2, params=actor_params, alg='mappo_ns', agent=actor_network)
            # snd_value = 0

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
        greedy_ret = _get_greedy_metrics(_rng, actor_train_state.params) # initial greedy metrics
        test_metrics, viz_env_states = greedy_ret["metrics"], greedy_ret["viz_env_states"]
        # test_metrics = {}

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"], config["NUM_ENVS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            viz_env_states,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="mappo_homogenous_rnn_mpe")
def main(config):

    config = OmegaConf.to_container(config)
    config["NUM_STEPS"] = config["ENV_KWARGS"]["max_steps"]

    env_name = config["ENV_NAME"]
    alg_name = "MAPPO-NO-PS"

    hyper_tag = "hyper" if config["AGENT_HYPERAWARE"] else "normal"
    recurrent_tag = "RNN" if config["AGENT_RECURRENT"] else "MLP"
    aware_tag = "aware" if config["ENV_KWARGS"]["capability_aware"] else "unaware"

    wandb_tags = [
        alg_name.upper(),
        env_name,
        hyper_tag,
        recurrent_tag,
        aware_tag,
        f"jax_{jax.__version__}",
    ]
    if 'tag' in config:
        wandb_tags.append(config['tag'])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=wandb_tags,
        name=f'{alg_name} / {hyper_tag} {recurrent_tag} {aware_tag} / {env_name}',
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
        actor_state = outs['runner_state'][0][0][0]
        params = jax.tree.map(lambda x: x[0], actor_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')
        if config["VISUALIZE_FINAL_POLICY"]:

            # TODO: I have no idea what this object is from
            # print(outs['runner_state'][1])
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


if __name__=="__main__":
    main()
