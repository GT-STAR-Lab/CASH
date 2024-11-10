"""
End-to-End JAX Implementation of QMix.

Notice:
- Agents are controlled by a single RNN architecture.
- You can choose if sharing parameters between agents or not.
- Works also with non-homogenous agents (different obs/action spaces)
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can use TD Loss (pymarl2) or DDQN loss (pymarl)
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
from flax.core import frozen_dict
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts

from jaxmarl.environments.mpe import MPEVisualizer
from jaxmarl.environments.mpe.simple import State

from jaxmarl.policies import ScannedRNN, AgentMLP, AgentHyperMLP, AgentRNN, AgentHyperRNN, HyperNetwork
from jaxmarl.utils import snd


class MixingNetwork(nn.Module):
    """
    Mixing network for projecting individual agent Q-values into Q_tot. Follows the original QMix implementation.
    """
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, q_vals, states):

        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0)) # (time_steps, batch_size, n_agents)

        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim*n_agents, init_scale=self.init_scale)(states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(states)
        w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim, init_scale=self.init_scale)(states)
        b_2 = HyperNetwork(hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale)(states)

        # monotonicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze() # (time_steps, batch_size)


class EpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e  = start_e
        self.end_e    = end_e
        self.duration = duration
        self.slope    = (end_e - start_e) / duration

    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)

    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):

        def explore(q, eps, key):
            key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
            pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions

        eps = self.get_epsilon(t)
        keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals)))) # get a key for each agent
        chosen_actions = jax.tree.map(lambda q, k: explore(q, eps, k), q_vals, keys)
        return chosen_actions

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict


def make_train(config, log_train_env, log_test_env, viz_test_env, env_name="MPE_simple_fire"):
    """
    NOTE: log_train_env and log_test_env should be identical, other than a single
    test_env_flag, which causes log_test_env to only sample from test_capabilities.
    """

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )


    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        # for SimpleFacmac / other MPE envs with adversaries, only use a subset of the agents
        has_adversaries = hasattr(log_train_env._env, "adversaries")
        training_agents = log_train_env._env.adversaries if has_adversaries else None
        # TODO: add preprocess_obs flag to config, maybe rename to "agent_ID"
        # NOTE: added preprocess_obs=False to avoid adding agent_ids to each obs
        # NOTE: this has the side effect of also removing any zero-padding to
        # standardize obs dimensions across agents, may be issue later
        wrapped_env = CTRolloutManager(log_train_env, batch_size=config["NUM_ENVS"], preprocess_obs=False, training_agents=training_agents)
        test_env = CTRolloutManager(log_test_env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False, training_agents=training_agents) # batched env for testing (has different batch size), chooses fixed team compositions on reset (may be different from train set)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in wrapped_env.training_agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, log_train_env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(wrapped_env.training_agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones, infos)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree.map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # INIT NETWORK
        # init agent
        if not config["AGENT_RECURRENT"]:
            if not config["AGENT_HYPERAWARE"]:
                agent = AgentMLP(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
            else:
                exit("HyperMLP deprecated currently!") # TODO: to fix, pass in AGENT_HYPERNET_KWARGS
                # agent = AgentHyperMLP(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'], hypernet_hidden_dim=config["AGENT_HYPERNET_KWARGS"]["HIDDEN_DIM"], hypernet_init_scale=config["AGENT_HYPERNET_KWARGS"]["INIT_SCALE"], dim_capabilities=log_train_env.dim_capabilities)
        else:
            if not config["AGENT_HYPERAWARE"]:
                agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
            else:
                agent = AgentHyperRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'], hypernet_kwargs=config["AGENT_HYPERNET_KWARGS"], dim_capabilities=log_train_env.dim_capabilities)

        rng, _rng = jax.random.split(rng)

        if config["PARAMETERS_SHARING"]:
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
            agent_params = agent.init(_rng, init_hs, init_x)
        else:
            init_x = (
                jnp.zeros((len(wrapped_env.training_agents), 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((len(wrapped_env.training_agents), 1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents),  1) # (n_agents, batch_size, hidden_dim)
            rngs = jax.random.split(_rng, len(wrapped_env.training_agents)) # a random init for each agent
            agent_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

        # log agent param count
        agent_param_count = sum(x.size for x in jax.tree_util.tree_leaves(agent_params))
        wandb.log({"agent_param_count": agent_param_count})
        print("-" * 10)
        print("DETAILED AGENT PARAM COUNT:")
        for name, param in jax.tree_util.tree_flatten_with_path(agent_params)[0]:
            print(f"{name}: {param.shape}")
        print("-" * 10)

        # init mixer
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((len(wrapped_env.training_agents), 1, 1))
        state_size = sample_traj.obs['__all__'].shape[-1]  # get the state shape from the buffer
        init_state = jnp.zeros((1, 1, state_size))
        mixer = MixingNetwork(config['MIXER_EMBEDDING_DIM'], config["MIXER_HYPERNET_HIDDEN_DIM"], config['MIXER_INIT_SCALE'])
        mixer_params = mixer.init(_rng, init_x, init_state)

        # init optimizer
        network_params = frozen_dict.freeze({'agent':agent_params, 'mixer':mixer_params})
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=None,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_network_params = jax.tree.map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        if config["PARAMETERS_SHARING"]:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # concatenate agents and parallel envs to process them in one batch
                agents, flatten_agents_obs = zip(*obs.items())
                original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
                batched_input = (
                    jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                    jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
                )
                hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
                q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
                q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
                return hidden_state, q_vals
        else:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
                agents, flatten_agents_obs = zip(*obs.items())
                batched_input = (
                    jnp.stack(flatten_agents_obs, axis=0), # (n_agents, time_step, n_envs, obs_size)
                    jnp.stack([dones[agent] for agent in agents], axis=0), # ensure to not pass other keys (like __all__)
                )
                # computes the q_vals with the params of each agent separately by vmapping
                hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
                q_vals = {a:q_vals[i] for i,a in enumerate(agents)}
                return hidden_state, q_vals


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_network_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, viz_env_states, rng = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in wrapped_env.training_agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent netwoek
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents)*config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents), config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)

            step_state = (
                train_state.params['agent'],
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
                time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def _loss_fn(params, target_network_params, init_hstate, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in wrapped_env.training_agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = homogeneous_pass(params['agent'], init_hstate, obs_, learn_traj.dones)
                _, target_q_vals = homogeneous_pass(target_network_params['agent'], init_hstate, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree.map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q value of the greedy actions for each agent
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree.map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
                    target_q_vals,
                    jax.lax.stop_gradient(valid_q_vals)
                )

                # compute q_tot with the mixer network
                chosen_action_qvals_mix = mixer.apply(
                    params['mixer'],
                    jnp.stack(list(chosen_action_qvals.values())),
                    learn_traj.obs['__all__'][:-1] # avoid last timestep
                )
                target_max_qvals_mix = mixer.apply(
                    target_network_params['mixer'],
                    jnp.stack(list(target_max_qvals.values())),
                    learn_traj.obs['__all__'][1:] # avoid first timestep
                )

                # compute target
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            target_qs,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*(1-done)*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals_mix[-1] * (1-learn_traj.dones['__all__'][-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (learn_traj.rewards['__all__'][-2::-1], learn_traj.dones['__all__'][-2::-1], target_max_qvals_mix[-1::-1])
                    )
                    targets = td_targets[::-1]
                    loss = jnp.mean(0.5*((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2))
                else:
                    # standard DQN loss
                    targets = (
                        learn_traj.rewards['__all__'][:-1]
                        + config['GAMMA']*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_mix
                    )
                    loss = jnp.mean((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2)

                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
            learn_traj = jax.tree.map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                learn_traj
            ) # (max_time_steps, batch_size, ...)
            if config["PARAMETERS_SHARING"]:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents)*config["BUFFER_BATCH_SIZE"]) # (n_agents*batch_size, hs_size)
            else:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(wrapped_env.training_agents), config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_network_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in wrapped_env.training_agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree.map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_network_params,
                operand=None
            )

            # update the greedy rewards/metrics/viz
            rng, _rng = jax.random.split(rng)
            test_results = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params['agent'], time_state),
                lambda _: {"metrics": test_metrics, "viz_env_states": viz_env_states},
                operand=None
            )
            test_metrics, viz_env_states = test_results["metrics"], test_results["viz_env_states"]

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards),
                'eps': explorer.get_epsilon(time_state['timesteps'])
            }
            metrics['test_metrics'] = test_metrics # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics, infos):
                    # NOTE: these do not work for MPE envs
                    # info_metrics = {
                    #     k:v[...,0][infos["returned_episode"][..., 0]].mean()
                    #     for k,v in infos.items() if k!="returned_episode"
                    # }
                    if env_name == "MPE_simple_fire":
                        wandb.log(
                            {
                                "returns": metrics['rewards']['__all__'].mean(),
                                "timestep": metrics['timesteps'],
                                "updates": metrics['updates'],
                                "loss": metrics['loss'],
                                'epsilon': metrics['eps'],
                                # **info_metrics,
                                **{k:v.mean() for k, v in metrics['test_metrics'].items()}
                            }
                        )
                    elif env_name == "MPE_simple_transport":
                        info_metrics = {
                            'quota_met': jnp.max(infos['quota_met'], axis=0).mean(),
                            'makespan': jnp.min(infos['makespan'], axis=0).mean(),
                        }
                        wandb.log(
                            {
                                "returns": metrics['rewards']['__all__'].mean(),
                                "timestep": metrics['timesteps'],
                                "updates": metrics['updates'],
                                "loss": metrics['loss'],
                                'epsilon': metrics['eps'],
                                **info_metrics,
                                **{k:v.mean() for k, v in metrics['test_metrics'].items()}
                            }
                        )
                jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state,
                target_network_params,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                viz_env_states,
                rng,
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params, time_state):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in test_env.training_agents}
                obs_   = jax.tree.map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree.map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, test_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos, env_state.env_state, obs, hstate) # save all EnvState (not LogEnvState) to visualize
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in test_env.training_agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(test_env.training_agents)*config["NUM_TEST_EPISODES"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(test_env.training_agents), config["NUM_TEST_EPISODES"]) # (n_agents, n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos, viz_env_states, obs, hstate) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )

            # get snd, NOTE: dim_c multiplier is currently hardcoded since it works for both fire and transport
            snd_value = snd(rollouts=obs, hiddens=hstate, dim_c=len(test_env.training_agents)*2, params=params, alg='qmix' if config["PARAMETERS_SHARING"] else 'qmix_ns', agent=agent)

            def fire_env_metrics(final_env_state):
                """
                Return success rate (pct of envs where both fires are put out)
                and percent of fires which are put out, out of all fires.
                """
                p_pos = final_env_state.p_pos
                rads = final_env_state.rad

                num_agents = len(test_env.training_agents)
                num_landmarks = rads.shape[-1] - num_agents
                num_envs = config["NUM_TEST_EPISODES"]

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

            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = dones['__all__']
            first_returns = jax.tree.map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
            first_infos   = jax.tree.map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)

            final_env_state = step_state[1].env_state
            fire_env_metrics = fire_env_metrics(final_env_state)
            if env_name == "MPE_simple_fire":
                metrics = {
                    'test_returns': first_returns['__all__'],# episode returns
                    'test_fire_success_rate': fire_env_metrics[0],
                    'test_pct_fires_put_out': fire_env_metrics[1],
                    'test_snd': snd_value,
                    **{'test_'+k:v for k,v in first_infos.items()},
                }
            elif env_name == "MPE_simple_transport":
                info_metrics = {
                    'quota_met': jnp.max(infos['quota_met'], axis=0),
                    'makespan': jnp.min(infos['makespan'], axis=0)
                }
                metrics = {
                    'test_returns': first_returns['__all__'],# episode returns
                    # NOTE: only works for simple spread
                    # 'test_pct_landmarks_covered': pct_landmarks_covered(step_state),
                    'test_snd': snd_value,
                    **{'test_'+k:v for k,v in info_metrics.items()},
                }
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], first_returns['__all__'].mean())
            return {"metrics": metrics, "viz_env_states": viz_env_states}

        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        greedy_ret = get_greedy_metrics(_rng, train_state.params['agent'],time_state) # initial greedy metrics
        test_metrics, viz_env_states = greedy_ret["metrics"], greedy_ret["viz_env_states"]

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_network_params,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            viz_env_states,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state, 'metrics':metrics}

    return train

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'qmix_{"ps" if config["alg"].get("PARAMETERS_SHARING", True) else "ns"}'

    # smac init neeeds a scenario
    if 'smax' in env_name.lower():
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        orig_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = SMAXLogWrapper(orig_env)
        # TODO: add test_env to smax
    else: # assuming MPE (Overcooked won't work with current pipeline)
        train_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        log_train_env = LogWrapper(train_env)
        viz_test_env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'], test_env_flag=True)
        log_test_env = LogWrapper(viz_test_env)

    config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", train_env.max_steps) # default steps defined by the env

    hyper_tag = "hyper" if config["alg"]["AGENT_HYPERAWARE"] else "normal"
    recurrent_tag = "RNN" if config["alg"]["AGENT_RECURRENT"] else "MLP"
    aware_tag = "aware" if config["env"]["ENV_KWARGS"]["capability_aware"] else "unaware"

    wandb_tags = [
        alg_name.upper(),
        env_name,
        hyper_tag,
        recurrent_tag,
        aware_tag,
        "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
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

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config["alg"], log_train_env, log_test_env, viz_test_env, env_name=config["env"]["ENV_NAME"])))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    if config['SAVE_PATH'] is not None:

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = outs['runner_state'][0]
        params = jax.tree.map(lambda x: x[0], model_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')

        if config["VISUALIZE_FINAL_POLICY"]:
            viz_env_states = outs['runner_state'][-2]

            # build a list of states manually from vectorized seq returned by
            # make_train() for desired seeds/envs
            for seed in range(config["NUM_SEEDS"]):
                for env in range(config["VIZ_NUM_ENVS"]):
                    state_seq = []
                    for i in range(config["alg"]["NUM_STEPS"]):
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
                    visualizer = MPEVisualizer(viz_test_env, state_seq, env_name=config["env"]["ENV_NAME"])
                    video_fpath = f'{save_dir}/{alg_name}-seed-{seed}-rollout.gif'
                    visualizer.animate(video_fpath)
                    wandb.log({f"env-{env}-seed-{seed}-rollout": wandb.Video(video_fpath)})

    # force multiruns to finish correctly
    wandb.finish()

if __name__ == "__main__":
    main()

