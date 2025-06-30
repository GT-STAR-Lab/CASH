import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Dict, List
from functools import partial
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from gymnax.environments.spaces import Box

class SimpleFireMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=3,
        num_landmarks=2,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=2,
        **kwargs,
    ):
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities
        self.dim_capabilities = num_agents * num_capabilities

        self.id_aware = kwargs.get("id_aware", False)
        if self.id_aware:
            self.dim_capabilities = 5 * num_agents

        # components of the observation
        pos_dim = num_agents * 2
        vel_dim = 2 # only ego agent
        fire_pos_dim = num_landmarks * 2
        fire_rad_dim = num_landmarks
        observation_spaces = {
            i:Box(-jnp.inf, jnp.inf, (pos_dim + vel_dim + self.dim_capabilities + fire_pos_dim + fire_rad_dim))
            for i in agents
        }

        # agent color, landmark color
        self.colour = [(115, 243, 115)] * num_agents + [(255, 64, 64)] * num_landmarks

        # env specific parameters
        self.test_teams = jnp.array(kwargs["test_teams"]) if "test_teams" in kwargs else None
        self.fire_rad_range = kwargs["fire_rad_range"] if "fire_rad_range" in kwargs else [0.2, 0.3]

        # reward shaping
        self.fire_out_reward = kwargs["fire_out_reward"] if "fire_out_reward" in kwargs else 1
        self.uncovered_penalty_factor = kwargs["uncovered_penalty_factor"] if "uncovered_penalty_factor" in kwargs else 2
        self.pos_shaping_factor = kwargs["pos_shaping"] if "pos_shaping" in kwargs else 0.01

        # no collisions in this env
        collide = jnp.concatenate(
            [jnp.full((num_agents+num_landmarks), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=num_landmarks,
            landmarks=landmarks,
            action_type=action_type,
            observation_spaces=observation_spaces,
            dim_c=0, # no comm
            colour=self.colour,
            # NOTE: modified via reset(), see below
            # rad=rad,
            collide=collide,
            **kwargs,
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        def _obs(aidx: int):
            def shift_array(arr, i):
                """
                Assuming arr is 2D, moves row i to the front
                """
                i = i % arr.shape[0]
                first_part = arr[i:]
                second_part = arr[:i]
                return jnp.concatenate([first_part, second_part])

            # move ego_pos to front of agent_pos, then remove
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = shift_array(agent_pos, aidx)
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]
            rel_other_pos = other_pos - ego_pos # and transform to relative pos

            ego_vel = state.p_vel[aidx, :]

            other_cap = jnp.stack([
                state.accel.flatten(), state.rad[:self.num_agents].flatten(), # landmark rad is included in state.rad
            ], axis=-1)
            # jax.debug.print("other cap {} accel {} rad {}", other_cap, state.accel, state.rad[:self.num_agents])

            ego_cap = other_cap[aidx, :]
            # roll to remove ego agent
            other_cap = jnp.roll(other_cap, shift=self.num_agents - aidx - 1, axis=0)[:self.num_agents-1, :]

            # mask out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                if not self.id_aware:
                    other_cap = jnp.full(other_cap.shape, MASK_VAL)
                    ego_cap = jnp.full(ego_cap.shape, MASK_VAL)
                else:
                    ego_cap = state.agent_ids[aidx]
                    other_cap = jnp.roll(state.agent_ids, shift=self.num_agents - aidx - 1, axis=0)[:self.num_agents-1]

            # give agents the pos and rad of all landmarks (fires)
            landmark_p_pos = state.p_pos[self.num_agents:]
            # transform to relative pos
            rel_landmark_p_pos = landmark_p_pos - ego_pos
            landmark_rads = state.rad[self.num_agents:]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 2
                rel_other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                rel_landmark_p_pos.flatten(), # 2, 2
                landmark_rads.flatten(), # 1, 2
                # NOTE: caps must go last for hypernet logic
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
            ])

            return obs

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Goal is to put out fires as quickly as possible.

        Fires are put out when the sum of all robot firefighting capacity on the fire is greater than the strength of the fire.
        """
        global_rew = 0

        # reward team when fires are being put out, penalize otherwise
        def _agent_in_range(agent_i: int, landmark_pos, landmark_rad):
            delta_pos = state.p_pos[agent_i] - landmark_pos
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
            return (dist < landmark_rad)

        agent_rads = state.rad[:self.num_agents]
        landmark_rads = state.rad[self.num_agents:]
        landmark_p_pos = state.p_pos[self.num_agents:]
        for i, landmark_pos in enumerate(landmark_p_pos):
            landmark_rad = landmark_rads[i]

            # take the sum of all agents ff on this landmark
            agents_on_landmark = jax.vmap(_agent_in_range, in_axes=[0, None, None])(self.agent_range, landmark_pos, landmark_rad)
            firefighting_level = jnp.sum(jnp.where(agents_on_landmark, agent_rads, 0))

            # dense rew for firefighting
            enough_firefight = firefighting_level >= landmark_rads[i]
            # NOTE: reward based on how much of fire is covered, but add a
            # ceil() so agents are encouraged to spread out
            #
            # (2nd term is <0 since !enough_firefight means ff_level < landmark_rads)
            ff_rew = jnp.where(enough_firefight, self.fire_out_reward,
                               self.uncovered_penalty_factor*(firefighting_level-landmark_rads[i]))

            global_rew += ff_rew
            # only add reward if this fire is valid (rad > 0)
            # global_rew = jnp.where(landmark_rad > 0, global_rew+ff_rew, global_rew)

        # normalize global rew based on how many active fires there are
        # active_fires = jnp.count_nonzero(landmark_rads > 0)
        # global_rew /= active_fires

        # reward each agent for getting closer to one of the landmarks
        def _dist_to_landmarks(agent_pos):
            dist_to_landmarks = jnp.linalg.norm(agent_pos - landmark_p_pos, axis=1)
            return dist_to_landmarks

        def _agent_rew(agent_i):
            agent_pos = state.p_pos[agent_i]
            dists = _dist_to_landmarks(agent_pos)
            return -jnp.min(dists)

        rew = {
            a: global_rew + (self.pos_shaping_factor * _agent_rew(i))
            for i, a in enumerate(self.agents)
        }
        return rew

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """Overriding superclass simple.py"""
        # NOTE: copy-pasted from simple.py, bad practice

        key_a, key_l, key_c, key_fr = jax.random.split(key, num=4)

        # spawn landmarks (fires) s.t. they don't overlap
        landmark_rads = jax.random.uniform(key_fr, (self.num_landmarks,), minval=self.fire_rad_range[0], maxval=self.fire_rad_range[1])

        def _spawn_one_fire(carry, _):
            key_l, existing_fires, fire_index = carry

            # spawn a new random fire
            key_l, key_fs = jax.random.split(key_l)
            new_fire = jax.random.uniform(
                key_fs, (1, 2), minval=-1.0, maxval=+1.0
            )

            # it's valid if it doesn't intersect with any existing fires (that fire's rad + new fire rad)
            dists = jnp.linalg.norm(existing_fires - new_fire, axis=1)
            new_fire_valid = jnp.all(dists > (landmark_rads + landmark_rads[fire_index]))

            new_fire = jnp.reshape(new_fire, (2,))
            new_fire_added = existing_fires.at[fire_index, :].set(new_fire)

            # if new fire spawn is valid, add it to the fire list, and incr the fire index
            new_fires = jax.lax.cond(
                new_fire_valid,
                lambda: new_fire_added, # T
                lambda: existing_fires, # F
            )
            fire_index = jax.lax.cond(
                new_fire_valid,
                lambda: fire_index+1,
                lambda: fire_index,
            )
            return (key_l, new_fires, fire_index), None

        # use jax.lax.scan to spawn N more fires, each of which does not collide with any prev
        key_l, key_fs, key_tt = jax.random.split(key_l, 3)
        # set the init spawns to be far from the actual spawns, s.t. array shape can be maintained in the carry
        init_fires = jax.random.uniform(
            key_fs, (self.num_landmarks, 2), minval=-100.0, maxval=-100.0
        )
        initial_state = (key_l, init_fires, 0)
        MAX_ITERS = 10*self.num_landmarks # try ~10 times for each fire, each time
        (key_l, landmark_p_pos, _), _ = jax.lax.scan(_spawn_one_fire, initial_state, None, length=MAX_ITERS)

        p_pos = jnp.concatenate(
            [
                # spawn agents in the same spot (center), like a real fire depot
                jnp.zeros((self.num_agents, 2)),
                # OLD: spawn randomly, in same range as fires
                # jax.random.uniform(
                #     key_a, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                # ),
                landmark_p_pos,
            ]
        )

        if self.independent_agents:
            # if independent policies do not sample teams and capabilities, keep constant
            # NOTE: assumes that agent_rad and agent_accels are n_agent length
            agent_rads = self.agent_rads
            agent_accels = self.agent_accels
        else:
            # randomly sample N_agents' capabilities from the possible agent pool (hence w/out replacement)
            selected_agents = jax.random.choice(key_c, self.num_agents, shape=(self.num_agents,), replace=False)
            agent_rads = self.agent_rads[selected_agents]
            agent_accels = self.agent_accels[selected_agents]
            agent_ids = self.agent_ids[selected_agents]
            # unless a test distribution is provided and this is a test_env
            if self.test_env_flag and self.test_teams is not None:
                # pick one of the test teams at random
                selected_team = jax.random.choice(key_tt, self.test_teams.shape[0], shape=(1,))
                test_team = self.test_teams[selected_team].squeeze()
                agent_rads = test_team[0::2]
                agent_accels = test_team[1::2]

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.05
                [agent_rads, landmark_rads]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
            agent_ids=agent_ids
        )

        return self.get_obs(state), state

def generate_teams(seed):
    import numpy as np
    np.random.seed(seed)
    small_fire_rad = np.random.uniform(0.1, 0.2, 2)
    big_fire_rad = np.random.uniform(0.3, 0.4, 1)
    accel = np.random.uniform(1, 3, 3)
    print("Train Agents:")
    print("rad:\n", np.round(np.concatenate((small_fire_rad, big_fire_rad, small_fire_rad, big_fire_rad)), decimals=2))
    print("accel:\n", np.round(np.concatenate((accel, accel)), decimals=2))
    print()

    N_test_teams = 10
    smaller_fire_rad = np.random.uniform(0.05, 0.094, N_test_teams)
    med_fire_rad = 0.25 - smaller_fire_rad
    bigger_fire_rad = np.random.uniform(0.4, 0.5, N_test_teams)

    small_accel = np.random.uniform(3, 3.5, N_test_teams)
    med_accel = np.random.uniform(1, 3, N_test_teams)
    big_accel = np.random.uniform(0.5, 1, N_test_teams)

    # tie caps together by agent
    small_agents = np.stack((smaller_fire_rad, small_accel)).T
    med_agents = np.stack((med_fire_rad, med_accel)).T
    big_agents = np.stack((bigger_fire_rad, big_accel)).T

    # then group teams, transpose to be N_teams first
    test_teams = np.round(np.stack((small_agents, med_agents, big_agents)).transpose((1, 0, 2)), decimals=2)

    print("Test Teams:")
    for i in range(N_test_teams):
        test_team = test_teams[i, ...].squeeze()
        print(test_team.flatten())
    print()

if __name__ == "__main__":
    SEED = 76
    generate_teams(SEED)
