import jax
import jax.numpy as jnp
import chex
from functools import partial
from gymnax.environments.spaces import Box
from jaxmarl.environments.mpe.simple import SimpleMPE, State
from jaxmarl.environments.mpe.default_params import *
from typing import Tuple, Dict

class SimpleTransportMPE(SimpleMPE):
    def __init__(
        self,
        num_agents=4,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=2,
        **kwargs,
    ):
        agents = ["agent_{}".format(i) for i in range(num_agents)]

        # 3 landmarks, [concrete depo, lumber depo, construction site]
        self.num_landmarks = 3
        landmarks = ["landmark_{}".format(i) for i in range(3)]

        self.num_agents = num_agents
        self.num_neighbors = kwargs.get("num_neighbors", num_agents-1)
        self.capability_aware = capability_aware
        self.num_capabilities = num_capabilities
        self.dim_capabilities = (self.num_neighbors+1) * num_capabilities
        self.test_team_capacities = kwargs.get("test_team_capacities", None)
        if self.test_team_capacities is not None:
            self.test_team_capacities = jnp.array(self.test_team_capacities)

        # observation dimensions
        pos_dim = self.num_neighbors * 2
        vel_dim = 2  # for ego agent
        material_depot_dim = 2 * 2 # 2 materials, 2 positions
        construction_site_dim = 2
        quota_dim = 2
        payload_dim = 2

        # initialize observation space for each agent
        observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (pos_dim + vel_dim + material_depot_dim + construction_site_dim + payload_dim + quota_dim + self.dim_capabilities))
            for i in agents
        }

        # overriden in reset to reflect max capability
        self.colour = [(169, 169, 169)] * num_agents + [(0, 0, 255), (0, 255, 0), (128,0,128)]

        # reward shaping terms
        self.concrete_pickup_reward = kwargs.get("concrete_pickup_reward", 0.25)
        self.lumber_pickup_reward = kwargs.get("lumber_pickup_reward", 0.25)
        self.dropoff_reward = kwargs.get("dropoff_reward", 0.75)
        self.quota_penalty = kwargs.get("quota_penalty", -0.005)

        # no collisions in this env
        collide = jnp.concatenate(
            [jnp.full((num_agents+len(landmarks)), False)]
        )

        super().__init__(
            num_agents=num_agents,
            agents=agents,
            num_landmarks=len(landmarks),
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

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """
        Override simple.py step_env to update payload information
        """
        u, c = self.set_actions(state, actions)
        if (
            c.shape[1] < self.dim_c
        ):  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1
            )

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)
        done = jnp.full((self.num_agents), state.step >= self.max_steps)

        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        reward = self.rewards(state)

        #######################################################################################
        # Start modified step
        # only update payloads after rewards are applied (I think this makes sense?)
        #######################################################################################
        # get distance between agents and landmarks
        agent_p_pos = state.p_pos[:self.num_agents]
        landmark_p_pos = state.p_pos[self.num_agents:]
        relative_positions = agent_p_pos[:, None, :] - landmark_p_pos[None, :, :]
        dists = jnp.linalg.norm(relative_positions, axis=-1)

        # get mask of agents within landmark radius
        landmark_rads = state.rad[self.num_agents:]
        mask = dists <= landmark_rads

        def io_callback(x, i):
            if x[0][0].item() == 1.0:
                print(f"payload {i}: {x[0][0].item()}")

        able_to_load = jnp.bitwise_and(state.payload[:, 0] == 0., state.payload[:, 1] == 0.)
        # update payload for agents on concrete depot
        payload_concrete = jnp.where(jnp.bitwise_and(mask[:, 0], able_to_load), state.capacity[:, 0], state.payload[:, 0])
        # jax.debug.callback(io_callback, payload, 0)

        # update payload for agents on lumber depot
        payload_lumber = jnp.where(jnp.bitwise_and(mask[:, 1], able_to_load), state.capacity[:, 1], state.payload[:, 1])
        # jax.debug.callback(io_callback, payload, 1)

        # update quotas
        concrete_delivered = jnp.where(jnp.bitwise_and(mask[:, 2], (state.payload[:, 0] > 0.)), state.payload[:, 0], jnp.zeros_like(state.payload[:, 0]))
        lumber_delivered = jnp.where(jnp.bitwise_and(mask[:, 2], (state.payload[:, 1] > 0.)), state.payload[:, 1], jnp.zeros_like(state.payload[:, 1]))
        quota_concrete = jnp.where(state.site_quota[0] < 0, state.site_quota[0] + jnp.sum(concrete_delivered), jnp.zeros_like(state.site_quota[0]))
        quota_lumber = jnp.where(state.site_quota[1] < 0, state.site_quota[1] + jnp.sum(lumber_delivered), jnp.zeros_like(state.site_quota[1]))

        # reset payload for agents on construction site
        able_to_dropoff = jnp.bitwise_and(mask[:, 2], state.site_quota[0] < 0)
        payload_concrete = jnp.where(jnp.bitwise_and(able_to_dropoff, (payload_concrete > 0.)), jnp.zeros_like(payload_concrete), payload_concrete)
        able_to_dropoff = jnp.bitwise_and(mask[:, 2], state.site_quota[1] < 0)
        payload_lumber = jnp.where(jnp.bitwise_and(able_to_dropoff, (payload_lumber > 0.)), jnp.zeros_like(payload_lumber), payload_lumber)
        # jax.debug.callback(io_callback, payload, 2)

        payload = jnp.concatenate([payload_concrete.reshape(-1,1), payload_lumber.reshape(-1,1)], axis=-1)
        site_quota = jnp.array([quota_concrete, quota_lumber])

        state = state.replace(
            payload=payload,
            site_quota=site_quota
        )
        # jax.debug.callback(io_callback, state.site_quota, 2)
        #######################################################################################
        # End modified step
        #######################################################################################

        obs = self.get_obs(state)

        # mask to indicate if quota has been met
        quota_done = jnp.bitwise_and(state.site_quota[0] >= 0, state.site_quota[1] >= 0).reshape(-1,1)

        # mask to log makespan if quota has been met, otherwise just trivially set to max steps
        makespan = jnp.where(quota_done, state.step, self.max_steps)

        info = {
            "quota_met": quota_done,
            "makespan": makespan,
        }

        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})

        return obs, state, reward, dones, info

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        def _obs(aidx: int):
            def get_nearest_neighbors(aidx):
                """
                Returns the indices of the n nearest neighbors to agent aidx
                """
                agent_pos = state.p_pos[:self.num_agents]
                dists = jnp.linalg.norm(agent_pos - agent_pos[aidx], axis=1)
                nearest_indices = jnp.argsort(dists)[1:self.num_neighbors+1]
                return nearest_indices

            def shift_array(arr, i):
                """
                Assuming arr is 2D, moves row i to the front
                """
                i = i % arr.shape[0]
                first_part = arr[i:]
                second_part = arr[:i]
                return jnp.concatenate([first_part, second_part])
            
            # get neighbor indices
            nbr_idxs = get_nearest_neighbors(aidx)

            # agent positions, move ego_pos to front of agent_pos, then remove
            agent_pos = state.p_pos[:self.num_agents, :]
            other_pos = state.p_pos[nbr_idxs, :]
            ego_pos = other_pos[0]
            other_pos = other_pos[1:]
            rel_other_pos = other_pos - ego_pos # and transform to relative pos

            # ego agent velocities
            ego_vel = state.p_vel[aidx, :]

            # agent capabilities, separate ego capability from other agents capability
            other_cap = state.capacity
            ego_cap = other_cap[aidx, :]
            other_cap = other_cap[nbr_idxs, :]

            # mask out capabilities for non-capability-aware baselines
            if not self.capability_aware:
                other_cap = jnp.full(other_cap.shape, MASK_VAL)
                ego_cap = jnp.full(ego_cap.shape, MASK_VAL)

            # relative position of all landmarks
            landmark_p_pos = state.p_pos[self.num_agents:]
            rel_landmark_p_pos = landmark_p_pos - ego_pos

            # current payload
            payload = state.payload[aidx, :]

            obs = jnp.concatenate([
                ego_pos.flatten(),  # 2
                rel_other_pos.flatten(),  # N-1, 2
                ego_vel.flatten(),  # 2
                rel_landmark_p_pos.flatten(), # 2, 2
                state.site_quota.flatten(), # 2
                payload.flatten(), # 2
                # NOTE: caps must go last for hypernet logic
                ego_cap.flatten(),  # n_cap
                other_cap.flatten(),  # N-1, n_cap
            ])

            return obs

        obs = {a: _obs(i) for i, a in enumerate(self.agents)}
        return obs

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Reward agents according to objective of maximizing delivered materials.
        """

        def _load_concrete_rew(agent_i):
            """
            Reward agent for loading concrete if payload is empty.
            """
            agent_pos = state.p_pos[agent_i]
            concrete_depot_pos = state.p_pos[-3]
            dist = jnp.array([jnp.linalg.norm(agent_pos - concrete_depot_pos)])
            able_to_load = jnp.bitwise_and(state.payload[agent_i][0] == 0, state.payload[agent_i][1] == 0)
            able_to_load = jnp.bitwise_and(able_to_load, state.capacity[agent_i][0] > 0)
            return jnp.bitwise_and(dist <= state.rad[-3], able_to_load)

        def _load_lumber_rew(agent_i):
            """
            Reward agent for loading lumber if payload is empty.
            """
            agent_pos = state.p_pos[agent_i]
            lumber_depot_pos = state.p_pos[-2]
            dist = jnp.array([jnp.linalg.norm(agent_pos - lumber_depot_pos)])
            able_to_load = jnp.bitwise_and(state.payload[agent_i][0] == 0, state.payload[agent_i][1] == 0)
            able_to_load = jnp.bitwise_and(able_to_load, state.capacity[agent_i][1] > 0)
            return jnp.bitwise_and(dist <= state.rad[-2], able_to_load)

        def _dropoff_rew(agent_i):
            """
            Reward agent for dropping of materials if payload is non empty and quota is not met
            """
            agent_pos = state.p_pos[agent_i]
            construction_site_pos = state.p_pos[-1]
            dist = jnp.array([jnp.linalg.norm(agent_pos - construction_site_pos)])
            able_to_dropoff = jnp.bitwise_and(dist <= state.rad[-1], state.site_quota < 0)

            # Return both summed and unsummed value for quota reward logic
            return jnp.sum(jnp.bitwise_and(able_to_dropoff, state.payload[agent_i] > 0), axis=-1), jnp.bitwise_and(able_to_dropoff, state.payload[agent_i] > 0)

        def _dist_to_landmarks(agent_pos):
            landmark_p_pos = state.p_pos[self.num_agents:]
            dist_to_landmarks = jnp.linalg.norm(agent_pos - landmark_p_pos, axis=1)
            return dist_to_landmarks

        def _pos_rew(agent_i):
            agent_pos = state.p_pos[agent_i]
            dists = _dist_to_landmarks(agent_pos)
            return -jnp.min(dists)

        rew = {
            a: (self.concrete_pickup_reward * _load_concrete_rew(i) +
                self.lumber_pickup_reward * _load_lumber_rew(i) +
                self.dropoff_reward * _dropoff_rew(i)[0]
                # 0.005 * _pos_rew(i)
                )[0]
            for i, a in enumerate(self.agents)
        }

        # # get progress towards quota
        quota_step = jnp.sum(jnp.array([_dropoff_rew(i)[1] * state.payload[i] for i in range(self.num_agents)]), axis=0)
        quota_new = state.site_quota + quota_step

        # # if quota is met, stop applying penalty, otherwise, apply penalty
        quota_rew = jnp.where(jnp.all(quota_new >= 0), -2*self.quota_penalty, self.quota_penalty)
        rew = {a: rew[a] + quota_rew for a in rew}

        return rew

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, State]:
        """
        Override reset in simple.py to fix the location of the depots and construction site.
        """

        key_a, key_t, key_q = jax.random.split(key, 3)

        p_pos = jnp.concatenate(
            [
                jnp.zeros((self.num_agents, 2)),
                jnp.array(
                    [
                        [-0.5, 0.5],
                        [0.5, 0.5],
                        [0.0, -0.5],
                    ]
                ),
            ]
        )

        agent_rads = self.agent_rads
        agent_accels = self.agent_accels

        if self.independent_agents:
            # if independent policies do not sample teams and capabilities, keep constant
            # NOTE: assumes that agent_capacities is n_agent length
            agent_capacities = self.agent_capacities
        else:
            # randomly sample a team from the capacity team pool
            selected_team = jax.random.choice(key_a, self.agent_capacities.shape[0], shape=(1,))
            agent_capacities = self.agent_capacities[selected_team].squeeze()
            # if a test distribution is provided and this is a test_env, override capacities
            # NOTE: also add other capabilities here?
            if self.test_env_flag and self.test_team_capacities is not None:
                selected_team = jax.random.choice(key_t, self.test_team_capacities.shape[0], shape=(1,))
                agent_capacities = self.test_team_capacities[selected_team].squeeze()

        # initialize with empty payload or a payload corresponding to capacity
        # payload = jnp.where(
        #     jax.random.uniform(key_l, (self.num_agents, 1)) < 0.5,
        #     0,
        #     jnp.take_along_axis(agent_capacities, jax.random.randint(key_l, (self.num_agents, 1), minval=0, maxval=2), axis=1)
        # )

        self.site_quota = -jax.random.uniform(key_q, (2,), minval=0.125*self.num_agents, maxval=0.25*self.num_agents)
        # if self.test_env_flag:
        #     self.site_quota = -jax.random.uniform(key_q, (2), minval=0.5*self.num_agents, maxval=0.75*self.num_agents)

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            accel=agent_accels,
            rad=jnp.concatenate(
                # NOTE: here, must define landmark rad as well, by default landmarks are 0.30
                [agent_rads, jnp.full((self.num_landmarks), 0.30)]
            ),
            done=jnp.full((self.num_agents), False),
            step=0,
            payload=jnp.zeros((self.num_agents, 2)),
            capacity=agent_capacities,
            site_quota=self.site_quota
        )

        return self.get_obs(state), state

def unit_test():
    key = jax.random.PRNGKey(0)

    # Initialize environment with default settings
    env_kwargs = {
        'agent_rads': [0.2, 0.2, 0.2],
        'agent_accels': [2, 2, 2],
        'agent_capacities': [[1.0, 0.0],
                            [0.0, 1.0],
                            [0.5, 0.5]],
        'site_quota': [5., 5.],
        'quota_penalty': -0.005
    }
    env = SimpleTransportMPE(
        num_agents=3,
        action_type=DISCRETE_ACT,
        capability_aware=True,
        num_capabilities=2,
        **env_kwargs,
    )
    obs, state = env.reset(key)

    assert obs is not None, "reset failed to return obs"
    assert state is not None, "reset failed to return state"

    # initialize agent just outside of depot
    concrete_depot_pos = state.p_pos[-3]
    agent_pos = concrete_depot_pos + jnp.array([0.3 + 0.01, 0.0])
    state = state.replace(p_pos=state.p_pos.at[0].set(agent_pos))
    assert (state.p_pos[0] == agent_pos).all(), f"FAIL: expected {agent_pos}, got {state.p_pos[0]}"

    # check that action that doesn't place agent in depot results in no reward
    actions = {f"agent_{i}": jnp.array([0]) for i in range(env.num_agents)}
    obs, state, reward, dones, info = env.step_env(key, state, actions)
    assert reward['agent_0'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 0 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_0']}"
    print("PASS: reward when agent has no payload and doesn't enter depot is 0!")

    # check that payloads and rewards are correct when a agent enters the depot
    prev_state = state
    for _ in range(2):
        actions = {f"agent_{i}": jnp.array([1]) for i in range(env.num_agents)}
        obs, state, reward, dones, info = env.step_env(key, state, actions)
    expected_payload = jnp.array([[1., 0.], [0., 0.], [0., 0.]])
    assert (state.p_pos != prev_state.p_pos).any(), f"FAIL: state before and after step with nonzero action match"
    assert reward['agent_0'] == 0.25 + 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 0 to be {0.25 - 2*env_kwargs['quota_penalty']}, got {reward['agent_0']}"
    assert reward['agent_1'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 1 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_1']}"
    assert reward['agent_2'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 2 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_2']}"
    assert (state.payload == expected_payload).all(), f"FAIL: expected payload to be{expected_payload}, got {state.payload}"
    print("PASS: rewards and payload update correctly when agent 0 has no payload and enters the depot")

    # check that payloads and rewards don't update if agent enters depot and already has a payload
    actions = {f"agent_{i}": jnp.array([0.0, 1.0, 0.0, 0.0, 0.0]) for i in range(env.num_agents)}
    obs, state, reward, dones, info = env.step_env(key, state, actions)
    expected_payload = jnp.array([[1., 0.], [0., 0.], [0., 0.]])
    assert reward['agent_0'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 0 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_0']}"
    assert reward['agent_1'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 1 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_1']}"
    assert reward['agent_2'] == 2*env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 2 to be {2*env_kwargs['quota_penalty']}, got {reward['agent_2']}"
    assert (state.payload == expected_payload).all(), f"FAIL: expected payload to be {expected_payload}, got {state.payload}"
    print("PASS: rewards and payload update correctly when agent 0 has a payload and enters the depot")

    # check that payloads and rewards update if agent enters constrction site and has a payload
    obs, state = env.reset(key)
    site_pos = state.p_pos[-1]
    agent_pos = site_pos + jnp.array([0.0, 0.3+0.01])
    for i in range(3):
        state = state.replace(p_pos=state.p_pos.at[i].set(agent_pos))
    state = state.replace(payload=jnp.array([[1., 0.], [0., 1.], [0., 0.]]))
    state = state.replace(site_quota=jnp.array([-5, 0]))
    prev_state = state
    for _ in range(2):
        actions = {f"agent_{i}": jnp.array([3]) for i in range(env.num_agents)}
        obs, state, reward, dones, info = env.step_env(key, state, actions)
    expected_payload = jnp.array([[0., 0.], [0., 1.], [0., 0.]])
    expected_quota = jnp.array([-4, 0])
    assert (state.p_pos != prev_state.p_pos).any(), f"FAIL: state before and after step with nonzero action match"
    assert reward['agent_0'] == 0.75 + env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 0 to be {0.75 + env_kwargs['quota_penalty']}, got {reward['agent_0']}"
    assert reward['agent_1'] == env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 1 to be {env_kwargs['quota_penalty']}, got {reward['agent_1']}"
    assert reward['agent_2'] == env_kwargs['quota_penalty'], f"FAIL: expected reward for agent 2 to be {env_kwargs['quota_penalty']}, got {reward['agent_2']}"
    assert (state.payload == expected_payload).all(), f"FAIL: expected payload to be {expected_payload}, got {state.payload}"
    assert (state.site_quota == expected_quota).all(), f"FAIL: expected quota to be {expected_quota}, got {state.site_quota}"
    print("PASS: rewards and payload update correctly when agent 0 has payload and enters the site")

def generate_teams(seed):
    # # Dictionary to track team occurrences
    # team_occurrences = {}

    # # Check for duplicates
    # for i, team in enumerate(teams):
    #     # Convert the team to a sorted tuple
    #     sorted_team = tuple(sorted(tuple(agent) for agent in team))

    #     # Track occurrences of each team
    #     if sorted_team in team_occurrences:
    #         team_occurrences[sorted_team].append(team)
    #         print(i)
    #         print(team)
    #     else:
    #         team_occurrences[sorted_team] = [team]

    # # Print duplicates
    # print("Duplicate teams:")
    # for team, occurrences in team_occurrences.items():
    #     if len(occurrences) > 1:
    #         print(f"Team {occurrences[0]} is duplicated {len(occurrences)} times.")


    import numpy as np
    np.random.seed(seed)

    agent_values = np.array([
        [0.0, 0.5], [0.1, 0.4], [0.2, 0.3], [0.3, 0.2], [0.4, 0.1], [0.5, 0.0]
    ])

    # Number of teams and team size
    num_teams = 10
    team_size = 4

    # To store unique teams
    teams = set()

    # Helper function to create a sorted tuple of the team to ensure uniqueness
    def create_team():
        team_indices = np.random.choice(agent_values.shape[0], team_size, replace=False)
        team = agent_values[team_indices].tolist()
        team = tuple(sorted(tuple(agent) for agent in team))
        return team

    # Generate 10 unique teams
    while len(teams) < num_teams:
        team_tuple = create_team()
        teams.add(team_tuple)

    # Convert back to array for easier handling
    teams = np.array([list([list(agent) for agent in team]) for team in teams])

    # Print the teams
    print("Train Teams:")
    for i, team in enumerate(teams):
        team = [list(agent) for agent in team]
        print(f" - {team}")

    # for test teams, sample cap_0 from range (0-1), set cap_1 to be (1-cap_0)
    N_test_teams = 10
    big_cap_0 = np.random.uniform(0, 1, N_test_teams*(team_size-1)).reshape(N_test_teams, (team_size-1))
    small_cap_0 = np.random.uniform(0, 0.25, N_test_teams).reshape(N_test_teams, 1)
    big_cap_1 = 1-big_cap_0
    small_cap_1 = 0.25-small_cap_0
    cap_0 = np.concatenate([big_cap_0, small_cap_0], axis=-1)
    cap_1 = np.concatenate([big_cap_1, small_cap_1], axis=-1)

    # tie caps together by agent, rearrange to shape [N_teams, N_agents, N_cap]
    test_teams = np.round(np.stack([cap_0, cap_1]).transpose(1, 2, 0), decimals=2)

    print("Test Teams:")
    for i in range(N_test_teams):
        test_team = test_teams[i, ...].squeeze()
        print("-", test_team.flatten())
    print()


if __name__ == "__main__":
    # unit_test()

    SEED = 76
    generate_teams(SEED)
