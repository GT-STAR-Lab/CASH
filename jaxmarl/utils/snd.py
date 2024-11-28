import jax
import jax.numpy as jnp

def homogeneous_pass_qmix(params, hidden_state, obs, dones, agent=None):
    """
    Copied and slightly modified from qmix.py
    """
    original_shape = obs.shape
    # concatenate agents and parallel envs to process them in one batch
    batched_input = (
        obs.reshape(1, obs.shape[0]*obs.shape[1], obs.shape[2]), # [1, n_envs*n_agents, obs_dim] for broadcasting with scanned
        dones.reshape(1, dones.shape[0]*dones.shape[1]) # [1, n_envs*n_agents] for broadcasting with scanned
    )

    hidden_state = hidden_state.reshape(hidden_state.shape[0]*hidden_state.shape[1], hidden_state.shape[2])
    hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)

    q_vals = q_vals.reshape(*original_shape[:-1], -1) # (time_steps, n_envs, n_agents, action_dim)

    return q_vals

def homogeneous_pass_qmix_ns(params, hidden_state, obs, dones, agent=None):
    original_shape = obs.shape
    # reshape and transpose observations for vmap
    obs = jnp.transpose(obs.reshape(1, obs.shape[0], obs.shape[1], obs.shape[2]), (2, 0, 1, 3))
    dones = jnp.transpose(dones.reshape(1, dones.shape[0], dones.shape[1]), (2, 0, 1))
    hidden_state = jnp.transpose(hidden_state, (1, 0, 2))

    batched_input = (obs, dones)
    hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
    q_vals = jnp.transpose(q_vals, (1, 2, 0, 3))    # transpose back
    q_vals = q_vals.squeeze(0)

    return q_vals

def homogeneous_pass_mappo(params, hidden_state, obs, dones, agent=None):
    """
    Forward pass for mappo, copied from mappo_rnn_mpe.py
    """
    original_shape = obs.shape
    # concatenate agents and parallel envs to process them in one batch
    batched_input = (
        obs.reshape(1, obs.shape[0]*obs.shape[1], obs.shape[2]), # [1, n_envs*n_agents, obs_dim] for broadcasting with scanned
        dones.reshape(1, dones.shape[0]*dones.shape[1]) # [1, n_envs*n_agents] for broadcasting with scanned
    )

    hidden_state = hidden_state.reshape(hidden_state.shape[0]*hidden_state.shape[1], hidden_state.shape[2])
    hidden_state, pi = agent.apply(params, hidden_state, batched_input)

    pi = pi.probs  # (time_steps, n_envs, n_agents, action_dim)
    pi = pi.reshape(*original_shape[:-1], -1)

    return pi

def homogeneous_pass_mappo_ns(params, hidden_state, obs, dones, agent=None):
    """
    Forward pass for mappo, copied from mappo_rnn_mpe.py
    """
    original_shape = obs.shape
    # reshape and transpose observations for vmap
    obs = jnp.transpose(obs.reshape(1, obs.shape[0], obs.shape[1], obs.shape[2]), (2, 0, 1, 3))
    dones = jnp.transpose(dones.reshape(1, dones.shape[0], dones.shape[1]), (2, 0, 1))
    hidden_state = jnp.transpose(hidden_state, (1, 0, 2))

    batched_input = (obs, dones)
    hidden_state, pi = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)

    pi = pi.probs  # (time_steps, n_envs, n_agents, action_dim)
    pi = pi.reshape(*original_shape[:-1], -1)

    return pi

def total_variational_distance(p, q):
    """
    Get the distance between two categorical distributions
    """

    return 0.5 * jnp.sum(jnp.abs(p - q), axis=-1)

def snd(rollouts, hiddens, dim_c, params, alg='qmix', agent=None):
    """
    Calculate system neural diversity metric
    """

    if 'qmix' in alg:
        policy = homogeneous_pass_qmix if alg == 'qmix' else homogeneous_pass_qmix_ns
        agents, agents_obs = zip(*rollouts.items())
        rollouts = jnp.stack(agents_obs[1:]) # [n_agents, timesteps, batch_dim, obs_dim]
        original_shape = rollouts.shape
        rollouts = jnp.transpose(rollouts, (1, 2, 0, 3)) # [timesteps, batch_dim, n_agents, obs_dim]

        hiddens = hiddens.reshape(original_shape[0], original_shape[1], original_shape[2], -1) # [n_agents, timesteps, batch_dim, hidden_dim]
        hiddens = jnp.transpose(hiddens, (1, 2, 0, 3)) # [n_agents, batch_dim, n_agents, hidden_dim]

    if 'mappo' in alg:
        rollouts = jnp.transpose(rollouts, (0, 2, 1, 3)) # [timesteps, batch_dim, n_agents, obs_dim]
        hiddens = jnp.transpose(hiddens, (0, 2, 1, 3)) # [timesteps, batch_dim, n_agents, obs_dim]
        policy = homogeneous_pass_mappo if alg == 'mappo' else homogeneous_pass_mappo_ns

    timesteps, batch_size, n_agents, obs_dim = rollouts.shape

    def mask_obs(agent_i):
        """
        For agents j /= i, mask the non-capability observation of j with i
        """

        obs = rollouts[..., :-dim_c]
        cap = rollouts[..., -dim_c:]
        obs_i = obs[:, :, agent_i, :]
        obs_i = obs_i[:, :, None, :] # add extra dim for broadcasting
        obs_masked = jnp.tile(obs_i, (1, 1, n_agents, 1))

        return jnp.concatenate((obs_masked, cap), axis=-1)

    def get_policy_outputs(agent_i):
        """
        Using mask obs, get policy outputs for masked observations simulating generating
        outputs for the same observation conditioned on different capabilities
        """
        obs = mask_obs(agent_i)

        # duplicate hidden state for i across each agent
        hs = hiddens[:, :, agent_i, :]
        hs = hs[:, :, None, :] # add extra dim for broadcasting
        hs = jnp.tile(hs, (1, 1, n_agents, 1))

        # trivally set dones to 0
        dones = jnp.zeros((hs.shape[0], hs.shape[1], hs.shape[2]))

        def apply_policy_per_timestep(cary, t):
            return carry, policy(params, hs[t, ...], obs[t, ...], dones[t, ...], agent=agent)

        # vectorize the policy over timesteps, necessary for use with scanned functions
        carry = None
        _, outputs = jax.lax.scan(apply_policy_per_timestep, carry, jnp.arange(timesteps))

        return outputs


    def distance_vector(agent_i):
        """
        Using get policy outputs, compute the distance between the distributions outputted by
        corresponding observations, add to flatten into a vector containing the added distances
        between i and each other agent
        """
        if 'qmix' in alg:
            qvals_i = get_policy_outputs(agent_i) # [timesteps, batch_size, n_agents, action_dim]

            # convert qvals to categorical distributions
            qval_maxs = jnp.max(qvals_i, axis=-1, keepdims=True)
            qvals_i = qvals_i - qval_maxs
            categorical_i = jnp.exp(qvals_i) / jnp.sum(jnp.exp(qvals_i), axis=-1, keepdims=True)
        elif alg == 'mappo':
            categorical_i = get_policy_outputs(agent_i)
        else:
            categorical_i = get_policy_outputs(agent_i)

        # get pairwise distance between agent i and all other agents
        def tvd_for_agent_j(j):
            return total_variational_distance(categorical_i[:, :, agent_i, :], categorical_i[:, :, j, :])

        tvd_all_agents = jax.vmap(tvd_for_agent_j)(jnp.arange(n_agents))  # [n_agents, timesteps, batch_size]

        return tvd_all_agents  # [n_agents, batch_size, timesteps]

    # get distance matrix
    dist_matrix = jax.vmap(distance_vector)(jnp.arange(n_agents))  # [n_agents, n_agents, timesteps, batch_size]

    # average the distance matrix over the batch size and timesteps
    dist_matrix_avg = jnp.mean(dist_matrix, axis=(2, 3))  # [n_agents, n_agents]

    # compute the final SND metric
    snd_value = (2 / (n_agents * (n_agents - 1))) * jnp.sum(dist_matrix_avg) / 2 # divide by 2 to account for double counting distances

    return snd_value

def dummy_homogenous_policy(params, hs, obs, dones, agent=None):
    """
    dummy policy for testing, all qvals are 1
    """

    batch_size, n_agents, _ = hs.shape

    return jnp.ones((batch_size, n_agents, 5))

def dummy_homogenous_policy_2(params, hs, obs, dones, agent=None):
    """
    dummy policy for testing, qvals for agent 0 are all 1's, agent 1 are all 2's, etc.
    """
    batch_size, n_agents, obs_dim = hs.shape

    qvals = jnp.arange(1, n_agents + 1)
    qvals = jnp.tile(qvals, (batch_size, 1))
    qvals = qvals[:, :, None]  # add dim for tiling

    return jnp.tile(qvals, (1, 1, 1, 4))

def dummy_heterogeneous_policy(params, hs, obs, dones, agent=None):
    """
    dummy policy for testing, qvals are all distinct
    """
    batch_size, n_agents, obs_dim = hs.shape

    qvals = jnp.eye(n_agents)
    qvals = qvals[None, :, :] # add extra dims for tiling

    return jnp.tile(qvals, (batch_size, 1, 1))

def main():
    """
    sanity checks
    """

    batch_size = 2
    timesteps = 3
    n_agents = 4
    obs_dim = 5
    dim_c = 2

    # ignorable observations, hidden states, and params
    same_obs = jnp.ones((timesteps, batch_size, n_agents, obs_dim))
    hiddens = jnp.ones((timesteps, batch_size, n_agents, 8))
    params = 0

    # get snd for homogenous policy 1
    snd_value = snd(same_obs, hiddens, dim_c, params, policy=dummy_homogenous_policy)
    assert jnp.isclose(snd_value, 0.0), f"Test failed: SND expected be 0 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_1")

    # get snd homoenous policy 2
    snd_value = snd(same_obs, hiddens, dim_c, params, policy=dummy_homogenous_policy_2)
    assert jnp.isclose(snd_value, 0.0), f"Test failed: SND expected be 0 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_2")

    # get snd heterogeneous policy
    snd_value = snd(same_obs, hiddens, dim_c, params, policy=dummy_heterogeneous_policy)
    assert snd_value > 0, f"Test failed: SND expected be greater than 0 but got {snd_value}"
    print(f"YAY: SND = {snd_value} for homogenous_policy_2")

if __name__ == "__main__":
    main()
