import equinox as eqx
from typing import List
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from voltage_fitting.single_build_simulator import transform_uniform_to_normal, get_prior
from voltage_fitting.initialization import (
    rejection_sample_eigenvalues,
    sample_eigenvalues_on_circle,
    generate_all_lambda_matrices,
    generate_all_pi_matrices,
    generate_W_matrices,
)



class SimpleNeuralNet(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __init__(self, *, key, u_dimension=8, W_matrices: List[jax.Array], epsilon: float = 1e-4):
        k1, k2, k3, k_b1, k_b2, k_b3 = jr.split(key, 6)

        self.linear1 = eqx.nn.Linear(u_dimension + 1, 32, key=k1)
        self.linear2 = eqx.nn.Linear(32, 16, key=k2)
        self.linear3 = eqx.nn.Linear(16, u_dimension, key=k3)


        self.linear1 = eqx.tree_at(lambda l: l.weight, self.linear1, W_matrices[0])
        self.linear2 = eqx.tree_at(lambda l: l.weight, self.linear2, W_matrices[1])
        self.linear3 = eqx.tree_at(lambda l: l.weight, self.linear3, W_matrices[2])

        self.linear1 = eqx.tree_at(
            lambda l: l.bias, self.linear1,
            jr.uniform(k_b1, (32,), minval=-epsilon, maxval=epsilon)
        )
        self.linear2 = eqx.tree_at(
            lambda l: l.bias, self.linear2,
            jr.uniform(k_b2, (16,), minval=-epsilon, maxval=epsilon)
        )
        self.linear3 = eqx.tree_at(
            lambda l: l.bias, self.linear3,
            jr.uniform(k_b3, (u_dimension,), minval=-epsilon, maxval=epsilon)
        )

    def __call__(self, x):
        x = jax.nn.relu(self.linear1(x))
        x = jax.nn.relu(self.linear2(x))
        return self.linear3(x)


class ReadoutNet(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, *, key, u_dimension=8):
        k1, k2 = jr.split(key, 2)
        self.linear1 = eqx.nn.Linear(u_dimension, 32, key=k1)
        self.linear2 = eqx.nn.Linear(32, 1, key=k2)

    def __call__(self, u):
        x = jax.nn.relu(self.linear1(u))
        return self.linear2(x)


class CorrectionModel(eqx.Module):
    neural_net: SimpleNeuralNet
    readout_net: ReadoutNet
    channel_params_normal: jax.Array

    def __init__(self, *, key, init_params, u_dimension=8, W_matrices: List[jax.Array], epsilon: float = 1e-4):
        k1, k2 = jr.split(key, 2)
        self.neural_net = SimpleNeuralNet(key=k1, u_dimension=u_dimension, W_matrices=W_matrices, epsilon=epsilon)
        self.readout_net = ReadoutNet(key=k2, u_dimension=u_dimension)
        self.channel_params_normal = init_params

    def __call__(self, u, v):
        u_new = self.neural_net(u, v)
        return self.readout_net(u_new)
    
class DoubleCorrectionModel(eqx.Module):
    spike_neural_net: eqx.Module
    spike_readout_net: eqx.Module
    nospike_neural_net: eqx.Module
    nospike_readout_net: eqx.Module
    channel_params_normal: jnp.ndarray
    Na_gating_params_normal: jnp.ndarray
    K_gating_params_normal: jnp.ndarray
    Ca_gating_params_normal: jnp.ndarray
    powers_normal: jnp.ndarray


    def __init__(self, *, key, init_params, init_Na_gating_params, init_K_gating_params, init_Ca_gating_params, init_powers_normal, u_dimension=8, W_matrices: List[jax.Array], epsilon: float = 1e-4):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.spike_neural_net = SimpleNeuralNet(key=k1, u_dimension=u_dimension, W_matrices=W_matrices, epsilon=epsilon)
        self.spike_readout_net = ReadoutNet(key=k2, u_dimension=u_dimension)
        self.nospike_neural_net = SimpleNeuralNet(key=k3, u_dimension=u_dimension, W_matrices=W_matrices, epsilon=epsilon)
        self.nospike_readout_net = ReadoutNet(key=k4, u_dimension=u_dimension)
        self.channel_params_normal = init_params
        self.Na_gating_params_normal = init_Na_gating_params
        self.K_gating_params_normal = init_K_gating_params
        self.Ca_gating_params_normal = init_Ca_gating_params
        self.powers_normal = init_powers_normal

    # def __call__(self, x):
    #     u_new = self.neural_net(x)
    #     return self.readout_net(u_new)


def init_correction_network(key, get_bounds, get_power_bounds, get_gating_bounds, n_particles = 300, u_dimension = 8, W_matrices = None, epsilon = 1e-4):
    key1, key2, key3, key4 = jr.split(key, 4)
    keys = jr.split(key, num=n_particles)

    _, lower, upper = get_bounds()
    transform_params, inv_transform_params = transform_uniform_to_normal(lower, upper)
    sample_prior = get_prior(lower, upper, transform_params)
    init_params_normal = jax.vmap(sample_prior)(jr.split(key1, n_particles))

    _, lower_powers, upper_powers = get_power_bounds()
    transform_powers, inv_transform_powers = transform_uniform_to_normal(lower_powers, upper_powers)
    sample_powers = get_prior(lower_powers, upper_powers, transform_powers)
    init_powers_normal = jax.vmap(sample_powers)(jr.split(key2, n_particles))

    _, lower_gatings, upper_gatings = get_gating_bounds()
    transform_gatings, inv_transform_gatings = transform_uniform_to_normal(lower_gatings, upper_gatings)
    sample_gatings = get_prior(lower_gatings, upper_gatings, transform_gatings)
    init_gating_params = jax.vmap(sample_gatings)(jr.split(key3, n_particles))
    init_Na_gating_params = init_gating_params[:,:40]
    init_K_gating_params = init_gating_params[:,40:60]
    init_Ca_gating_params = init_gating_params[:,60:80]

    if W_matrices is None:
        eigs = rejection_sample_eigenvalues(
            key = key4,
            solver_order=1,   
            h=0.025,
            u_dim=u_dimension,
            use_complex=True
        )
        num_layers = 3
        layer_shapes = [(u_dimension + 1, 32), (32, 16), (16, u_dimension)]
        lambda_matrices = generate_all_lambda_matrices(layer_shapes, eigs, num_layers)
        pi_matrices = generate_all_pi_matrices(key, layer_shapes)
        W_matrices = generate_W_matrices(lambda_matrices, pi_matrices)


    def initialize_one_model(key, init_params, Na_gating_params, K_gating_params, Ca_gating_params, powers, u_dimension, W_matrices, epsilon):
        model = DoubleCorrectionModel(
            key=key,
            init_params=init_params,
            init_Na_gating_params=Na_gating_params,
            init_K_gating_params=K_gating_params,
            init_Ca_gating_params=Ca_gating_params,
            init_powers_normal=powers,
            u_dimension=u_dimension,
            W_matrices=W_matrices,
            epsilon=epsilon
        )
        return model
    
    models = jax.vmap(initialize_one_model, in_axes=(0, 0, 0, 0, 0, 0, None, None, None))(
        keys, init_params_normal, init_Na_gating_params, init_K_gating_params, init_Ca_gating_params, init_powers_normal, u_dimension, W_matrices, epsilon
    )

    return models, inv_transform_params, inv_transform_gatings, inv_transform_powers


# SIMULATION


def simulate_full_trace(spike_neural_net, spike_readout_net, nospike_neural_net, nospike_readout_net, initial_state, step_fn, current, time_vec, u_init, x_o, added_curr_std, nospike_u_time_constant, spike_u_time_constant, thr = None, dt = 0.025, dv_treshold = 0.5):
    def normalize_voltage(v):
        # Normalize voltage between -1 and 1
        return 2 * (v - jnp.min(x_o)) / (jnp.max(x_o) - jnp.min(x_o)) - 1
    
    def cond_fn(carry):
        t, _, _, _, diff, _ = carry
        keep_running = t < len(time_vec)
    
        if thr is None:
            return keep_running
        else:
            return jnp.logical_and(keep_running, diff < thr)

    def body_fn(carry):
        t, state, u, recs_adds, diff, prev_voltage = carry

        voltage = state["v"].squeeze()
        voltage_normalized = normalize_voltage(voltage)
        #voltage_normalized = softsign_transform(voltage)
        #voltage_normalized = tempered_tanh(voltage)


        inputs = jnp.concatenate([u, jnp.expand_dims(voltage_normalized, 0)], axis=0)

        # Spike/Nospike?
        dv = jnp.abs(voltage - prev_voltage)
        use_spike_net = dv > dv_treshold #jnp.logical_or(dv > dv_treshold), voltage > -40.0)
        
        # Compute add_current using spike/nospike NNs
        neural_net_output = jax.lax.cond(use_spike_net, lambda: spike_neural_net(inputs), lambda: nospike_neural_net(inputs))
        u_time_constant = jax.lax.cond(use_spike_net, lambda: spike_u_time_constant, lambda: nospike_u_time_constant)
        u_new = u + dt*neural_net_output / u_time_constant

        readout_net_output = jax.lax.cond(use_spike_net, lambda: spike_readout_net(u_new), lambda: nospike_readout_net(u_new))
        add_current = readout_net_output.squeeze()
        



        total_current = current[t] + add_current * added_curr_std
        state, rec = step_fn(state, total_current)
        rec_voltage = rec[0]



      
        new_us, new_recs, new_adds, nn_reses = recs_adds
        new_recs = new_recs.at[t].set(rec_voltage)
        new_adds = new_adds.at[t].set(add_current)
        new_us = new_us.at[t].set(u_new)
        nn_reses = nn_reses.at[t].set(neural_net_output)

        new_diff = diff + (rec_voltage - x_o[t + 1])**2

        return (t + 1, state, u_new.squeeze(), (new_us, new_recs, new_adds, nn_reses), new_diff, voltage)


    # Initialize buffers
    recs = jnp.full((len(time_vec),), jnp.nan)
    adds = jnp.full((len(time_vec),), jnp.nan)
    us = jnp.full((len(time_vec), u_init.shape[0]), jnp.nan)
    nn_reses = jnp.full((len(time_vec), u_init.shape[0]), jnp.nan)

    carry = (0, initial_state, u_init.squeeze(), (us, recs, adds, nn_reses), 0.0, initial_state["v"][0].squeeze())

    final_carry = eqx.internal.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=carry,
        max_steps=len(time_vec),
        kind="bounded"
    )

    _, _, _, (us, recs, adds, nn_reses), _, _ = final_carry

    return recs, adds, us, nn_reses


def simulate_with_nn(model, setup_simulator_step, cell, cut, t_max, all_setup, i_amp, v_data, inv_transform_params, inv_transform_gatings, inv_transform_powers, added_curr_std = 1.0/10.0, u_init = None, u_dimension = 8, thr = None, use_powers = False):
    if u_init is None:
        key = jr.PRNGKey(0)
        u_init = jax.random.normal(key, (u_dimension,))

    key = jr.PRNGKey(0)
    key1, key2 = jr.split(key, num = 2)
    log_uniform_samples = jr.uniform(key1, shape=(u_dimension,), minval=jnp.log(1), maxval=jnp.log(100.0))
    nospike_u_time_constant = jnp.exp(log_uniform_samples)
    log_uniform_samples = jr.uniform(key2, shape=(u_dimension,), minval=jnp.log(0.1), maxval=jnp.log(10.0))
    spike_u_time_constant = jnp.exp(log_uniform_samples)

    params = inv_transform_params(model.channel_params_normal)
    inv_powers = inv_transform_powers(model.powers_normal)
    inv_gating_params = inv_transform_gatings(jnp.concatenate([model.Na_gating_params_normal, model.K_gating_params_normal, model.Ca_gating_params_normal]))
    inv_Na_gating_params = inv_gating_params[:40]
    inv_K_gating_params = inv_gating_params[40:60]
    inv_Ca_gating_params = inv_gating_params[60:80]
    if use_powers:
        initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, gating = True, params = params, Na_gating_params = inv_Na_gating_params, K_gating_params = inv_K_gating_params, Ca_gating_params = inv_Ca_gating_params, powers = inv_powers)
    else:
        initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, gating = True, params = params, Na_gating_params = inv_Na_gating_params, K_gating_params = inv_K_gating_params, Ca_gating_params = inv_Ca_gating_params, powers = [3,1,4])
    # Run simulation
    preds, additional_currents, _, _ = simulate_full_trace(
        model.spike_neural_net, model.spike_readout_net, model.nospike_neural_net, model.nospike_readout_net, initial_state, step_fn, current, time_vec, u_init, v_data.squeeze(), added_curr_std, nospike_u_time_constant, spike_u_time_constant, thr
    )

    return preds, additional_currents