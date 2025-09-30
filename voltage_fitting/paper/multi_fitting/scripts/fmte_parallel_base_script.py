# Script to run a multi_compartment training to data with 4 spikes, parallelized
# Original setup, no neural ODE augmentation

import sys
import os
os.environ['XLA_FLAGS'] = '--xla_cpu_use_thunk_runtime=false'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))



from jax import config
config.update("jax_enable_x64", True)



import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import optax
import jaxley as jx

import equinox as eqx
import flax.nnx as nnx


from voltage_fitting.loss_util import (
    get_scaled_loss,
)


t_max = 250.0
cut = 1500  # We cut the first 1500 datapoints because this period contains no stimulus

setup = "488683425"
# setup = "485574832"
# setup = "480353286"
# setup = "473601979"



# Define target
from voltage_fitting.build_simulator import (
    get_experimental_data
)

t, v_data, i_amp = get_experimental_data(setup, cut, t_max)
v_data = jnp.asarray(v_data[None,])


# Define the multi-compartment cell to train
from voltage_fitting.build_simulator import (
    setup_simulator,
    setup_simulator_step,
    build_cell,
    set_setup,
    get_bounds,
    transform_uniform_to_normal,
    get_prior
)

seed = 0

rng_key = jax.random.PRNGKey(seed)

all_setup = set_setup(setup)
cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])
simulate = setup_simulator(cell, cut, t_max, all_setup["v_init"], i_amp)


# Sampling parameters
names, lower, upper = get_bounds()


transform_params, inv_transform_params = transform_uniform_to_normal(lower, upper)
sample_prior = get_prior(lower, upper, transform_params)

init_rng_key, sampling_rng_key = jax.random.split(rng_key)

# Sample 1 initialization randomly.
sampled_params = sample_prior(init_rng_key)
inv_sampled_params = inv_transform_params(sampled_params)

v_multi = simulate(inv_sampled_params)


# Possible normalization functions
@jax.jit
def normalize_voltage(v):
    # Normalize voltage between -1 and 1
    return 2 * (v - jnp.min(v_data)) / (jnp.max(v_data) - jnp.min(v_data)) - 1

@jax.jit
def softsign_transform(v):
    return v / (1 + jnp.abs(v))

@jax.jit
def tempered_tanh(v, scale=0.05):
    return jnp.tanh(scale * v)

class ParameterModel(eqx.Module):
    channel_params_normal: jax.Array

    def __init__(self, *, init_params):
        self.channel_params_normal = init_params


# Simulation and loss
dt = 0.025

def simulate_full_trace(initial_state, step_fn, current, time_vec, x_o, thr = None):
    def cond_fn(carry):
        t, _, _, diff = carry
        keep_running = t < len(time_vec)
    
        if thr is None:
            return keep_running
        else:
            return jnp.logical_and(keep_running, diff < thr)

    def body_fn(carry):
        t, state, new_recs, diff = carry
        state, rec = step_fn(state, current[t])

        rec_voltage = rec[0]

        new_recs = new_recs.at[t].set(rec_voltage)

        new_diff = diff + (rec_voltage - x_o[t + 1])**2

        return (t + 1, state, new_recs, new_diff)

    # Initialize buffers
    recs = jnp.full((len(time_vec),), jnp.nan)

    carry = (0, initial_state, recs, 0.0)

    final_carry = eqx.internal.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=carry,
        max_steps=len(time_vec),
        kind="bounded"
    )

    _, _, recs, _ = final_carry

    return recs

cost_fn_power = 1.0
scale = 0.05

def loss_fn(model, all_setup, t_max, i_amp, v_data, thr=300, dtw_reg = 1.0, spike_reg = 0.0, sse_reg = 0.0):

    params = inv_transform_params(model.channel_params_normal)

    # Rebuild the simulator with new cell
    initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = params)

    # Run simulation
    preds = simulate_full_trace(
        initial_state, step_fn, current, time_vec, v_data.squeeze(), thr
    )

    preds = preds.squeeze()
    v_data = v_data.squeeze()

    sse_loss = jnp.sum(jnp.nan_to_num((preds - v_data) ** 2, nan=0.0))

    scaled_loss = get_scaled_loss(v_data, preds, cost_fn_power, scale)

    spike_loss = jnp.abs(jnp.max(preds) - jnp.max(v_data))
    
    return scaled_loss * dtw_reg + sse_loss * sse_reg + spike_loss * spike_reg

# Initialize and train
np.random.seed(42)
n_particles = 300

rngs = nnx.Rngs(1)
key = jax.random.PRNGKey(0)
keys = jax.random.split(jax.random.PRNGKey(0), num=n_particles)


all_setup = set_setup(setup)

thr = None


initial_lr = 15e-3
decay_factor = 0.99
transition_steps = 1  #  decay after every epoch
#decay_until = 400  # after x epochs, stop decaying
decay_start = 1
decay_end = 350


opt_params = jax.vmap(sample_prior)(
    jax.random.split(init_rng_key, n_particles)
)


def initialize_model(init_params):
    model = ParameterModel(init_params=init_params)
    return model


models = jax.vmap(initialize_model)(opt_params)

raw_schedule = optax.exponential_decay(
    init_value=initial_lr,
    transition_steps=transition_steps,
    decay_rate=decay_factor,
    staircase=True,
)


def custom_schedule(step):
    # Epochs before decay_start → constant
    lr = initial_lr

    # Epochs between decay_start and decay_end → decay
    decay_steps = jnp.clip(step - decay_start, 0, decay_end - decay_start)
    lr = jnp.where(
        step < decay_start,
        initial_lr,
        raw_schedule(decay_steps)
    )

    # After decay_end → freeze at last decayed value
    final_lr = raw_schedule(decay_end - decay_start)
    lr = jnp.where(step > decay_end, final_lr, lr)

    return lr


adam_optimizer = optax.adam(learning_rate=custom_schedule)
lbfgs_optimizer = optax.chain(
            optax.scale_by_lbfgs(memory_size = 6), #8
            optax.scale(-1e-3), #0.3
        )
adamw_optimizer = optax.adamw(
    learning_rate=custom_schedule, 
    weight_decay=1e-4              
)

def init_adam_optimizer_state(model):
    return adam_optimizer.init(eqx.filter(model, eqx.is_inexact_array))

def init_lbfgs_optimizer_state(model):
    return lbfgs_optimizer.init(eqx.filter(model, eqx.is_inexact_array)) 

def init_adamw_optimizer_state(model):
    return adamw_optimizer.init(eqx.filter(model, eqx.is_inexact_array))


optimizer_states = jax.vmap(init_adam_optimizer_state)(models)
# optimizer_states = jax.vmap(init_adam_optimizer_state)(models)
# optimizer_states = jax.vmap(init_lbfgs_optimizer_state)(models)

def simulation_loss_fn(all_params):
        loss = loss_fn(all_params, all_setup, t_max, i_amp, v_data, thr, dtw_reg = 1.0, spike_reg = 0.0, sse_reg = 0.0)
        return loss

def fine_simulation_loss_fn(all_params):
        loss = loss_fn(all_params, all_setup, t_max, i_amp, v_data, thr, dtw_reg = 0.0, spike_reg = 0.0, sse_reg = 1.0)
        return loss



grad_fn = nnx.jit(jax.vmap(nnx.value_and_grad(simulation_loss_fn)))
fine_grad_fn = nnx.jit(jax.vmap(nnx.value_and_grad(fine_simulation_loss_fn)))

# Set up the cell
cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

cell.delete_stimuli()
cell.delete_recordings()

current = jx.step_current(50, 1000, i_amp, dt, t_max)
current = current[cut:]

cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
cell.soma.branch(0).comp(0).record(verbose = False)
cell.set("v", all_setup['v_init'])
cell.init_states()


# Normalizing, scaling or clipping gradients
def clip_gradients(grads, max_norm=10.0, eps=1e-8):
    def clip_leaf(grad):
        flat = grad.reshape((grad.shape[0], -1)) if grad.ndim > 1 else grad[:, None]
        norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
        scaling = jnp.minimum(1.0, max_norm / (norms + eps))
        while scaling.ndim < grad.ndim:
            scaling = scaling[..., None]
        return grad * scaling
    return jax.tree_util.tree_map(clip_leaf, grads)

def clip_channel_gradients(grads, max_norm=10.0, eps=1e-8):
    def clip_leaf(grad):
        flat = grad.reshape((grad.shape[0], -1)) if grad.ndim > 1 else grad[:, None]
        norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
        scaling = jnp.minimum(1.0, max_norm / (norms + eps))
        while scaling.ndim < grad.ndim:
            scaling = scaling[..., None]
        return grad * scaling

    clipped = clip_leaf(grads.channel_params_normal)
    grads = eqx.tree_at(lambda g: g.channel_params_normal, grads, clipped)
    return grads

def clip_network_gradients(grads, max_norm=1.0, eps=1e-8):
    def clip_leaf(grad):
        if grad.ndim < 2:
            return grad  # Skip biases or scalars
        flat = grad.reshape((grad.shape[0], -1))
        norms = jnp.linalg.norm(flat, axis=1, keepdims=True)
        scaling = jnp.minimum(1.0, max_norm / (norms + eps))
        while scaling.ndim < grad.ndim:
            scaling = scaling[..., None]
        return grad * scaling

    def clip_net(net_grads):
        return jax.tree_util.tree_map(clip_leaf, net_grads)

    grads = eqx.tree_at(lambda g: g.spike_neural_net, grads, clip_net(grads.spike_neural_net))
    grads = eqx.tree_at(lambda g: g.spike_readout_net, grads, clip_net(grads.spike_readout_net))
    grads = eqx.tree_at(lambda g: g.nospike_neural_net, grads, clip_net(grads.nospike_neural_net))
    grads = eqx.tree_at(lambda g: g.nospike_readout_net, grads, clip_net(grads.nospike_readout_net))

    return grads

def normalize_gradients(grads, beta=1.0, eps=1e-8):
    def normalize_leaf(grad):
        norm = jnp.linalg.norm(grad, axis=1, keepdims=True)
        return grad / (norm ** beta + eps)

    grads = jax.tree_util.tree_map(normalize_leaf, grads)
    return grads

def normalize_channel_gradients(grads, beta=1.0, eps=1e-8):
    norms = jnp.linalg.norm(grads.channel_params_normal, axis=1, keepdims=True)
    normalized = grads.channel_params_normal / (norms ** beta + eps)

    # Replace only the channel_params_normal field
    grads = eqx.tree_at(lambda g: g.channel_params_normal, grads, normalized)
    return grads

def scale_grads(grads, lr_spike, lr_nospike, lr_channel):
    def scale_tree(tree, factor):
        return jax.tree_util.tree_map(lambda x: x * factor, tree)

    grads_scaled = grads

    grads_scaled = eqx.tree_at(lambda g: g.spike_neural_net, grads_scaled,
                            scale_tree(grads.spike_neural_net, lr_spike))
    grads_scaled = eqx.tree_at(lambda g: g.spike_readout_net, grads_scaled,
                            scale_tree(grads.spike_readout_net, lr_spike))
    grads_scaled = eqx.tree_at(lambda g: g.nospike_neural_net, grads_scaled,
                            scale_tree(grads.nospike_neural_net, lr_nospike))
    grads_scaled = eqx.tree_at(lambda g: g.nospike_readout_net, grads_scaled,
                            scale_tree(grads.nospike_readout_net, lr_nospike))
    grads_scaled = eqx.tree_at(lambda g: g.channel_params_normal, grads_scaled,
                            grads.channel_params_normal * lr_channel)

    return grads_scaled

num_epochs = 400
mean_losses = []

# Storing the best models and their losses during training
best_losses = jnp.full((1,), jnp.inf)
best_models = models  # Start with the initial models as best
grads_list = []
#model_snapshots = []

for epoch in range(num_epochs):

    loss, grads = grad_fn(models)
    grads_list.append(grads)
        
   
    
    # grads = clip_channel_gradients(grads, max_norm=10.0, eps=1e-8)
    # #grads = scale_grads(grads, lr_spike=10, lr_nospike = 1e-2, lr_channel=1.0)
    grads = normalize_channel_gradients(grads, beta=0.99)



    improved = loss < best_losses
    best_losses = jnp.where(improved, loss, best_losses)

    def replace_if_improved(new, old):
        mask = improved.reshape((-1,) + (1,) * (new.ndim - 1))
        return jnp.where(mask, new, old)

    best_models = jax.tree_util.tree_map(replace_if_improved, models, best_models)

    mask = ~jnp.isnan(loss)
    mean_loss = jnp.sum(loss[mask]) / jnp.sum(mask)
    mean_losses.append(mean_loss)

    updates, optimizer_states = jax.vmap(lambda g, s, m: adam_optimizer.update(g, s, m, step = epoch))(grads, optimizer_states, models)
    models = jax.vmap(eqx.apply_updates)(models, updates)
    #model_snapshots.append(jax.tree_util.tree_map(lambda x: x.copy(), models))

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Mean Loss: {mean_loss:.6f}, Median Loss {jnp.median(loss[mask]):.6f}, Min Loss: {jnp.min(loss[mask]):.6f}")
        print('Max gradient norm: ', jnp.max(jnp.linalg.norm(grads.channel_params_normal, axis=1)), ', Min gradient norm: ', jnp.min(jnp.linalg.norm(grads.channel_params_normal, axis=1)))

nan_mask = jnp.isnan(grads_list[num_epochs-1].channel_params_normal)

# Get indices where the mask is True
nan_indices = jnp.argwhere(nan_mask)
print("Not successfully trained models:", int(len(nan_indices)/15))


# Save to file
with open("base_best_models.eqx", "wb") as f:
    eqx.tree_serialise_leaves(f, best_models)

print(f"Best models saved to best_models.eqx")


# # Load from file
# with open("best_models.eqx", "rb") as f:
#     loaded_models = eqx.tree_deserialise_leaves(f, models)

inds = np.argsort(best_losses)

fig, axs = plt.subplots(2,5, figsize = (15,5))
fig.subplots_adjust(hspace=0.5, wspace=0.3)
for i in range(2):
    for j in range(5):
        model = jax.tree_util.tree_map(lambda x: x[inds[0 + i*5+j]], best_models)
        opt_inv_params = inv_transform_params(model.channel_params_normal)

        # Rebuild the simulator with new cell
        cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

        cell.delete_stimuli()
        cell.delete_recordings()

        current = jx.step_current(50, 1000, i_amp, dt, t_max)
        current = current[cut:]

        cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
        cell.soma.branch(0).comp(0).record(verbose = False)
        cell.set("v", all_setup['v_init'])
        cell.init_states()
        
        initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

        # Run simulation
        final_preds = simulate_full_trace(
            initial_state, step_fn, current, time_vec, v_data.squeeze(), thr
        )
        axs[i,j].plot(time_vec, v_data[0], label="Experimental target", linewidth=2)
        axs[i,j].plot(time_vec, final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
        axs[i,j].set_title(f"Model {inds[i*5+j]}, Loss: {best_losses[inds[i*5+j]]:.4f}", fontsize = 10)
        axs[i,j].set_ylim(-100, 60)

fig.savefig("basic_top_10_results.png", dpi=300, bbox_inches='tight')

# Checking the best model

idx = 0
cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

cell.delete_stimuli()
cell.delete_recordings()

current = jx.step_current(50, 1000, i_amp, dt, t_max)
current = current[cut:]

cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
cell.soma.branch(0).comp(0).record(verbose = False)
cell.set("v", all_setup['v_init'])
cell.init_states()

model = jax.tree_util.tree_map(lambda x: x[inds[idx]], best_models)
opt_inv_params = inv_transform_params(model.channel_params_normal)

# Rebuild the simulator with new cell
initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

# Run simulation
final_preds = simulate_full_trace(
            initial_state, step_fn, current, time_vec, v_data.squeeze(), thr
        )

loss = simulation_loss_fn(model)


fig, ax = plt.subplots(1,1, figsize = (10,6))
ax.plot(v_data[0], label="Data target", linewidth=2)
ax.plot(final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
ax.set_title(f"Loss: {loss:.4f}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend(["Target", "Prediction"])
fig.savefig("basic_best_model_prediction.png", dpi=300, bbox_inches='tight')



# Final parameters
opt_inv_params_list = []
for i in range(10):
    model = jax.tree_util.tree_map(lambda x: x[inds[i]], best_models)
    opt_inv_params = inv_transform_params(model.channel_params_normal)
    opt_inv_params_list.append(opt_inv_params)

param_matrix = np.stack(opt_inv_params_list)

normalized_params = (param_matrix - lower) / (upper - lower)
normalized_params = np.clip(normalized_params, 0, 1)

fig, ax = plt.subplots(figsize=(9, 6))
indices = np.arange(len(opt_inv_params))

# Draw the [0, 1] range
ax.hlines(indices, 0, 1, color='lightgray', linewidth=5, zorder=1)
colors = plt.cm.tab10(np.linspace(0, 1, 10))
colors = plt.cm.Pastel1(np.linspace(0, 1, 10))
colors = plt.cm.viridis(np.linspace(0.0, 1.0, 10))
sizes = np.linspace(100, 30, 10)
alphas = np.linspace(1.0, 0.5, 10)


# Plot all models
for j in range(10):
    ax.scatter(normalized_params[j], indices, color=colors[j], s=sizes[j], alpha = alphas[j], label=f'Model {j+1}', zorder=2)

# Annotate only the best model (j == 0)
for k in range(len(names)):
    x = normalized_params[0, k]
    y = k
    val = param_matrix[0, k]
    ax.text(x - 0.03, y + 0.4, f"{val:.3f}", fontsize=8, color='k', va='center')


ax.set_yticks(indices)
ax.set_yticklabels(names)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel("Normalized parameter value")
ax.set_title("Normalized Optimized Parameters from 10 Best Models")
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

fig.tight_layout()
fig.savefig("basic_normalized_optimized_parameters.png", dpi=300, bbox_inches='tight')

#####################################################
# Fine-tuning the best 40 models
#####################################################

top_k = 40
top_k_inds = inds[:top_k]
top_k_models = jax.tree_util.tree_map(lambda x: x[top_k_inds], best_models)
top_k_losses = jnp.full((top_k,), jnp.inf)

fine_tune_lr = 5e-4 
decay_start = 100
decay_end = 200


fine_schedule = optax.exponential_decay(
    init_value=fine_tune_lr,
    transition_steps=1,
    decay_rate=0.99,
    staircase=True,
)
def fine_custom_schedule(step):
    # Epochs before decay_start → constant
    lr = fine_tune_lr

    # Epochs between decay_start and decay_end → decay
    decay_steps = jnp.clip(step - decay_start, 0, decay_end - decay_start)
    lr = jnp.where(
        step < decay_start,
        fine_tune_lr,
        fine_schedule(decay_steps)
    )

    # After decay_end → freeze at last decayed value
    final_lr = fine_schedule(decay_end - decay_start)
    lr = jnp.where(step > decay_end, final_lr, lr)

    return lr

fine_optimizer = optax.adam(fine_custom_schedule)
# fine_optimizer = optax.sgd(learning_rate=1e-8, momentum=0.9)
# fine_optimizer = optax.chain(
#             optax.scale_by_lbfgs(memory_size = 8), #8
#             optax.scale(-1e-3), #0.3
#         )

fine_optimizer_states = jax.vmap(lambda m: fine_optimizer.init(eqx.filter(m, eqx.is_inexact_array)))(top_k_models)

cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

cell.delete_stimuli()
cell.delete_recordings()

current = jx.step_current(50, 1000, i_amp, dt, t_max)
current = current[cut:]

cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
cell.soma.branch(0).comp(0).record(verbose = False)
cell.set("v", all_setup['v_init'])
cell.init_states()

num_fine_epochs = 400
fine_mean_losses = []
fine_best_models = top_k_models
fine_best_losses = top_k_losses

for epoch in range(num_fine_epochs):

    loss, grads = fine_grad_fn(top_k_models)

    #grads = clip_channel_gradients(grads, max_norm=10.0, eps=1e-8)
    #grads = eqx.tree_at(lambda g: g.channel_params_normal, grads, jnp.zeros_like(grads.channel_params_normal))
    #grads = scale_grads(grads, lr_spike=10.0, lr_nospike = 1e-2, lr_channel=1.0)

    grads = clip_channel_gradients(grads, max_norm=10.0, eps=1e-8)

    improved = loss < fine_best_losses
    fine_best_losses = jnp.where(improved, loss, fine_best_losses)
    fine_best_models = jax.tree_util.tree_map(replace_if_improved, top_k_models, fine_best_models)

    updates, fine_optimizer_states = jax.vmap(lambda g, s, m: fine_optimizer.update(g, s, m, step=epoch))(grads, fine_optimizer_states, top_k_models)
    top_k_models = jax.vmap(eqx.apply_updates)(top_k_models, updates)

    mask = ~jnp.isnan(loss)
    fine_mean_loss = jnp.sum(loss[mask]) / jnp.sum(mask)

    fine_mean_losses.append(fine_mean_loss)

    if epoch % 10 == 0:
        print(f"Fine-tune epoch {epoch}, Mean Loss: {fine_mean_loss:.6f}, Median Loss {jnp.median(loss[mask]):.6f}, Min Loss: {jnp.min(loss[mask]):.6f}")


fine_inds = np.argsort(fine_best_losses)
fig, axs = plt.subplots(2,5, figsize = (20,6))
fig.subplots_adjust(hspace=0.5, wspace=0.3)
fig.suptitle("Best fine-tuned models", fontsize=16)
for i in range(2):
    for j in range(5):
        model = jax.tree_util.tree_map(lambda x: x[fine_inds[i*5+j]], fine_best_models)
        opt_inv_params = inv_transform_params(model.channel_params_normal)

        # Rebuild the simulator with new cell
        cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

        cell.delete_stimuli()
        cell.delete_recordings()

        current = jx.step_current(50, 1000, i_amp, dt, t_max)
        current = current[cut:]


        cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
        cell.soma.branch(0).comp(0).record(verbose = False)
        cell.set("v", all_setup['v_init'])
        cell.init_states()
        
        initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

        # Run simulation
        final_preds = simulate_full_trace(
            initial_state, step_fn, current, time_vec, v_data.squeeze(), thr
        )
        axs[i,j].plot(time_vec, v_data[0], label="Experimental target", linewidth=2)
        axs[i,j].plot(time_vec, final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
        axs[i,j].set_title(f"Model {inds[fine_inds[i*5+j]]}, Loss: {fine_best_losses[fine_inds[i*5+j]]:.4f}", fontsize = 10)
        axs[i,j].set_ylim(-100, 60)

fig.savefig("basic_fine_tuned_top_10_results.png", dpi=300, bbox_inches='tight')

# Checking the best fine-tuned model
idx = 0
cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

cell.delete_stimuli()
cell.delete_recordings()

current = jx.step_current(50, 1000, i_amp, dt, t_max)
current = current[cut:]

cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
cell.soma.branch(0).comp(0).record(verbose = False)
cell.set("v", all_setup['v_init'])
cell.init_states()


model = jax.tree_util.tree_map(lambda x: x[fine_inds[idx]], fine_best_models)
opt_inv_params = inv_transform_params(model.channel_params_normal)

# Rebuild the simulator with new cell
initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

# Run simulation
final_preds = simulate_full_trace(
            initial_state, step_fn, current, time_vec, v_data.squeeze(), thr
        )

loss = fine_simulation_loss_fn(model)

fig, ax = plt.subplots(1,1, figsize = (10,6))
ax.plot(v_data[0], label="Data target", linewidth=2)
ax.plot(final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
ax.set_title(f"Loss: {loss:.4f}")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Voltage (mV)")
ax.legend(["Target", "Prediction"])
fig.savefig("basic_fine_tuned_best_model_prediction.png", dpi=300, bbox_inches='tight')


###############################
# Fine-tuning with dtw-loss
###############################


# def fine_simulation_loss_fn(all_params):
#         loss = loss_fn(all_params, all_setup, t_max, i_amp, u_init, v_data, added_curr_std, nospike_u_time_constant, spike_u_time_constant, thr, dtw_reg = 1.0, spike_reg = 0.0, sse_reg = 0.0, add_reg = 0.0)
#         return loss

# fine_grad_fn = nnx.jit(jax.vmap(nnx.value_and_grad(fine_simulation_loss_fn)))

# top_k = 40
# top_k_inds = inds[:top_k]
# top_k_models = jax.tree_util.tree_map(lambda x: x[top_k_inds], best_models)
# top_k_losses = jnp.full((top_k,), jnp.inf)

# fine_tune_lr = 5e-4 
# decay_start = 100
# decay_end = 200


# fine_schedule = optax.exponential_decay(
#     init_value=fine_tune_lr,
#     transition_steps=1,
#     decay_rate=0.99,
#     staircase=True,
# )
# def fine_custom_schedule(step):
#     # Epochs before decay_start → constant
#     lr = fine_tune_lr

#     # Epochs between decay_start and decay_end → decay
#     decay_steps = jnp.clip(step - decay_start, 0, decay_end - decay_start)
#     lr = jnp.where(
#         step < decay_start,
#         fine_tune_lr,
#         fine_schedule(decay_steps)
#     )

#     # After decay_end → freeze at last decayed value
#     final_lr = fine_schedule(decay_end - decay_start)
#     lr = jnp.where(step > decay_end, final_lr, lr)

#     return lr

# fine_optimizer = optax.adam(fine_custom_schedule)
# # fine_optimizer = optax.sgd(learning_rate=1e-8, momentum=0.9)
# # fine_optimizer = optax.chain(
# #             optax.scale_by_lbfgs(memory_size = 8), #8
# #             optax.scale(-1e-3), #0.3
# #         )

# fine_optimizer_states = jax.vmap(lambda m: fine_optimizer.init(eqx.filter(m, eqx.is_inexact_array)))(top_k_models)

# cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

# cell.delete_stimuli()
# cell.delete_recordings()

# current = jx.step_current(50, 1000, i_amp, dt, t_max)
# current = current[cut:]

# cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
# cell.soma.branch(0).comp(0).record(verbose = False)
# cell.set("v", all_setup['v_init'])
# cell.init_states()

# num_fine_epochs = 400
# fine_mean_losses = []
# fine_best_models = top_k_models
# fine_best_losses = top_k_losses

# for epoch in range(num_fine_epochs):

#     loss, grads = fine_grad_fn(top_k_models)

#     #grads = clip_channel_gradients(grads, max_norm=10.0, eps=1e-8)
#     #grads = eqx.tree_at(lambda g: g.channel_params_normal, grads, jnp.zeros_like(grads.channel_params_normal))
#     #grads = scale_grads(grads, lr_spike=10.0, lr_nospike = 1e-2, lr_channel=1.0)
    
#     grads = clip_network_gradients(grads, max_norm=100.0, eps=1e-8)
#     grads = clip_channel_gradients(grads, max_norm=10.0, eps=1e-8)
#     grads = scale_grads(grads, lr_spike = 10.0, lr_nospike = 10.0, lr_channel = 1.0)

#     improved = loss < fine_best_losses
#     fine_best_losses = jnp.where(improved, loss, fine_best_losses)
#     fine_best_models = jax.tree_util.tree_map(replace_if_improved, top_k_models, fine_best_models)

#     updates, fine_optimizer_states = jax.vmap(lambda g, s, m: fine_optimizer.update(g, s, m, step=epoch))(grads, fine_optimizer_states, top_k_models)
#     top_k_models = jax.vmap(eqx.apply_updates)(top_k_models, updates)

#     mask = ~jnp.isnan(loss)
#     fine_mean_loss = jnp.sum(loss[mask]) / jnp.sum(mask)

#     fine_mean_losses.append(fine_mean_loss)

#     if epoch % 10 == 0:
#         print(f"Fine-tune epoch {epoch}, Mean Loss: {fine_mean_loss:.6f}, Median Loss {jnp.median(loss[mask]):.6f}, Min Loss: {jnp.min(loss[mask]):.6f}")


# fine_inds = np.argsort(fine_best_losses)
# fig, axs = plt.subplots(2,5, figsize = (20,6))
# fig.subplots_adjust(hspace=0.5, wspace=0.3)
# fig.suptitle("Best fine-tuned models", fontsize=16)
# for i in range(2):
#     for j in range(5):
#         model = jax.tree_util.tree_map(lambda x: x[fine_inds[i*5+j]], fine_best_models)
#         opt_inv_params = inv_transform_params(model.channel_params_normal)

#         # Rebuild the simulator with new cell
#         cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])


#         cell.delete_stimuli()
#         cell.delete_recordings()

#         current = jx.step_current(50, 1000, i_amp, dt, t_max)
#         current = current[cut:]

#         cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
#         cell.soma.branch(0).comp(0).record(verbose = False)
#         cell.set("v", all_setup['v_init'])
#         cell.init_states()
        
#         initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

#         # Run simulation
#         final_preds, final_additional_currents, final_us, final_nn_reses = simulate_full_trace(
#             model.spike_neural_net, model.spike_readout_net, model.nospike_neural_net, model.nospike_readout_net, initial_state, step_fn, current, time_vec, u_init, v_data.squeeze(), added_curr_std, nospike_u_time_constant, spike_u_time_constant, thr
#         )
#         axs[i,j].plot(time_vec, v_data[0], label="Experimental target", linewidth=2)
#         axs[i,j].plot(time_vec, final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
#         axs[i,j].set_title(f"Model {inds[fine_inds[i*5+j]]}, Loss: {fine_best_losses[fine_inds[i*5+j]]:.4f}", fontsize = 10)
#         axs[i,j].set_ylim(-100, 60)

# fig.savefig("dtw_fine_tuned_top_10_results.png", dpi=300, bbox_inches='tight')

# # Checking the best fine-tuned model
# idx = 0
# cell = build_cell(all_setup["swc_fname"], all_setup["rotation"], all_setup['capacitance'], all_setup["eleak"], all_setup['gleak'])

# cell.delete_stimuli()
# cell.delete_recordings()

# current = jx.step_current(50, 1000, i_amp, dt, t_max)
# current = current[cut:]

# cell.soma.branch(0).comp(0).stimulate(current, verbose = False)
# cell.soma.branch(0).comp(0).record(verbose = False)
# cell.set("v", all_setup['v_init'])
# cell.init_states()


# model = jax.tree_util.tree_map(lambda x: x[fine_inds[idx]], fine_best_models)
# opt_inv_params = inv_transform_params(model.channel_params_normal)

# # Rebuild the simulator with new cell
# initial_state, step_fn, current, time_vec = setup_simulator_step(cell, cut, t_max, all_setup["v_init"], i_amp, params = opt_inv_params)

# # Run simulation
# final_preds, final_additional_currents, final_us, final_nn_reses = simulate_full_trace(
#     model.spike_neural_net, model.spike_readout_net, model.nospike_neural_net, model.nospike_readout_net, initial_state, step_fn, current, time_vec, u_init, v_data.squeeze(), added_curr_std, nospike_u_time_constant, spike_u_time_constant, thr
# )

# loss = fine_simulation_loss_fn(model)

# fig, ax = plt.subplots(1,1, figsize = (10,6))
# ax.plot(v_data[0], label="Data target", linewidth=2)
# ax.plot(final_preds, label="Multi-compartment prediction", linestyle="--", linewidth=2)
# ax.set_title(f"Loss: {loss:.4f}")
# ax.set_xlabel("Time (ms)")
# ax.set_ylabel("Voltage (mV)")
# ax.legend(["Target", "Prediction"])
# fig.savefig("dtw_fine_tuned_best_model_prediction.png", dpi=300, bbox_inches='tight')

# fig, axs = plt.subplots(1,3, figsize = (15,3))
# axs[0].plot(time_vec, final_us)
# axs[0].set_title("Hidden state (u)")
# axs[1].plot(time_vec, final_additional_currents)
# axs[1].set_title("Additional current")
# axs[2].plot(time_vec, final_nn_reses)
# axs[2].set_title("Dynamics network output (neural_net)")
# fig.savefig("dtw_fine_tuned_best_model_details.png", dpi=300, bbox_inches='tight')


