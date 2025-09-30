# universal_diff_eq
Master's thesis by @dorimolnar, supervised by @michaeldeistler


# Using Neural Networks with Jaxley Simulations  

This project extends the **Jaxley** simulator by adding a neural network on top of the basic ion channel model. The goal is to let the network generate *correction currents* that improve the match between a single-compartment simulation and experimental data.  

## How it works  

1. **Load experimental data**  
   We start by selecting an experimental setup ID and extracting voltage and current traces.  

   ```python
   import jax.numpy as jnp
   import jaxley as jx
   import jax
   from voltage_fitting.build_simulator import get_experimental_data
   import matplotlib.pyplot as plt

   

   setup = "473601979"
   t_max = 80.0
   cut = 1500
   dt = 0.025

   t, v_data, i_amp = get_experimental_data(setup, cut, t_max)
   v_data = jnp.asarray(v_data[None,])  # shape (1, T)

2.	**Build a baseline cell (here Pospischil-type cells)**

    Jaxley provides tools to set up a simple single-compartment neuron with sodium, potassium, and calcium channels.
    ```python
    from voltage_fitting.single_build_simulator import (
        setup_Na_K_simulator_step,
        build_Na_K_cell,
        set_setup,
        get_Na_K_bounds,
        get_gating_bounds,
        get_power_bounds,
    )

    all_setup = set_setup(setup)
    cell = build_Na_K_cell(
        all_setup['capacitance'], all_setup["eleak"],
        all_setup['gleak'], all_setup['length'],
        Na_gating_params=jnp.zeros(40),
        K_gating_params=jnp.zeros(20),
        Ca_gating_params=jnp.zeros(20),
        use_CaT=True, use_CaL=False, use_Km=True
    )

    current = jx.step_current(50, 1000, i_amp, dt, t_max)[cut:]
    cell.delete_stimuli(); cell.delete_recordings()
    cell.stimulate(current, verbose=False)
    cell.record(verbose=False)
    cell.set("v", all_setup['v_init'])
    cell.init_states()

3.	**Initialize neural correction networks**
    Instead of simulating a single fixed neuron, we create a population of networks (n_particles), each with randomly initialized channel, gating, and power parameters.
    ```python
    from correction_network import simulate_with_nn, init_correction_network

    master_key = jax.random.PRNGKey(0)
    models, inv_transform_params, inv_transform_gatings, inv_transform_powers = (
        init_correction_network(
            master_key,
            get_Na_K_bounds,
            get_power_bounds,
            get_gating_bounds,
            n_particles=500
        )
    )

4.	**Run parallel simulations**
    We use jax.vmap to run n_particles simulations in parallel, one for each network initialization.
    ```python
    predictions, added_currents = jax.vmap(
        simulate_with_nn,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None)
    )(
        models, setup_Na_K_simulator_step, cell, cut, t_max,
        all_setup, i_amp, v_data,
        inv_transform_params, inv_transform_gatings, inv_transform_powers
    )

5.  **Visualize results**
    ```python
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot(predictions[0])        # voltage prediction of the first model
    axs[1].plot(added_currents[0])     # added current of the first model
