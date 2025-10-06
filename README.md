# Universal Differential Equations for Biophysical Neuron Modeling  
**Master’s thesis by [@dorimolnar](https://github.com/dorimolnar)**  
Supervised by [@michaeldeistler](https://github.com/michaeldeistler)

---

## Overview  

This repository contains the code and experiments from my Master’s thesis, *[Neural ODEs and Parameter Inference in Differentiable Neuron Simulations](thesis.pdf)*.  
The work explores how **Neural Ordinary Differential Equations (neural ODEs)** can be integrated into **conductance-based neuron models** to create **hybrid models** that combine biological interpretability with data-driven flexibility.

Classical neuron models often fail to reproduce the full complexity of experimentally recorded voltage traces. To address this, the framework presented here augments mechanistic models (such as Markram-type or Pospischil-type single-compartment neurons) with neural ODE components that generate adaptive input currents. These learnable neural dynamics allow simpler single-compartment models to approximate more complex multi-compartment or experimental behaviors while maintaining computational efficiency and interpretability.

In addition, I also experiment with training separate networks for spiking and subthreshold regimes, computing the optimal additive current that would give a perfect fit, exploring the distribution of the current across channels, making ion channel kinetics trainable by learning the exponents or transition rates of gating variables, and several other explorations that together provide a broader picture of the model’s capabilities.

---

## Project Summary  

- **Goal:** Integrate neural ODEs with conductance-based models to create hybrid systems that learn from data while preserving biophysical interpretability.
- **Method:** Couple single-compartment neuron simulators written in **Jaxley** with small **neural ODEs** that generate additional currents.  
- **Approach:**  
  - Learn both **biophysical parameters** (e.g., conductances, capacitance, leak)  
  - and **neural ODE parameters** jointly from voltage recordings.  
- **Applications:**  
  - Fitting **multi-compartment simulations** with simplified models.  
  - Fitting **experimental voltage traces** from real neurons.  
  - Exploring **flexible ion channel kinetics** by learning gating exponents.  
- **Computational efficiency:** 
  - Supports **parallelized simulations** across multiple network initializations (n_particles) using JAX’s `vmap` and automatic vectorization, enabling faster experimentation and exploration of the parameter space.

---

## Repository Structure

```
voltagefit_neural_ode
│
├── voltage_fitting/
│   │
│   ├── gradient_descent/          # Early experiments: parameter fitting without neural ODEs
│   │
│   ├── paper/                     # Main experiments used in the thesis
│   │   ├── data/                  # Raw and preprocessed data
│   │   ├── fit_to_experiment/     # Fitting experimental voltage recordings
│   │   ├── grid_search_fit/       # Parameter grid searches and comparisons
│   │   ├── initial_tests/         # Preliminary or unused exploratory notebooks
│   │   ├── multi_fitting/         # Multi-compartment fitting experiments
│   │   ├── no_diff/               # Baseline models without neural ODE augmentation
│   │   ├── pospischil/            # Experiments with Pospischil-type channels
│   │   └── u_diff/                # Experiments using neural ODE augmentation
│   │
│   ├── .matplotlibrc              # Matplotlib stylefile
│   ├── __init__.py                # Initializes the voltage_fitting Python module
│   ├── build_simulator.py         # Base Jaxley-based neuron simulator setup for multi-compartment cells
│   ├── channels.py                # Markram-type channel mechanism definitions
│   ├── neural_ode.py              # Neural ODE augmentation definition
│   ├── initialization.py          # Stability-informed initialization
│   ├── loss_util.py               # Loss function and soft dynamic time warping implementation
│   ├── posthoc_summary_stats.py   # Trace analysis helper functions
│   ├── single_build_simulator.py  # Base Jaxley-based neuron simulator setup for single-compartment cells
│
├── .gitignore
├── README.md                    
├── setup.py                   
└── tasks.py  
```   

Each subfolder in `paper/` can typically include:
- `notebooks/`: Jupyter notebooks describing the experiment setup and results  
- `fig/` and `svg/`: Figures generated for the thesis in different formats
- `results/`: Saved data outputs  
- `scripts/`: Python scripts used for running experiments on a cluster  
- `slurm/`: SLURM submission files  


## Dependencies
This project relies on modern JAX-based scientific computing tools and libraries for differentiable simulation, optimization, and neural modeling.
The most important libraries with their version numbers used during this project:
- **Python** `3.12`
- **JAX** `0.6.0` and **jaxlib** `0.6.0`
- **Flax** `0.10.6`
- **Equinox** `0.12.0`
- **Optax** `0.2.4`
- **Jaxley** `0.8.2` and **jaxley-mech** `0.3.0`
- **SciPy** `1.15.2`
- **NumPy** `2.2.4`
- **Pandas** `2.2.3`
- **Matplotlib** `3.10.1`


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

3.	**Initialize neural ODE augmentation**
    Instead of simulating a single fixed neuron, we create a population of networks (n_particles), each with randomly initialized channel, gating, and power parameters.
    ```python
    from volatge_fitting.neural_ode import simulate_with_nn, init_correction_network

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
