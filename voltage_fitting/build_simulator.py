import os
from typing import Callable, Tuple
from jax import Array

from math import log10
#import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
import pickle
import jax.numpy as jnp
import jax

import pickle

import jaxley as jx

from jaxley.channels import Leak, Na, K
from voltage_fitting.channels import (
    NaTs2T,
    SKv3_1,
    M,
    H,
    SKE2,
    CaNernstReversal,
    CaPump,
    CaHVA,
    CaLVA,
    NaTaT,
    KPst,
    KTst,
)

from jaxley.integrate import build_init_and_step_fn

base_path = "/Users/moldor/Documents/Thesis/universal_diff_eq/voltage_fitting"

i_delay = 50.0
dt = 0.025


def set_setup(new_setup):
    """
    Define swc file and hyperparameters (i_amp, capacitance,...) depending on morph.

    """

    all_setup = {}
    
    if new_setup == "488683425":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_488683425/reconstruction.swc"
        all_setup["rotation"] = 195
        all_setup["v_init"] = -83.15625
        all_setup["gleak"] = 1e-4
        all_setup["capacitance"] = 2.0
        all_setup["eleak"] = -88.0
    elif new_setup == "485574832":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_485574832/reconstruction.swc"
        all_setup["rotation"] = 155
        all_setup["v_init"] = -85.6
        all_setup["gleak"] = 1e-4
        all_setup["capacitance"] = 6.0
        all_setup["eleak"] = -88.0
    elif new_setup == "480353286":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_480353286/reconstruction.swc"
        all_setup["rotation"] = 170
        all_setup["v_init"] = -88.90625
        all_setup["gleak"] = 1e-4
        all_setup["capacitance"] = 6.0
        all_setup["eleak"] = -95.0
    elif new_setup == "473601979":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_473601979/reconstruction.swc"
        all_setup["rotation"] = 195
        all_setup["v_init"] = -89.06251
        all_setup["gleak"] = 1e-4
        all_setup["capacitance"] = 2.5
        all_setup["eleak"] = -95.0

    

    return all_setup


def build_passive_cell(swc_fname, rotation, capacitance, eleak, gleak, ncomp=4):
    cell = jx.read_swc(swc_fname, ncomp=ncomp, max_branch_len=300.0, assign_groups=True)
    cell.rotate(rotation)
    soma_inds = np.unique(cell.soma.nodes.global_branch_index).tolist()
    apical_inds = np.unique(cell.apical.nodes.global_branch_index).tolist()

    cell.insert(Leak())
   

    ########## WHOLE CELL  ##########
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", capacitance)
    cell.set("Leak_eLeak", eleak)
    cell.set("Leak_gLeak", gleak)
    return cell

def build_cell(swc_fname, rotation, capacitance, eleak, gleak, ncomp=4):
    cell = jx.read_swc(swc_fname, ncomp=ncomp, max_branch_len=300.0, assign_groups=True)
    cell.rotate(rotation)
    soma_inds = np.unique(cell.soma.nodes.global_branch_index).tolist()
    apical_inds = np.unique(cell.apical.nodes.global_branch_index).tolist()

    ########## APICAL ##########
    cell.insert(Leak())
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))
    for b in apical_inds:
        for c in range(4):
            distance = (
                cell.branch(b)
                .comp(c)
                .distance(cell.branch(soma_inds[0]).loc(0.5))
            )
            cond = (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5
            cell.branch(b).comp(c).set("apical_H_gH", cond)

    ########## SOMA ##########
    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))

    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)

    ########## BASAL ##########
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)

    # ########## AXON ##########
    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))

    ca_dynamics_axonal = CaNernstReversal()
    ca_dynamics_axonal.channel_constants["T"] = 307.15
    cell.axon.insert(ca_dynamics)

    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)

    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))
    cell.axon.insert(CaPump().change_name("axonal_CaPump"))

    ########## WHOLE CELL  ##########
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", capacitance)
    cell.set("Leak_eLeak", eleak)
    cell.set("Leak_gLeak", gleak)
    return cell


def setup_simulator(cell, cut = 0, t_max=200, v_init = -85.6, i_amp=100.0):
    i_dur = 1000.0
    time_vec = np.arange(0, t_max+2*dt, dt)
    time_vec = time_vec[cut:]
    levels = 3
    checkpoints = [int(np.ceil(len(time_vec)**(1/levels)).item()) for _ in range(levels)]
    
    # Build cell with approriate stimuli.
    cell.delete_stimuli()
    cell.delete_recordings()

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    current = current[cut:]

    #cell.soma[0,0].stimulate(current)  # Stimulate soma
    cell.soma.branch(0).comp(0).stimulate(current)
    #cell.soma[0,0].record()
    cell.soma.branch(0).comp(0).record()
    cell.set("v", v_init)
    cell.init_states()

    def simulate(params):
        pstate = None

        pstate = cell.data_set("HVA_tau", 10 ** params[0], pstate)
        pstate = cell.data_set("LVA_tau", 10 ** params[1], pstate)        
        pstate = cell.data_set("vt", params[2], pstate)
        pstate = cell.data_set("eK", params[3], pstate)
        pstate = cell.data_set("eNa", params[4], pstate)

        pstate = cell.soma.data_set("somatic_NaTs2T_gNaTs2T", params[5], pstate)
        pstate = cell.soma.data_set("somatic_SKv3_1_gSKv3_1", params[6], pstate)
        pstate = cell.soma.data_set("somatic_SKE2_gSKE2", params[7], pstate)
        pstate = cell.soma.data_set("somatic_CaHVA_gCaHVA", params[8], pstate)
        pstate = cell.soma.data_set("somatic_CaLVA_gCaLVA", params[9], pstate)
        pstate = cell.soma.data_set("somatic_CaPump_gamma", params[10], pstate)
        pstate = cell.soma.data_set("somatic_CaPump_decay", 10 ** params[11], pstate)

        pstate = cell.apical.data_set("apical_NaTs2T_gNaTs2T", params[12], pstate)
        pstate = cell.apical.data_set("apical_SKv3_1_gSKv3_1", params[13], pstate)
        pstate = cell.apical.data_set("apical_M_gM", params[14], pstate)
        pstate = cell.apical.data_set("apical_M_tau", 10 ** params[15], pstate)

        pstate = cell.axon.data_set("axonal_NaTaT_gNaTaT", params[16], pstate)
        pstate = cell.axon.data_set("axonal_KPst_gKPst", params[17], pstate)
        pstate = cell.axon.data_set("axonal_KTst_gKTst", params[18], pstate)
        pstate = cell.axon.data_set("axonal_SKE2_gSKE2", params[19], pstate)
        pstate = cell.axon.data_set("axonal_SKv3_1_gSKv3_1", params[20], pstate)
        pstate = cell.axon.data_set("axonal_CaHVA_gCaHVA", params[21], pstate)
        pstate = cell.axon.data_set("axonal_CaLVA_gCaLVA", params[22], pstate)
        pstate = cell.axon.data_set("axonal_CaPump_gamma", params[23], pstate)
        pstate = cell.axon.data_set("axonal_CaPump_decay", 10 ** params[24], pstate)


        return jx.integrate(cell, param_state=pstate, checkpoint_lengths=checkpoints)
    
    return simulate

def setup_passive_simulator(cell, t_max=200, v_init = -85.6, i_amp=100.0):
    i_dur = 1000.0
    time_vec = np.arange(0, t_max+2*dt, dt)
    levels = 3
    checkpoints = [int(np.ceil(len(time_vec)**(1/levels)).item()) for _ in range(levels)]
    
    # Build cell with approriate stimuli.
    cell.delete_stimuli()
    cell.delete_recordings()

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)

    # Stimulate soma
    cell.soma.branch(0).comp(0).stimulate(current)
    cell.soma.branch(0).comp(0).record()
    cell.set("v", v_init)
    cell.init_states()

    def simulate():
        pstate = None
        return jx.integrate(cell, param_state=pstate, checkpoint_lengths=checkpoints)
    
    return simulate

def setup_simulator_step(cell, cut = 0, t_max=200, v_init = -85.6, i_amp=100.0, params = None):
    i_dur = 1000.0
    time_vec = np.arange(0, t_max+2*dt, dt)
    time_vec = time_vec[cut:]
    levels = 3
    checkpoints = [int(np.ceil(len(time_vec)**(1/levels)).item()) for _ in range(levels)]
    
    # Build cell with approriate stimuli.
    # cell.delete_stimuli()
    # cell.delete_recordings()

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    current = current[cut:]

    # cell.stimulate(current)
    # cell.record()
    # cell.set("v", v_init)
    # cell.init_states()

    pstate = None

    pstate = cell.data_set("HVA_tau", 10 ** params[0], pstate)
    pstate = cell.data_set("LVA_tau", 10 ** params[1], pstate)        
    pstate = cell.data_set("vt", params[2], pstate)
    pstate = cell.data_set("eK", params[3], pstate)
    pstate = cell.data_set("eNa", params[4], pstate)

    pstate = cell.soma.data_set("somatic_NaTs2T_gNaTs2T", params[5], pstate)
    pstate = cell.soma.data_set("somatic_SKv3_1_gSKv3_1", params[6], pstate)
    pstate = cell.soma.data_set("somatic_SKE2_gSKE2", params[7], pstate)
    pstate = cell.soma.data_set("somatic_CaHVA_gCaHVA", params[8], pstate)
    pstate = cell.soma.data_set("somatic_CaLVA_gCaLVA", params[9], pstate)
    pstate = cell.soma.data_set("somatic_CaPump_gamma", params[10], pstate)
    pstate = cell.soma.data_set("somatic_CaPump_decay", 10 ** params[11], pstate)

    pstate = cell.apical.data_set("apical_NaTs2T_gNaTs2T", params[12], pstate)
    pstate = cell.apical.data_set("apical_SKv3_1_gSKv3_1", params[13], pstate)
    pstate = cell.apical.data_set("apical_M_gM", params[14], pstate)
    pstate = cell.apical.data_set("apical_M_tau", 10 ** params[15], pstate)

    pstate = cell.axon.data_set("axonal_NaTaT_gNaTaT", params[16], pstate)
    pstate = cell.axon.data_set("axonal_KPst_gKPst", params[17], pstate)
    pstate = cell.axon.data_set("axonal_KTst_gKTst", params[18], pstate)
    pstate = cell.axon.data_set("axonal_SKE2_gSKE2", params[19], pstate)
    pstate = cell.axon.data_set("axonal_SKv3_1_gSKv3_1", params[20], pstate)
    pstate = cell.axon.data_set("axonal_CaHVA_gCaHVA", params[21], pstate)
    pstate = cell.axon.data_set("axonal_CaLVA_gCaLVA", params[22], pstate)
    pstate = cell.axon.data_set("axonal_CaPump_gamma", params[23], pstate)
    pstate = cell.axon.data_set("axonal_CaPump_decay", 10 ** params[24], pstate)


    init_fn, step_fn = build_init_and_step_fn(cell)
    states, all_params = init_fn([], None, pstate)  # no pstate
    step_fn_ = jax.jit(step_fn)


    rec_inds = cell.recordings.rec_index.to_numpy()
    rec_states = cell.recordings.state.to_numpy()


    def simulate_step(state, input_current):
        externals = {'i': input_current}
        new_state = step_fn_(state, all_params, externals, delta_t=dt)

        recs = jnp.asarray([
            new_state[rec_state][rec_ind]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ])

        return new_state, recs
    
    return states, simulate_step, current, time_vec


def get_experimental_data(setup, cut = 0, t_max=200):
    with open(f"{base_path}/cell_types/specimen_{setup}/ephys_01.pkl", "rb") as handle:
        ephys = pickle.load(handle)

    dt_stim = np.mean(np.diff(ephys["time"]))
    dt_difference = dt / dt_stim / 1000
    print("dt_difference", dt_difference)
    junction_potential = -14.0

    ephys_stim = ephys["stimulus"][::int(dt_difference)]
    ephys_rec = ephys["response"][::int(dt_difference)] + junction_potential
    ephys_time_vec = ephys["time"][::int(dt_difference)]

    time_pad_on = 50.0
    time_pad_off = 150.0

    stim_onset = np.where(ephys_stim > 0.05)[0][0]
    protocol_start = int(stim_onset - time_pad_on / 0.025)

    stim_offset = np.where(ephys_stim < 0.05)[0]
    stim_offset = stim_offset[stim_offset > 20_000][0]
    protocol_end = int(stim_offset + time_pad_off / 0.025)

    ephys_stim = ephys_stim[protocol_start:protocol_end]
    ephys_rec = ephys_rec[protocol_start:protocol_end]
    ephys_time_vec = ephys_time_vec[protocol_start:protocol_end] * 1000
    ephys_time_vec -= ephys_time_vec[0]
    
    # cut_off = int((t_max+2*dt)/dt)
    cut_off = round((t_max+2*dt)/dt)
    
    #global i_amp
    i_amp = np.max(ephys_stim)
    print(f"Amplitude stimulus: {i_amp}")

    return ephys_time_vec[cut:cut_off], ephys_rec[cut:cut_off], i_amp


def transform_uniform_to_normal(
    lower: Array, upper: Array
) -> Tuple[Callable, Callable]:
    def transform(params: Array) -> Array:
        p = (params - lower) / (upper - lower)
        eps = jax.scipy.stats.norm.ppf(p)
        return eps

    def inv_transform(params: Array) -> Array:
        u = jax.scipy.stats.norm.cdf(params)
        return u * (upper - lower) + lower

    return transform, inv_transform


def get_prior(lowers, uppers, transform_params: lambda x: x):
    def sample_prior(key):
        u = jax.random.uniform(key, shape=lowers.shape, minval=lowers, maxval=uppers)
        u = transform_params(u)
        return u
    return sample_prior


def get_bounds():
    bounds = {}

    #### GLOBAL PARAMETERS ####
    bounds["HVA_tau"] = [log10(0.2), log10(5.0)]
    bounds["LVA_tau"] = [log10(0.2), log10(5.0)]
    bounds["vt"] = [0.0, 10.0]
    bounds["eK"] = [-100.0, -70.0]
    bounds["eNa"] = [40.0, 60.0]

    #### SOMATIC ####
    bounds["somatic_NaTs2T_gNaTs2T"] = [0.0, 6.0]
    bounds["somatic_SKv3_1_gSKv3_1"] = [0.25, 1.0]  # [0, 1]
    bounds["somatic_SKE2_gSKE2"] = [0, 0.1]
    bounds["somatic_CaHVA_gCaHVA"] = [0, 0.001]
    bounds["somatic_CaLVA_gCaLVA"] = [0, 0.01]
    bounds["somatic_CaPump_gamma"] = [0.0005, 0.05]
    bounds["somatic_CaPump_decay"] = [log10(1), log10(100)]  # [5, 100]

    #### APICAL ####
    bounds["apical_NaTs2T_gNaTs2T"] = [0, 0.04]
    bounds["apical_SKv3_1_gSKv3_1"] = [0, 0.001]
    bounds["apical_M_gM"] = [0, 0.1]
    bounds["apical_M_tau"] = [log10(0.2), log10(5.0)]  # Newly added parameter.

    #### AXONAL ####
    bounds["axonal_NaTaT_gNaTaT"] = [0.0, 6.0]
    bounds["axonal_KPst_gKPst"] = [0.0, 1.0]
    bounds["axonal_KTst_gKTst"] = [0.0, 0.1]
    bounds["axonal_SKE2_gSKE2"] = [0.0, 0.1]
    bounds["axonal_SKv3_1_gSKv3_1"] = [0.0, 2.0]
    bounds["axonal_CaHVA_gCaHVA"] = [0, 0.001]
    bounds["axonal_CaLVA_gCaLVA"] = [0, 0.01]
    bounds["axonal_CaPump_gamma"] = [0.0005, 0.05]
    bounds["axonal_CaPump_decay"] = [log10(1), log10(100)]  # [5, 100]

    # Number of params:
    print(f"Number of parameters: {len(bounds.keys())}")
    lowers_and_uppers = jnp.asarray(list(bounds.values()))

    names = list(bounds.keys())
    lowers = lowers_and_uppers[:, 0]
    uppers = lowers_and_uppers[:, 1]
    return names, lowers, uppers
