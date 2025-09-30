import os
from tkinter.font import names
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

from jaxley.channels import Leak, Na, K, CaT, CaL, Km
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
import equinox as eqx

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
        all_setup["gleak"] = 5.545e-5 #active models grid search, cap = 1.0: 1e-4 #grid search on gleak, cap and length result = 7.22e-5 #original = 1e-4
        all_setup["capacitance"] = 1.0 #grid search on gleak, cap and length result = 7.5 #original = 2.0
        all_setup["eleak"] = -88.0
        all_setup["length"] = 197 #142 #grid search on gleak, cap and length result = 31 #without channels: 113.5, radius = 8
    elif new_setup == "485574832":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_485574832/reconstruction.swc"
        all_setup["rotation"] = 155
        all_setup["v_init"] = -85.6
        all_setup["gleak"] = 2.09e-5 #1.25e-4 #grid search on gleak, cap and length result =8.57e-5 #original = 1e-4
        all_setup["capacitance"] = 1.0#grid search on gleak, cap and length result = 67 #original = 6.0
        all_setup["eleak"] = -88.0
        all_setup["length"] = 410 #220 #grid search on gleak, cap and length result = 7 #without channels: 86.5, radius = 8
    elif new_setup == "480353286":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_480353286/reconstruction.swc"
        all_setup["rotation"] = 170
        all_setup["v_init"] = -88.90625
        all_setup["gleak"] = 3.07e-5 #8.06e-5#grid search on gleak, cap and length result = 1.2e-4 #original = 1e-4
        all_setup["capacitance"] = 1.0 #grid search on gleak, cap and length result = 26 #original = 6.0
        all_setup["eleak"] = -95.0
        all_setup["length"] = 277 #216 #grid search on gleak, cap and length result = 16 #without channels: 81.5, radius = 8
    elif new_setup == "473601979":
        all_setup["swc_fname"] = f"{base_path}/cell_types/specimen_473601979/reconstruction.swc"
        all_setup["rotation"] = 195
        all_setup["v_init"] = -89.06251
        all_setup["gleak"] = 5.09e-5 #1.05e-4 #grid search on gleak, cap and length result = 1.75e-4 #original = 1e-4
        all_setup["capacitance"] = 1.0 #grid search on gleak, cap and length result = 16.5 #original = 6.0
        all_setup["eleak"] = -95.0
        all_setup["length"] = 98 #80 #grid search on gleak, cap and length result = 7.5 #without channels: 50.5, radius = 8


    return all_setup

def build_passive_cell(capacitance, eleak, gleak, length):
    cell = jx.Cell()
    cell.set("radius", 8)
    cell.set("length", length)

    cell.insert(Leak())

    ########## WHOLE CELL  ##########
    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", capacitance)
    cell.set("Leak_eLeak", eleak)
    cell.set("Leak_gLeak", gleak)
    return cell

def build_Na_K_cell(capacitance, eleak, gleak, length, Na_gating_params = None, K_gating_params = None, Ca_gating_params = None, use_CaT = False, use_CaL = False, use_Km = False):

    cell = jx.Cell()
    cell.set("radius", 8)
    cell.set("length", length)

    cell.insert(Leak())
    cell.insert(Na(gating_params=Na_gating_params))
    cell.insert(K(gating_params=K_gating_params))
    if(use_CaT):
        cell.insert(CaT(gating_params=Ca_gating_params))
    if(use_CaL):
        cell.insert(CaL())
    if(use_Km):
        cell.insert(Km())

    cell.set("axial_resistivity", 100.0)
    cell.set("capacitance", capacitance)
    cell.set("Leak_eLeak", eleak)
    cell.set("Leak_gLeak", 1e-4)

    return cell

def build_cell(capacitance, eleak, gleak, length, include_H = True, include_M = True, include_Ca = True, include_SKv = True):
    cell = jx.Cell()
    cell.set("radius", 8)
    cell.set("length", length)

    if(include_H):
        cell.insert(H())
    if(include_M):
        cell.insert(M())
    cell.insert(Leak())

    ########## SOMATIC CHANNELS ##########
    cell.insert(NaTs2T())
    if(include_SKv):
        cell.insert(SKv3_1())
    cell.insert(SKE2())
    if(include_Ca):
        ca_dynamics = CaNernstReversal()
        ca_dynamics.channel_constants["T"] = 307.15
        cell.insert(ca_dynamics)
        cell.insert(CaPump())
        cell.insert(CaHVA())
        cell.insert(CaLVA())

        cell.set("CaCon_i", 5e-05)
        cell.set("CaCon_e", 2.0)

    # ########## BASAL ##########
    # cell.basal.insert(H().change_name("basal_H"))
    # cell.basal.set("basal_H_gH", 8e-5)

    # # ########## AXON ##########
    # cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    # cell.axon.insert(KPst().change_name("axonal_KPst"))
    # cell.axon.insert(KTst().change_name("axonal_KTst"))
    # cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    # cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))

    # ca_dynamics_axonal = CaNernstReversal()
    # ca_dynamics_axonal.channel_constants["T"] = 307.15
    # cell.axon.insert(ca_dynamics)
    # cell.set("CaCon_i", 5e-05)
    # cell.set("CaCon_e", 2.0)
    # cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    # cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))
    # cell.axon.insert(CaPump().change_name("axonal_CaPump"))

    ########## WHOLE CELL  ##########
    if(include_H):
        cell.set("H_gH", 8e-5)
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

    cell.stimulate(current)
    cell.record()
    cell.set("v", v_init)
    cell.init_states()

    def simulate(params, x_o, thr = None, include_H = True, include_M = True, include_Ca = True, include_SKv = True):
        pstate = None
        if(include_Ca):
            pstate = cell.data_set("HVA_tau", 10 ** params[0], pstate)
            pstate = cell.data_set("LVA_tau", 10 ** params[1], pstate)
        pstate = cell.data_set("vt", params[2], pstate)
        pstate = cell.data_set("eK", params[3], pstate)
        pstate = cell.data_set("eNa", params[4], pstate)
        

        pstate = cell.data_set("NaTs2T_gNaTs2T", params[5], pstate)
        if(include_SKv):
            pstate = cell.data_set("SKv3_1_gSKv3_1", params[6], pstate)
        pstate = cell.data_set("SKE2_gSKE2", params[7], pstate)
        if(include_Ca):
            pstate = cell.data_set("CaHVA_gCaHVA", params[8], pstate)
            pstate = cell.data_set("CaLVA_gCaLVA", params[9], pstate)
            pstate = cell.data_set("CaPump_gamma", params[10], pstate)
            pstate = cell.data_set("CaPump_decay", 10 ** params[11], pstate)
        
        if(include_M):
            pstate = cell.data_set("M_gM", params[12], pstate)
            pstate = cell.data_set("M_tau", 10 ** params[13], pstate)
        pstate = cell.data_set("length", params[14], pstate)


        init_fn, step_fn = build_init_and_step_fn(cell)
        states, all_params = init_fn([], None, pstate)
        step_fn_ = jax.jit(step_fn)

        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()

        externals = cell.externals
        externals['i'] = externals['i'][0]

        first_rec = jnp.asarray([
            states[rec_state][rec_ind]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ])
        #recordings = first_rec[None]
        diff = jnp.abs(first_rec[0] - x_o[0])
        x_o = jnp.asarray(x_o)
        
        recording_buffer = jnp.full((len(time_vec), 1), jnp.nan)
        recording_buffer = recording_buffer.at[0].set(first_rec)


        def cond_fn(carry_t):
            _, _, _, diff = carry_t
            return thr == None or diff < thr

        def body_fn(carry_t):
            t, states, recordings, diff = carry_t

            each_externals = {'i': externals['i'][t]}
            new_states = step_fn_(states, all_params, each_externals, delta_t=dt)

            recs = jnp.asarray([
                new_states[rec_state][rec_ind]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ])

            new_diff = diff + jnp.abs(recs[0] - x_o[t + 1])
            new_recordings = recordings.at[t+1].set(recs)
            
            return (t + 1, new_states, new_recordings, new_diff)

        # Initialize loop state
        init_val = (0, states, recording_buffer, diff)

        # Run the while loop
        t_final, final_states, final_recordings, final_diff = eqx.internal.while_loop(
            cond_fun = cond_fn,
            body_fun = body_fn,
            init_val = init_val,
            max_steps = len(time_vec)-1,
            kind = "bounded"
        )

        rec = final_recordings.T
        return rec, t_final+1

        # return jx.integrate(cell, param_state = pstate, checkpoint_lengths=checkpoints)
    
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

    cell.stimulate(current)
    cell.record()
    cell.set("v", v_init)
    cell.init_states()

    def simulate(x_o, thr = None):
        pstate = None

        init_fn, step_fn = build_init_and_step_fn(cell)
        states, all_params = init_fn([], None, pstate)
        step_fn_ = jax.jit(step_fn)

        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()

        externals = cell.externals
        externals['i'] = externals['i'][0]

        first_rec = jnp.asarray([
            states[rec_state][rec_ind]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ])
        #recordings = first_rec[None]
        diff = jnp.abs(first_rec[0] - x_o[0])
        x_o = jnp.asarray(x_o)
        
        recording_buffer = jnp.full((len(time_vec), 1), jnp.nan)
        recording_buffer = recording_buffer.at[0].set(first_rec)


        def cond_fn(carry_t):
            _, _, _, diff = carry_t
            return thr == None or diff < thr

        def body_fn(carry_t):
            t, states, recordings, diff = carry_t

            each_externals = {'i': externals['i'][t]}
            new_states = step_fn_(states, all_params, each_externals, delta_t=dt)

            recs = jnp.asarray([
                new_states[rec_state][rec_ind]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ])

            new_diff = diff + jnp.abs(recs[0] - x_o[t + 1])
            new_recordings = recordings.at[t+1].set(recs)
            
            return (t + 1, new_states, new_recordings, new_diff)

        # Initialize loop state
        init_val = (0, states, recording_buffer, diff)

        # Run the while loop
        t_final, final_states, final_recordings, final_diff = eqx.internal.while_loop(
            cond_fun = cond_fn,
            body_fun = body_fn,
            init_val = init_val,
            max_steps = len(time_vec)-1,
            kind = "bounded"
        )

        rec = final_recordings.T
        return rec, t_final+1

        # return jx.integrate(cell, param_state = pstate, checkpoint_lengths=checkpoints)
    
    return simulate

def setup_Na_K_simulator(cell, cut = 0, t_max=200, v_init = -85.6, i_amp=100.0):
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

    cell.stimulate(current)
    cell.record()
    cell.set("v", v_init)
    cell.init_states()

    def simulate(params, x_o, thr = None, gating = False, Na_gating_params = None, K_gating_params = None, powers = None):
        pstate = None
        
        pstate = cell.data_set("vt", params[0], pstate)
        pstate = cell.data_set("eK", params[1], pstate)
        pstate = cell.data_set("eNa", params[2], pstate)
        pstate = cell.data_set("eCa", params[3], pstate)
        pstate = cell.data_set("Km_taumax", params[4], pstate)
        

        pstate = cell.data_set("Na_gNa", params[5], pstate)
        pstate = cell.data_set("K_gK", params[6], pstate)
        pstate = cell.data_set("CaT_gCaT", params[7], pstate)
        # pstate = cell.data_set("CaL_gCaL", params[8], pstate)
        pstate = cell.data_set("Km_gKm", params[9], pstate)
        pstate = cell.data_set("length", params[10], pstate)

        if(gating):
            pstate = cell.data_set('m_alpha_params_0', Na_gating_params[0], pstate)
            pstate = cell.data_set('m_alpha_params_1', Na_gating_params[1], pstate)
            pstate = cell.data_set('m_alpha_params_2', Na_gating_params[2], pstate)
            pstate = cell.data_set('m_alpha_params_3', Na_gating_params[3], pstate)
            pstate = cell.data_set('m_alpha_params_4', Na_gating_params[4], pstate)
            pstate = cell.data_set('m_alpha_params_5', Na_gating_params[5], pstate)
            pstate = cell.data_set('m_alpha_params_6', Na_gating_params[6], pstate)
            pstate = cell.data_set('m_alpha_params_7', Na_gating_params[7], pstate)
            pstate = cell.data_set('m_alpha_params_8', Na_gating_params[8], pstate)
            pstate = cell.data_set('m_alpha_params_9', Na_gating_params[9], pstate)
            pstate = cell.data_set('m_beta_params_0', Na_gating_params[10], pstate)
            pstate = cell.data_set('m_beta_params_1', Na_gating_params[11], pstate)
            pstate = cell.data_set('m_beta_params_2', Na_gating_params[12], pstate)
            pstate = cell.data_set('m_beta_params_3', Na_gating_params[13], pstate)
            pstate = cell.data_set('m_beta_params_4', Na_gating_params[14], pstate)
            pstate = cell.data_set('m_beta_params_5', Na_gating_params[15], pstate)
            pstate = cell.data_set('m_beta_params_6', Na_gating_params[16], pstate)
            pstate = cell.data_set('m_beta_params_7', Na_gating_params[17], pstate)
            pstate = cell.data_set('m_beta_params_8', Na_gating_params[18], pstate)
            pstate = cell.data_set('m_beta_params_9', Na_gating_params[19], pstate)
            pstate = cell.data_set('m_power', powers[0], pstate)

            pstate = cell.data_set('h_alpha_params_0', Na_gating_params[20], pstate)
            pstate = cell.data_set('h_alpha_params_1', Na_gating_params[21], pstate)
            pstate = cell.data_set('h_alpha_params_2', Na_gating_params[22], pstate)
            pstate = cell.data_set('h_alpha_params_3', Na_gating_params[23], pstate)
            pstate = cell.data_set('h_alpha_params_4', Na_gating_params[24], pstate)
            pstate = cell.data_set('h_alpha_params_5', Na_gating_params[25], pstate)
            pstate = cell.data_set('h_alpha_params_6', Na_gating_params[26], pstate)
            pstate = cell.data_set('h_alpha_params_7', Na_gating_params[27], pstate)
            pstate = cell.data_set('h_alpha_params_8', Na_gating_params[28], pstate)
            pstate = cell.data_set('h_alpha_params_9', Na_gating_params[29], pstate)
            pstate = cell.data_set('h_beta_params_0', Na_gating_params[30], pstate)
            pstate = cell.data_set('h_beta_params_1', Na_gating_params[31], pstate)
            pstate = cell.data_set('h_beta_params_2', Na_gating_params[32], pstate)
            pstate = cell.data_set('h_beta_params_3', Na_gating_params[33], pstate)
            pstate = cell.data_set('h_beta_params_4', Na_gating_params[34], pstate)
            pstate = cell.data_set('h_beta_params_5', Na_gating_params[35], pstate)
            pstate = cell.data_set('h_beta_params_6', Na_gating_params[36], pstate)
            pstate = cell.data_set('h_beta_params_7', Na_gating_params[37], pstate)
            pstate = cell.data_set('h_beta_params_8', Na_gating_params[38], pstate)
            pstate = cell.data_set('h_beta_params_9', Na_gating_params[39], pstate)
            pstate = cell.data_set('h_power', powers[1], pstate)

            pstate = cell.data_set('n_alpha_params_0', K_gating_params[0], pstate)
            pstate = cell.data_set('n_alpha_params_1', K_gating_params[1], pstate)
            pstate = cell.data_set('n_alpha_params_2', K_gating_params[2], pstate)
            pstate = cell.data_set('n_alpha_params_3', K_gating_params[3], pstate)
            pstate = cell.data_set('n_alpha_params_4', K_gating_params[4], pstate)
            pstate = cell.data_set('n_alpha_params_5', K_gating_params[5], pstate)
            pstate = cell.data_set('n_alpha_params_6', K_gating_params[6], pstate)
            pstate = cell.data_set('n_alpha_params_7', K_gating_params[7], pstate)
            pstate = cell.data_set('n_alpha_params_8', K_gating_params[8], pstate)
            pstate = cell.data_set('n_alpha_params_9', K_gating_params[9], pstate)
            pstate = cell.data_set('n_beta_params_0', K_gating_params[10], pstate)
            pstate = cell.data_set('n_beta_params_1', K_gating_params[11], pstate)
            pstate = cell.data_set('n_beta_params_2', K_gating_params[12], pstate)
            pstate = cell.data_set('n_beta_params_3', K_gating_params[13], pstate)
            pstate = cell.data_set('n_beta_params_4', K_gating_params[14], pstate)
            pstate = cell.data_set('n_beta_params_5', K_gating_params[15], pstate)
            pstate = cell.data_set('n_beta_params_6', K_gating_params[16], pstate)
            pstate = cell.data_set('n_beta_params_7', K_gating_params[17], pstate)
            pstate = cell.data_set('n_beta_params_8', K_gating_params[18], pstate)
            pstate = cell.data_set('n_beta_params_9', K_gating_params[19], pstate)
            pstate = cell.data_set('n_power', powers[2], pstate)

        init_fn, step_fn = build_init_and_step_fn(cell)
        states, all_params = init_fn([], None, pstate)
        step_fn_ = jax.jit(step_fn)

        rec_inds = cell.recordings.rec_index.to_numpy()
        rec_states = cell.recordings.state.to_numpy()

        externals = cell.externals
        externals['i'] = externals['i'][0]

        first_rec = jnp.asarray([
            states[rec_state][rec_ind]
            for rec_state, rec_ind in zip(rec_states, rec_inds)
        ])
        #recordings = first_rec[None]
        diff = jnp.abs(first_rec[0] - x_o[0])
        x_o = jnp.asarray(x_o)
        
        recording_buffer = jnp.full((len(time_vec), 1), jnp.nan)
        recording_buffer = recording_buffer.at[0].set(first_rec)


        def cond_fn(carry_t):
            _, _, _, diff = carry_t
            return thr == None or diff < thr

        def body_fn(carry_t):
            t, states, recordings, diff = carry_t

            each_externals = {'i': externals['i'][t]}
            new_states = step_fn_(states, all_params, each_externals, delta_t=dt)

            recs = jnp.asarray([
                new_states[rec_state][rec_ind]
                for rec_state, rec_ind in zip(rec_states, rec_inds)
            ])

            new_diff = diff + jnp.abs(recs[0] - x_o[t + 1])
            new_recordings = recordings.at[t+1].set(recs)
            
            return (t + 1, new_states, new_recordings, new_diff)

        # Initialize loop state
        init_val = (0, states, recording_buffer, diff)

        # Run the while loop
        t_final, final_states, final_recordings, final_diff = eqx.internal.while_loop(
            cond_fun = cond_fn,
            body_fun = body_fn,
            init_val = init_val,
            max_steps = len(time_vec)-1,
            kind = "bounded"
        )

        rec = final_recordings.T
        return rec, t_final+1

        # return jx.integrate(cell, param_state = pstate, checkpoint_lengths=checkpoints)
    
    return simulate



def setup_simulator_step(cell, cut = 0, t_max=200, v_init = -85.6, i_amp=100.0, params = None, include_H = True, include_M = True, include_Ca = True, include_SKv = True):
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
    if(include_Ca):
        pstate = cell.data_set("HVA_tau", 10 ** params[0], pstate)
        pstate = cell.data_set("LVA_tau", 10 ** params[1], pstate)
    pstate = cell.data_set("vt", params[2], pstate)
    pstate = cell.data_set("eK", params[3], pstate)
    pstate = cell.data_set("eNa", params[4], pstate)
    

    pstate = cell.data_set("NaTs2T_gNaTs2T", params[5], pstate)
    if(include_SKv):
        pstate = cell.data_set("SKv3_1_gSKv3_1", params[6], pstate)
    pstate = cell.data_set("SKE2_gSKE2", params[7], pstate)
    if(include_Ca):
        pstate = cell.data_set("CaHVA_gCaHVA", params[8], pstate)
        pstate = cell.data_set("CaLVA_gCaLVA", params[9], pstate)
        pstate = cell.data_set("CaPump_gamma", params[10], pstate)
        pstate = cell.data_set("CaPump_decay", 10 ** params[11], pstate)
    
    if(include_M):
        pstate = cell.data_set("M_gM", params[12], pstate)
        pstate = cell.data_set("M_tau", 10 ** params[13], pstate)
    pstate = cell.data_set("length", params[14], pstate)

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


def setup_passive_simulator_step(cell, t_max=200, v_init = -85.6, i_amp=100.0, pstate = None):
    i_dur = 1000.0
    time_vec = np.arange(0, t_max+2*dt, dt)
    levels = 3
    checkpoints = [int(np.ceil(len(time_vec)**(1/levels)).item()) for _ in range(levels)]
    
    # Build cell with approriate stimuli.
    cell.delete_stimuli()
    cell.delete_recordings()

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)

    cell.stimulate(current)
    cell.record()
    cell.set("v", v_init)
    cell.init_states()

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

def setup_Na_K_simulator_step(cell, cut = 0, t_max=200, v_init = -85.6, i_amp=100.0, gating = False, params = None, Na_gating_params = None, K_gating_params = None, Ca_gating_params = None, powers = None):
    i_dur = 1000.0
    time_vec = np.arange(0, t_max+2*dt, dt)
    time_vec = time_vec[cut:]
    levels = 3
    checkpoints = [int(np.ceil(len(time_vec)**(1/levels)).item()) for _ in range(levels)]
    

    current = jx.step_current(i_delay, i_dur, i_amp, dt, t_max)
    current = current[cut:]


    pstate = None
    pstate = cell.data_set("vt", params[0], pstate)
    pstate = cell.data_set("eK", params[1], pstate)
    pstate = cell.data_set("eNa", params[2], pstate)
    pstate = cell.data_set("eCa", params[3], pstate)
    pstate = cell.data_set("Km_taumax", params[4], pstate)
    

    pstate = cell.data_set("Na_gNa", params[5], pstate)
    pstate = cell.data_set("K_gK", params[6], pstate)
    pstate = cell.data_set("CaT_gCaT", params[7], pstate)
    # pstate = cell.data_set("CaL_gCaL", params[8], pstate)
    pstate = cell.data_set("Km_gKm", params[9], pstate)
    pstate = cell.data_set("length", params[10], pstate)

    if(gating):
        pstate = cell.data_set('m_alpha_params_0', Na_gating_params[0], pstate)
        pstate = cell.data_set('m_alpha_params_1', Na_gating_params[1], pstate)
        pstate = cell.data_set('m_alpha_params_2', Na_gating_params[2], pstate)
        pstate = cell.data_set('m_alpha_params_3', Na_gating_params[3], pstate)
        pstate = cell.data_set('m_alpha_params_4', Na_gating_params[4], pstate)
        pstate = cell.data_set('m_alpha_params_5', Na_gating_params[5], pstate)
        pstate = cell.data_set('m_alpha_params_6', Na_gating_params[6], pstate)
        pstate = cell.data_set('m_alpha_params_7', Na_gating_params[7], pstate)
        pstate = cell.data_set('m_alpha_params_8', Na_gating_params[8], pstate)
        pstate = cell.data_set('m_alpha_params_9', Na_gating_params[9], pstate)
        pstate = cell.data_set('m_beta_params_0', Na_gating_params[10], pstate)
        pstate = cell.data_set('m_beta_params_1', Na_gating_params[11], pstate)
        pstate = cell.data_set('m_beta_params_2', Na_gating_params[12], pstate)
        pstate = cell.data_set('m_beta_params_3', Na_gating_params[13], pstate)
        pstate = cell.data_set('m_beta_params_4', Na_gating_params[14], pstate)
        pstate = cell.data_set('m_beta_params_5', Na_gating_params[15], pstate)
        pstate = cell.data_set('m_beta_params_6', Na_gating_params[16], pstate)
        pstate = cell.data_set('m_beta_params_7', Na_gating_params[17], pstate)
        pstate = cell.data_set('m_beta_params_8', Na_gating_params[18], pstate)
        pstate = cell.data_set('m_beta_params_9', Na_gating_params[19], pstate)
        pstate = cell.data_set('m_power', powers[0], pstate)

        pstate = cell.data_set('h_alpha_params_0', Na_gating_params[20], pstate)
        pstate = cell.data_set('h_alpha_params_1', Na_gating_params[21], pstate)
        pstate = cell.data_set('h_alpha_params_2', Na_gating_params[22], pstate)
        pstate = cell.data_set('h_alpha_params_3', Na_gating_params[23], pstate)
        pstate = cell.data_set('h_alpha_params_4', Na_gating_params[24], pstate)
        pstate = cell.data_set('h_alpha_params_5', Na_gating_params[25], pstate)
        pstate = cell.data_set('h_alpha_params_6', Na_gating_params[26], pstate)
        pstate = cell.data_set('h_alpha_params_7', Na_gating_params[27], pstate)
        pstate = cell.data_set('h_alpha_params_8', Na_gating_params[28], pstate)
        pstate = cell.data_set('h_alpha_params_9', Na_gating_params[29], pstate)
        pstate = cell.data_set('h_beta_params_0', Na_gating_params[30], pstate)
        pstate = cell.data_set('h_beta_params_1', Na_gating_params[31], pstate)
        pstate = cell.data_set('h_beta_params_2', Na_gating_params[32], pstate)
        pstate = cell.data_set('h_beta_params_3', Na_gating_params[33], pstate)
        pstate = cell.data_set('h_beta_params_4', Na_gating_params[34], pstate)
        pstate = cell.data_set('h_beta_params_5', Na_gating_params[35], pstate)
        pstate = cell.data_set('h_beta_params_6', Na_gating_params[36], pstate)
        pstate = cell.data_set('h_beta_params_7', Na_gating_params[37], pstate)
        pstate = cell.data_set('h_beta_params_8', Na_gating_params[38], pstate)
        pstate = cell.data_set('h_beta_params_9', Na_gating_params[39], pstate)
        #pstate = cell.data_set('h_power', powers[1], pstate)

        pstate = cell.data_set('n_alpha_params_0', K_gating_params[0], pstate)
        pstate = cell.data_set('n_alpha_params_1', K_gating_params[1], pstate)
        pstate = cell.data_set('n_alpha_params_2', K_gating_params[2], pstate)
        pstate = cell.data_set('n_alpha_params_3', K_gating_params[3], pstate)
        pstate = cell.data_set('n_alpha_params_4', K_gating_params[4], pstate)
        pstate = cell.data_set('n_alpha_params_5', K_gating_params[5], pstate)
        pstate = cell.data_set('n_alpha_params_6', K_gating_params[6], pstate)
        pstate = cell.data_set('n_alpha_params_7', K_gating_params[7], pstate)
        pstate = cell.data_set('n_alpha_params_8', K_gating_params[8], pstate)
        pstate = cell.data_set('n_alpha_params_9', K_gating_params[9], pstate)
        pstate = cell.data_set('n_beta_params_0', K_gating_params[10], pstate)
        pstate = cell.data_set('n_beta_params_1', K_gating_params[11], pstate)
        pstate = cell.data_set('n_beta_params_2', K_gating_params[12], pstate)
        pstate = cell.data_set('n_beta_params_3', K_gating_params[13], pstate)
        pstate = cell.data_set('n_beta_params_4', K_gating_params[14], pstate)
        pstate = cell.data_set('n_beta_params_5', K_gating_params[15], pstate)
        pstate = cell.data_set('n_beta_params_6', K_gating_params[16], pstate)
        pstate = cell.data_set('n_beta_params_7', K_gating_params[17], pstate)
        pstate = cell.data_set('n_beta_params_8', K_gating_params[18], pstate)
        pstate = cell.data_set('n_beta_params_9', K_gating_params[19], pstate)
        pstate = cell.data_set('n_power', powers[2], pstate)

        pstate = cell.data_set('u_inf_params_0', Ca_gating_params[0], pstate)
        pstate = cell.data_set('u_inf_params_1', Ca_gating_params[1], pstate)
        pstate = cell.data_set('u_inf_params_2', Ca_gating_params[2], pstate)
        pstate = cell.data_set('u_inf_params_3', Ca_gating_params[3], pstate)
        pstate = cell.data_set('u_inf_params_4', Ca_gating_params[4], pstate)
        pstate = cell.data_set('u_inf_params_5', Ca_gating_params[5], pstate)
        pstate = cell.data_set('u_inf_params_6', Ca_gating_params[6], pstate)
        pstate = cell.data_set('u_inf_params_7', Ca_gating_params[7], pstate)
        pstate = cell.data_set('u_inf_params_8', Ca_gating_params[8], pstate)
        pstate = cell.data_set('u_inf_params_9', Ca_gating_params[9], pstate)
        pstate = cell.data_set('u_tau_params_0', Ca_gating_params[10], pstate)
        pstate = cell.data_set('u_tau_params_1', Ca_gating_params[11], pstate)
        pstate = cell.data_set('u_tau_params_2', Ca_gating_params[12], pstate)
        pstate = cell.data_set('u_tau_params_3', Ca_gating_params[13], pstate)
        pstate = cell.data_set('u_tau_params_4', Ca_gating_params[14], pstate)
        pstate = cell.data_set('u_tau_params_5', Ca_gating_params[15], pstate)
        pstate = cell.data_set('u_tau_params_6', Ca_gating_params[16], pstate)
        pstate = cell.data_set('u_tau_params_7', Ca_gating_params[17], pstate)
        pstate = cell.data_set('u_tau_params_8', Ca_gating_params[18], pstate)
        pstate = cell.data_set('u_tau_params_9', Ca_gating_params[19], pstate)




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

    
    
    # bounds["eH"] = [-50, -30]

    #### SOMATIC ####
    bounds["NaTs2T_gNaTs2T"] = [0.0, 1.0] #[0.0, 6.0] # [0.2, 0.25] 
    bounds["SKv3_1_gSKv3_1"] = [0.01, 0.5] #[0.25, 1.0] # [0.01, 0.5]
    bounds["SKE2_gSKE2"] = [0, 0.1]
    bounds["CaHVA_gCaHVA"] = [0, 0.001]
    bounds["CaLVA_gCaLVA"] = [0, 0.01]
    bounds["CaPump_gamma"] = [0.0005, 0.05]
    bounds["CaPump_decay"] = [log10(1), log10(100)]  # [5, 100]

    # #### APICAL ####
    # bounds["apical_NaTs2T_gNaTs2T"] = [0, 0.04]
    # bounds["apical_SKv3_1_gSKv3_1"] = [0, 0.001]
    # bounds["apical_M_gM"] = [0, 0.1]
    # bounds["apical_M_tau"] = [log10(0.2), log10(5.0)]  # Newly added parameter.

    bounds["M_gM"] = [0, 0.1]
    bounds["M_tau"] = [log10(0.2), log10(5.0)]

    # #### AXONAL ####
    # bounds["axonal_NaTaT_gNaTaT"] = [0.0, 6.0]
    # bounds["axonal_KPst_gKPst"] = [0.0, 1.0]
    # bounds["axonal_KTst_gKTst"] = [0.0, 0.1]
    # bounds["axonal_SKE2_gSKE2"] = [0.0, 0.1]
    # bounds["axonal_SKv3_1_gSKv3_1"] = [0.0, 2.0]
    # bounds["axonal_CaHVA_gCaHVA"] = [0, 0.001]
    # bounds["axonal_CaLVA_gCaLVA"] = [0, 0.01]
    # bounds["axonal_CaPump_gamma"] = [0.0005, 0.05]
    # bounds["axonal_CaPump_decay"] = [log10(1), log10(100)]  # [5, 100]

    # bounds["H_gH"] = [0, 0.05]

    bounds['length'] = [10.0, 400.0]

    # Number of params:
    print(f"Number of parameters: {len(bounds.keys())}")
    lowers_and_uppers = jnp.asarray(list(bounds.values()))

    names = list(bounds.keys())
    lowers = lowers_and_uppers[:, 0]
    uppers = lowers_and_uppers[:, 1]
    return names, lowers, uppers

def get_Na_K_bounds():
    bounds = {}

    #### GLOBAL PARAMETERS ####
    bounds["vt"] = [-75, -60] #[0.0, 10.0]
    bounds["eK"] = [-80.0, -70.0]
    bounds["eNa"] = [45.0, 65.0] #[40.0, 60.0]
    bounds["eCa"] = [110, 130]
    bounds["Km_taumax"] = [3000, 5000]
   

    

    #### SOMATIC ####
    bounds["Na_gNa"] = [0.005, 0.05] #[0.005, 0.03] #[0.024, 0.026] #[0.0, 1.0] #[0.0, 6.0]
    bounds["K_gK"] = [1e-4, 0.015] #[0.006, 0.008] #[0.25, 1.0] # [0.01, 0.5]
    bounds["CaT_gCaT"] = [0.2e-4, 0.6e-4]
    bounds["CaL_gCaL"] = [0.01e-3, 0.2e-3]
    bounds["Km_gKm"] = [0.002e-3, 0.6e-3]


    bounds['length'] = [100.0, 400.0]

    # Number of params:
    print(f"Number of parameters: {len(bounds.keys())}")
    lowers_and_uppers = jnp.asarray(list(bounds.values()))

    names = list(bounds.keys())
    lowers = lowers_and_uppers[:, 0]
    uppers = lowers_and_uppers[:, 1]
    return names, lowers, uppers

def get_gating_bounds():
    lowers = jnp.concatenate([jnp.full((20,), -0.1), jnp.full((50,), -0.1), jnp.full((10,), -1.0)]) # m_gate, h_gate, n_gate
    uppers = jnp.concatenate([jnp.full((20,), 0.1), jnp.full((50,), 0.1), jnp.full((10,), 1.0)])
    names = [f"gate_{i}" for i in range(lowers.shape[0])]
    return names, lowers, uppers

def get_power_bounds():
    bounds = {}

    #### GLOBAL PARAMETERS ####
    bounds["m"] = [2.8, 3.2] #[2.0, 4.0]
    bounds["h"] = [0.99, 1.01]
    bounds["n"] = [3.8, 4.2] #[3.0, 5.0]
   
    lowers_and_uppers = jnp.asarray(list(bounds.values()))

    names = list(bounds.keys())
    lowers = lowers_and_uppers[:, 0]
    uppers = lowers_and_uppers[:, 1]
    return names, lowers, uppers