#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 19:53:10 2022

@author: th
"""
import argparse
import os
import random
import sys
import timeit

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from brian2 import *
from scipy import integrate

sys.path.append('../util')
import util
"""
This code is an adapted version of the original code of :
Susin, Eduarda, and Alain Destexhe. 2021. “Integration, Coincidence Detection and Resonance in Networks of Spiking Neurons Expressing Gamma Oscillations and Asynchronous States.” PLoS Computational Biology 17 (9): e1009416.
Alain Destexhe 2009. "Self-sustained asynchronous irregular states and Up–Downstates in thalamic, cortical and thalamocortical networksof nonlinear integrate-and-fire neurons", J Comput Neurosci (2009) 27:493–506DOI 10.1007/s10827-009-0164-4


"""

parser = argparse.ArgumentParser()
parser.add_argument("--sim_time") # in seconds
parser.add_argument("--N")
parser.add_argument("--a")
parser.add_argument("--b")
parser.add_argument("--instance")
parser.add_argument("--noise")

args=parser.parse_args()


params = dict()
params['sim_time'] = float(args.sim_time)
params['a'] = float(args.a)
params['b'] = float(args.b)
params['N'] = int(args.N)
params['noise'] = float(args.noise)



instance = str(args.instance)


print('weight adaptation parameter a (leaky) : ', str(params['a']))
print('weight adaptation parameter b : ', str(params['b']))


root_dir = '../example_data'

curr_dir = root_dir + '/' + str(params['N'])

params['save_fol'] = curr_dir

save_fol = params['save_fol']
save_name = '_'.join([str(params['N']), str(float(params['a'])),str(float(params['b'])), str(float(params['sim_time']))])

save_fol_name = save_fol + '/' + save_name
save_file_path = save_fol_name + '/instances/' + str(instance)
    


# load base simulation
base_sim = np.load(save_file_path+ '.npy', allow_pickle=True).item()
print('base sim result loaded')

def run_sim(params):
    #========================================================================
    
    start_scope() 
    t_simulation = params['sim_time']*second
    print('simulation will generate activity for ' + str(t_simulation)+ ' instance :' + str(instance))
    
    defaultclock.dt = 0.1*ms
    
    
    ################################################################################
    #Network Structure
    ################################################################################
    
    N=params['N']
    
    NE=int(N*4./5.); NI=int(N/5.)
    
    #-----------------------------
    
    prob_Pee=0.02 #(RS->RS)
    prob_Pei=0.02 #(RS->FS)
    
    prob_Pii=0.02 #(FS->FS)
    prob_Pie=0.02 #(FS->RS)
    
    
    ################################################################################
    #Reescaling Synaptic Weights based on Synaptic Decay
    ################################################################################
    
    tau_i= 10*ms; tau_e= 5*ms # follows Destexhe 2009. 
    
    #----------------------------
    #Reference synaptic weights
    #----------------------------
    
    
    Gee_r=10*nS  #(RS->RS)
    Gei_r=10*nS  #(RS->FS) 
    
    Gii_r=65*nS #(FS->FS) #67nS for Destexhe (for small scale simulations << 10000 neurons)
    Gie_r=65*nS #(FS->RS)
    
    
    #-----------------------------
    #This allows to study the effect of the time scales alone
    
    tauI_r= 10.*ms; tauE_r= 5.*ms #References time scales
    
    Gee=Gee_r*tauE_r/tau_e
    Gei=Gei_r*tauE_r/tau_e
    
    Gii=Gii_r*tauI_r/tau_i 
    Gie=Gie_r*tauI_r/tau_i 
    
    
    ################################################################################
    #Neuron Model 
    ################################################################################
    
    #######Parameters#######
    
    V_reset=-60.*mvolt; VT=-50.*mV
    Ei= -80.*mvolt; Ee=0.*mvolt; t_ref=5*ms
    C = 200 * pF; gL = 10 * nS  # C = 1 microF gL = 0.05 mS when S (membrane area) = 20,000 um^2
    
    tauw=600*ms #600 for Destexhe 2009. 500~600
    taun = 20*ms # ~C/gL, time constant of the equation.
    
    Delay= 1.5*ms
    
    #######Eleaky Heterogenities#######
    
    Eleaky_RS=np.full(NE,-60)*mV
    Eleaky_FS=np.full(NI,-60)*mV
    
    ####### noise params ########
    sigma = params['noise']*nA*ms 
    print('used noise level : ' + str(params['noise']))

    ########Equation#########
    
    eqs= """
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) + ge*(Ee-v)+ gi*(Ei-v) - w + sigma*sqrt(2/taun)*xi)/C : volt (unless refractory)
    IsynE=ge*(Ee-v) : amp
    IsynI=gi*(Ei-v) : amp
    dge/dt = -ge/tau_e : siemens
    dgi/dt = -gi/tau_i : siemens
    dw/dt = (a*(v - EL) - w)/tauw : amp
    taum= C/gL : second
   
    a : siemens
    b : amp
    DeltaT: volt
    Vcut: volt
    EL : volt
    """
    
    
    ###### Initialize neuron group#############
    
    #FS
    neuronsI = NeuronGroup(NI, eqs, threshold='v>Vcut',reset="v=V_reset; w+=b", refractory=t_ref)
    neuronsI.a=0.*nS; neuronsI.b=0.*pA; neuronsI.DeltaT = 2.5*mV; neuronsI.Vcut = VT 
    neuronsI.EL=Eleaky_FS
    
    #RS
    neuronsE = NeuronGroup(NE, eqs, threshold='v>Vcut',reset="v=V_reset; w+=b", refractory=t_ref)
    neuronsE.a=params['a']*nS; neuronsE.b=params['b']*pA; neuronsE.DeltaT = 2.5*mV; neuronsE.Vcut = VT  #a = 1ns, b = 0.04nA for Destexhe 2009. # b = [40, 0] for grid search
    neuronsE.EL=Eleaky_RS
    
    ############################################################################################
    #Initial conditions
    ############################################################################################
    
    #Random Membrane Potentials
    neuronsI.v=np.random.uniform(low=-60,high=-50,size=NI)*mV
    neuronsE.v=np.random.uniform(low=-60,high=-50,size=NE)*mV
    
    #Conductances
    neuronsI.gi = 0.*nS;       neuronsI.ge = 0.*nS
    neuronsE.gi = 0.*nS;       neuronsE.ge = 0.*nS
    
    #Adaptation Current
    neuronsI.w = 0.*amp; neuronsE.w = 0.*amp
    
    
   

    
    ##########################################################################################
    #Synaptic Connections
    ############################################################################################
    
    #===========================================
    #FS-RS Network (AI Network)
    #===========================================
    
    # use existing connection in the base simulation.

    

    con_ee = Synapses(neuronsE, neuronsE, on_pre='ge_post += Gee')
    con_ee.connect(i=base_sim['conn_ee'][0,:], j=base_sim['conn_ee'][1,:])
    con_ee.delay=base_sim['delay_ee']*ms
    
    con_ii = Synapses(neuronsI, neuronsI, on_pre='gi_post += Gii')
    con_ii.connect(i=base_sim['conn_ii'][0,:]-NE, j=base_sim['conn_ii'][1,:]-NE)
    con_ii.delay= base_sim['delay_ii']*ms
       
      
    con_ie = Synapses(neuronsI, neuronsE, on_pre='gi_post += Gie')
    con_ie.connect(i=base_sim['conn_ie'][0,:]-NE, j=base_sim['conn_ie'][1,:])
    con_ie.delay=base_sim['delay_ie']*ms 
    
    con_ei = Synapses(neuronsE, neuronsI, on_pre='ge_post += Gei')
    con_ei.connect(i=base_sim['conn_ei'][0,:], j=base_sim['conn_ei'][1,:]-NE)
    con_ei.delay=base_sim['delay_ei']*ms
    

    
    
    ##########################################################################################
    #initial excitation
    ############################################################################################
    
    stimulus = TimedArray(np.array([500,0])*Hz, dt=100.*ms)
    P = PoissonGroup(1, rates='stimulus(t)')
    
    con_pe = Synapses(P, neuronsE, on_pre='ge_post += Gee', delay=Delay)
    con_pe.connect(p=1)
    
    spikemonP = SpikeMonitor(P,variables='t')
    
    con_pi = Synapses(P, neuronsI, on_pre='ge_post += Gei', delay=Delay)
    con_pi.connect(p=1)
    
    
    ########################################################################################
    # Simulation
    ########################################################################################
    
    #Recording informations from the groups of neurons
    
    #FS
    # statemonI = StateMonitor(neuronsI, ['v'], record=[0])
    spikemonI = SpikeMonitor(neuronsI, variables='t') 
    
    
    #RS
    # statemonE = StateMonitor(neuronsE, ['v'], record=[0])
    spikemonE = SpikeMonitor(neuronsE, variables='t') 
    
    starttime = timeit.default_timer()
    # print("The start time is :",starttime)
    run(t_simulation) 
    comp_time = timeit.default_timer() - starttime
    print("The time difference is :", comp_time)
    
    print(np.array(spikemonE.t/ms))
    print(np.array(spikemonP.t/ms))
    
    # plt.plot(np.array(statemonE.v).ravel()[:1500])
    
    ####################################################################################################
    #save simulation results
    ####################################################################################################
    
    
    result = dict()
    
    result['comp_time']= comp_time
    
    NeuronIDE=np.array(spikemonE.i)
    NeuronIDI=np.array(spikemonI.i)
    
    timeE=np.array(spikemonE.t/ms) #time in ms
    timeI=np.array(spikemonI.t/ms)

    result['ex_idx']= NeuronIDE
    result['in_idx']= NeuronIDI + NE # for indexing
    
    result['ex_time']= timeE
    result['in_time']= timeI
    
    
    #connections
    
    connected_ee = np.array((con_ee.i, con_ee.j))
    connected_ii = np.array((con_ii.i+NE, con_ii.j+NE))
    connected_ei = np.array((con_ei.i, con_ei.j+NE))
    connected_ie = np.array((con_ie.i+NE, con_ie.j))
    
    
    result['conn_ee'] = connected_ee
    result['conn_ii'] = connected_ii
    result['conn_ei'] = connected_ei
    result['conn_ie'] = connected_ie
    
    result['params'] = params
    
    
    
    # for noise sims. 
    save_file_path = save_fol_name + '/instances/noise_'+ str(params['noise']) + '/' + str(instance)
    save_stem = save_fol_name + '/instances/noise_'+ str(params['noise'])

    if(not(os.path.exists(save_stem))):
        os.makedirs(save_stem)
    
    result['save_name']= save_name +  '/instances/noise_'+ str(params['noise'])
    result['save_file_path']= save_file_path
    
    np.save(save_file_path, result)

    # recursive call if the activity didn't generate lasting act. 
    
    verdict = util.pass_check(save_file_path + '.npy')
    if(verdict is None):
        print('recursive call!')
        run_sim(params) # recursive call
    
    print('simulation successfullly ran for ' + save_file_path)    
    
    return result




   

if __name__ == '__main__':
    result = run_sim(params)    
    print('simulation successfully finished for ' + result['save_name'])


