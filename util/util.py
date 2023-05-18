#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:14:39 2022

@author: th
"""


import numpy as np
import sys
from elephant.statistics import isi, cv, time_histogram
import quantities as pq
import matplotlib.pyplot as plt
from itertools import combinations, permutations
from sklearn import metrics
import pandas as pd
import neo
from elephant.conversion import BinnedSpikeTrain
import re



# class for handling spike trains (facilitating conversion to different formats)
class Spiketrain:
    def __init__(self, spktimes, spktemps):
        self.spktimes = spktimes
        self.spktemps = spktemps
        
    def set_trains(self):
        uniq_ids=np.unique(self.spktemps)
      
        spktime_list=[]
        for ids in uniq_ids:
            loc=np.where(self.spktemps==ids)[0]
            spktime_list.append(self.spktimes[loc])
        
        self.spktime_list = spktime_list
        self.uniq_ids = uniq_ids.astype(int)
        
        return spktime_list, uniq_ids
        
    def to_neotrain(self):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            if(len(spktrain)==1):
                neo_spk_train = neo.SpikeTrain(spktrain*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            else:
                neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            spk_mean_rate.append(neo_spk_train)
            
        self.neotrain = spk_mean_rate    
            
        return spk_mean_rate
    
    def to_countmat(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            if(len(spktrain)==1):
                neo_spk_train = neo.SpikeTrain(spktrain*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            else:
                neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            histogram_count = time_histogram([neo_spk_train], binsize*pq.ms, output='counts')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
           
        spk_mean_rate=np.asarray(spk_mean_rate)
        
        self.countmat = spk_mean_rate
    
        return spk_mean_rate
    
        
    def to_ratemat(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        
        spk_mean_rate=[]
        for spktrain in self.spktime_list:
            neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            histogram_count = time_histogram([neo_spk_train], binsize*pq.ms, output='rate')
            spk_mean_rate.append(np.squeeze(np.asarray(histogram_count)))
           
        spk_mean_rate=np.asarray(spk_mean_rate)
        
        self.countmat = spk_mean_rate
    
        return spk_mean_rate
    
    
    
    def to_binned(self, binsize):
        st_time_ms = np.floor(np.min(self.spktimes))
        ed_time_ms= np.ceil(np.max(self.spktimes))+1
        
        spk_vec=[]
        for spktrain in self.spktime_list:
            neo_spk_train = neo.SpikeTrain(np.squeeze(spktrain)*pq.ms, units=pq.ms, t_start=st_time_ms, t_stop=ed_time_ms)
            spk_vec.append(neo_spk_train)
            
        
        bst = BinnedSpikeTrain(spk_vec, bin_size=binsize * pq.ms)
        self.binned_mat = bst
        
        return bst
    
    
    def compute_corr(self, binsize):
        # compute pearson correlation for all data
        
        corr_mat = np.corrcoef(self.to_countmat(binsize))
        self.corr_mat = corr_mat
        
        return corr_mat


#######################################################################

# util functions below.


# gets neo format spike trains from the simulation output.
def extract_spktrain(sim_path):
    
    parsed = sim_path.split('/') # here we assume file/folder structure set by the simulation script.
    pattern = r'\d+_\d+\.\d+_\d+\.\d+_\d+\.\d+' 
    for ii in range(len(parsed)):   
        matched= re.fullmatch(pattern, parsed[ii])
        if matched:
            break
    N_size = parsed[ii].split('_')[0]
    
    uniq_tmps = np.arange(int(N_size))
    example_data = np.load(sim_path, allow_pickle=True).item()
    result_file = example_data
    
    
    spktimes = dict()
    spktimes['ex']= result_file['ex_time'] # in ms.
    spktimes['in'] = result_file['in_time']
    
    spktimes['ex_tmp']= result_file['ex_idx']
    spktimes['in_tmp']= result_file['in_idx']
    
    
    # concatenate
    
    concat_spk = np.concatenate((spktimes['ex'], spktimes['in']))
    concat_tmp = np.concatenate((spktimes['ex_tmp'], spktimes['in_tmp']))
    sort_idx = np.argsort(concat_spk)
    
    concat_spk = concat_spk[sort_idx]
    concat_tmp = concat_tmp[sort_idx]
    
    
    # take out first  (to clear external drives (poisson))
    rmv = 1000
    kick_idx = np.where(concat_spk<rmv)[0]
    
    concat_spk = np.delete(concat_spk,kick_idx)
    concat_tmp = np.delete(concat_tmp,kick_idx)
    
        
    # subset those belonging to uniq_tmp (not used anymore)
    # bool_idx = np.isin(concat_tmp, uniq_tmps)
    # concat_spk = concat_spk[bool_idx] - rmv
    # concat_tmp = concat_tmp[bool_idx]

    concat_spk = concat_spk - rmv
    
        
    na_train = Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    
    
    return na_train



# check if the simulation is lasting (spikes are observed for the entire simulation window)
def pass_check(result):
    verdict = dict()
    
    path = result
    
    dt = 0.1 #ms
    
    
    result_file = np.load(path, allow_pickle=True).item()
    
    spktimes = dict()
    spktimes['ex']= result_file['ex_time'] # in ms.
    spktimes['in'] = result_file['in_time']
    
    spktimes['ex_tmp']= result_file['ex_idx']
    spktimes['in_tmp']= result_file['in_idx']
    
    
    # concatenate
    
    concat_spk = np.concatenate((spktimes['ex'], spktimes['in']))
    concat_tmp = np.concatenate((spktimes['ex_tmp'], spktimes['in_tmp']))
    sort_idx = np.argsort(concat_spk)
    
    concat_spk = concat_spk[sort_idx]
    concat_tmp = concat_tmp[sort_idx]
    
    
    # take out first  (to clear external drives)
    rmv = 1000 # 1 sec.
    kick_idx = np.where(concat_spk<rmv)[0]
    
    concat_spk = np.delete(concat_spk,kick_idx)
    concat_tmp = np.delete(concat_tmp,kick_idx)
    
    
    sim_t = result_file['params']['sim_time']
    if(len(concat_spk)==0 or np.max(concat_spk)<(sim_t-1)*1000):
        print(path + ' didn''t generate lasting activity')
        return # returns None
    
    
    # use neural assmebly
    
    na_train = Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    neo_train = na_train.to_neotrain()
    
    # check firing rates 
    fr_vec = []
    for entry in neo_train:
        
        fr_vec.append(len(entry)/(sim_t-(rmv/1000)))
        
    
    #uniq_tmps
    uniq_tmps = np.unique(concat_tmp)
    
    # the below checks can be performed first to get CV, mean correlation etc...

    kick_idx = uniq_tmps[np.where(np.array(fr_vec)<=0.01)[0]] # apply check with a fixed FR
    kick_bool_idx = ~np.isin(concat_tmp, kick_idx)
    
    concat_spk = concat_spk[kick_bool_idx]
    concat_tmp = concat_tmp[kick_bool_idx]
        
    
    na_train = Spiketrain(concat_spk, concat_tmp)
    na_train.set_trains()
    neo_train = na_train.to_neotrain()
    
    fr_vec = []
    for entry in neo_train:
        sim_t = result_file['params']['sim_time']
        fr_vec.append(len(entry)/(sim_t-(rmv/1000)))
        
    params = result_file['params']
    save_name = '_'.join([str(params['N']), str(params['a']),str(params['b']), str(params['sim_time'])])

    # final checking of valid indices
    uniq_tmps = np.unique(concat_tmp)
    
    # quantify CV_ISIs and correlation 
    CV=[]
    for train in neo_train:
        CV.append(cv(isi(train)))
    
    # destexhe2009, CV_mean >1 for irregular spikes
    verdict['CV_mean'] = np.mean(CV)
    verdict['CV_std'] = np.std(CV)
    
    # corr mean <0.1 for Asynchronous trains
    corr_mat = na_train.compute_corr(5) # bin size of 5 ms.
    
    verdict['corr_mean']= np.nanmean(corr_mat[np.triu_indices(corr_mat.shape[0], k=1)]) # correlation can give nans if the activity of a neuron is dormant.
    verdict['corr_std'] = np.nanstd(corr_mat[np.triu_indices(corr_mat.shape[0], k=1)])
    
    verdict['mean_fr']= np.mean(fr_vec)
    verdict['valid_idx']=uniq_tmps # currently this just collects all existing neurons. 
        
    
    # label this network act. for its synchronicity
    if( np.logical_and(verdict['CV_mean']>1 , verdict['corr_mean']<0.1)):
        verdict['label'] = 'async'
        
    else:
        verdict['label']='sync'
        
    verdict['save_name']=save_name
        
    
    return verdict


   

def make_marked_edges(samp_conn):
    all_conn = samp_conn['all']
    all_idx= np.unique(all_conn)
    
    pairs = np.array(list(permutations(all_idx, 2)))
    
    marked_edges = np.zeros((pairs.shape[0],pairs.shape[1]+1))
    
    marked_edges[:,0]= pairs[:,0]
    marked_edges[:,1]= pairs[:,1]
    
    # for ii in range(all_conn.shape[1]):  # this can be very slow for large data / so changed it.
    first = np.searchsorted(pairs[:,0], all_conn[0,:])
    first_uniq = np.unique(first)    
    for ii, f_un in enumerate(first_uniq):
        first_chunk = pairs[f_un:f_un+len(all_idx)-1,:]
        second_chunk = np.where(all_conn[0,:]==all_idx[ii])[0]
        second_chunk = all_conn[1, second_chunk]
        second = np.searchsorted(first_chunk[:,1],second_chunk)
        idx = second + f_un
        
        # idx = np.where((pairs == all_conn[:,ii]).all(axis=1))[0]
        marked_edges[idx,2]=1
    
    
    return marked_edges



def eval_performance(result_mat):
    
    metrics_dict = {}
    metrics_dict['compute_time']= result_mat['compute_time']
    true_con_mat = result_mat['true_con']
    score_con_mat = result_mat['score_con']
    
    gt_edge_idx = np.where(np.logical_not(np.isnan(true_con_mat)))
    y_true = np.zeros(len(gt_edge_idx[0]))
    y_true[np.nonzero(true_con_mat[gt_edge_idx[0], gt_edge_idx[1]])] = 1
    y_score = score_con_mat[gt_edge_idx[0], gt_edge_idx[1]]
    metrics_dict['fpr'], metrics_dict['tpr'], metrics_dict['thresholds'] = metrics.roc_curve(y_true, y_score)
    metrics_dict['prc_precision'], metrics_dict['prc_recall'], metrics_dict['prc_thresholds'] = metrics.precision_recall_curve(y_true, y_score)
    metrics_dict['auc'] = metrics.roc_auc_score(y_true, y_score)
    metrics_dict['aps'] = metrics.average_precision_score(y_true, y_score)

    metrics_df = pd.DataFrame([metrics_dict.values()], index=[0], columns=metrics_dict.keys())
    return metrics_df


def create_connectivity_matrix(nodes, marked_edges):
    pairs = np.array(list(combinations(nodes, 2)))
    pairs = np.vstack([pairs, pairs[:, ::-1]])
    con_matrix = np.empty((len(nodes), len(nodes)))
    con_matrix[:,:] = np.nan
    edges_to_consider = np.where(np.logical_and(np.isin(marked_edges[:,0], nodes),
                                                      np.isin(marked_edges[:,1], nodes)))[0]
    idx1 = np.searchsorted(nodes, marked_edges[edges_to_consider,0])
    idx2 = np.searchsorted(nodes, marked_edges[edges_to_consider,1])
    con_matrix[idx1, idx2] = marked_edges[edges_to_consider,2]

    return con_matrix




def plot_raster(neo_train, st_time, ed_time, s=2, shift_time=False, ax=None):
    
    if(ax is None):
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        print('using handed over axis')
    for jj, train in enumerate(neo_train):
        
        bool_idx = np.logical_and(st_time<= train/1000, ed_time> train/1000)
        valid_spk = train[bool_idx]/1000
        valid_spk = np.array(valid_spk)
        if(shift_time):
            valid_spk = valid_spk - st_time
        
        ax.scatter(valid_spk,np.ones(len(valid_spk))*(jj+1), color='gray', alpha= 0.5, s = s)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron id')
    
    
    
def subset_time(sampled, t_compute):
    
    val_idx = np.where(sampled['spktrains_sec'] <= t_compute)[0]
    sampled['ids'] = sampled['ids'][val_idx]
    sampled['spktrains'] = sampled['spktrains'][val_idx]
    sampled['spktrains_sec'] = sampled['spktrains_sec'][val_idx]
    
    return sampled
        
def get_spktrains(sim_path):
    
    na_train = extract_spktrain(sim_path)
    subset_spk_ms = np.copy(na_train.spktimes)
    # subset_spk = na_train.spktimes/1000 # converting it to seconds

    sampled={}
    sampled['ids'] = na_train.spktemps
    sampled['spktrains'] = subset_spk_ms # in ms.
    # sampled['spktrains_sec'] = subset_spk
    
    return sampled

def get_gt_conn(sim_data):
    
    gt_conn =dict()
    keys = ['conn_ee', 'conn_ei', 'conn_ii', 'conn_ie']
    for key in keys:
        gt_conn[key]=sim_data[key]


    samp_conn=dict()
    keys = list(gt_conn.keys())
    
    sampled_idx = np.arange(sim_data['params']['N']) # change this line if you are subetting neurons, as is this doesn't do anything
    for key in keys:
        entry = gt_conn[key]
        pre_check = np.isin(entry[0,:], sampled_idx)
        post_check = np.isin(entry[1,:], sampled_idx)
        
        check = np.logical_and(pre_check, post_check)
        
        samp_conn[key]=gt_conn[key][:,check]
        
    samp_conn['all']= np.concatenate(list(samp_conn.values()),1)    
    


    return samp_conn
    

