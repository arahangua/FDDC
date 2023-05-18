#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 16:41:17 2022

@author: th
"""
import sys

sys.path.append('../../util')
import util
import numpy as np
import timeit
import scipy 

# inference using graphical lasso methods, "gglasso", Schaipp et al. 2021, https://doi.org/10.21105/joss.03865, / "regain", Tomasi et al. 2018 http://doi.acm.org/10.1145/3219819.3220121

from gglasso.problem import glasso_problem
from gglasso.solver.admm_solver import ADMM_MGL
from gglasso.helper.experiment_helper import lambda_grid, discovery_rate, error
from gglasso.helper.utils import get_K_identity
from gglasso.helper.model_selection import aic, ebic


from elephant import kernels
from elephant.statistics import instantaneous_rate
from elephant.kernels import GaussianKernel, AlphaKernel

import quantities as pq
import pylab as plt
from fracdiff.sklearn import Fracdiff, FracdiffStat


def run_inference(inf_dict):
    
    if(inf_dict['method']=='nd'):
        result= network_deconv(inf_dict['spktrains'], inf_dict['ids'], inf_dict['gt_conn'], inf_dict)
    elif(inf_dict['method']=='glasso'):
        result= glasso(inf_dict['spktrains'], inf_dict['ids'], inf_dict['gt_conn'], inf_dict)
    elif(inf_dict['method']=='dcov'):
        result= DCOV(inf_dict['spktrains'], inf_dict['ids'], inf_dict['gt_conn'], inf_dict)
    elif('ddc' in inf_dict['method']):
        result= DDC(inf_dict['spktrains'], inf_dict['ids'], inf_dict['gt_conn'], inf_dict)   

    print('inference successfully finished')
    return result    




def glasso(subset_spk_ms, subset_ids, samp_gt_conn, method_params, return_prec= False):
    
    # sampled neural assembly 
    na_train = util.Spiketrain(subset_spk_ms, subset_ids)
    na_train.set_trains()
    bin_size = method_params['covar_bin']
    
    # get cov mat
    countmat= na_train.to_countmat(bin_size)
    N = countmat.shape[1]

    #start measuring the wall clock, includes computing convariance matrices.
    st_time = timeit.default_timer()

    S = compute_covmat(countmat) #SGL case
    
    print("Shape of empirical covariance matrix: ", S.shape)
    print("Length of the sampled time series: ", N)
    
    
    # Static graphical lasso 
    
    P = glasso_problem(S, N, reg_params = {'lambda1': 0.05}, latent = False, do_scaling = False) #SGL case
    print(P)
    
    lambda1_range = np.logspace(0.5,-1.5,8)
    modelselect_params = {'lambda1_range': lambda1_range}
    
    P.model_selection(modelselect_params = modelselect_params, method = 'eBIC', gamma = method_params['gamma'])
    
    # regularization parameters are set to the best ones found during model selection
    print(P.reg_params)
    
           
    elapsed_time = timeit.default_timer() - st_time
    sol = P.solution.precision_
    
    # making gt tables
    nodes= np.unique(subset_ids)
    marked_edges= util.make_marked_edges(samp_gt_conn)
    
    
    true_con_mat = util.create_connectivity_matrix(nodes, marked_edges)
    sol_zero_diag = np.copy(sol)
   
    
    result_mat = {}
    result_mat['true_con'] = true_con_mat
    result_mat['thres_con'] = true_con_mat # for now no interpretation, no thresholding mechanism was implemented.
    result_mat['score_con'] = np.abs(sol_zero_diag) 
    result_mat['compute_time'] = elapsed_time
    result_mat['sol_mat'] = sol
    result_mat['method_params']=method_params
    
    return result_mat



def network_deconv(subset_spk_ms, subset_ids, samp_gt_conn, method_params):
    na_train = util.Spiketrain(subset_spk_ms, subset_ids)
    na_train.set_trains()
    
    bin_size = method_params['covar_bin']
    
    # get cov mat
    countmat= na_train.to_countmat(bin_size)
    
    st_time = timeit.default_timer()
    S = compute_covmat(countmat) 
    #start measuring the wall clock, includes computing convariance matrices.
    dir_mat = compute_net_deconv(S, beta=method_params['beta'])
    
    elapsed_time = timeit.default_timer() - st_time
    sol = dir_mat
    
    # making gt tables
    nodes= np.unique(subset_ids)
    marked_edges= util.make_marked_edges(samp_gt_conn)
    
    
    true_con_mat = util.create_connectivity_matrix(nodes, marked_edges)
    # true_con_mat = true_con_mat + true_con_mat.T, # no need to concern about the directionality given how auc is computed.
    sol_zero_diag = np.copy(sol)
    

    
    result_mat = {}
    result_mat['true_con'] = true_con_mat
    result_mat['thres_con'] = true_con_mat # for now no interpretation, no thresholding mechanism was implemented.
    result_mat['score_con'] = np.abs(sol_zero_diag) 
    result_mat['compute_time'] = elapsed_time
    result_mat['sol_mat'] = sol
    result_mat['method_params']=method_params
    
    return result_mat
    
    
def compute_net_deconv(covmat, beta=0.99, alpha=1): # python implementation of original matlab script (Feizi et al. 2013)
    
    n = covmat.shape[0]
    covmat = np.multiply(covmat,(1-np.eye(n)))
    
    # thresholding the input matrix
    
    y=np.quantile(covmat,1-alpha,axis=None)
    mat_th=np.multiply(covmat,(covmat>=y))
    
    # making the matrix symetric if already not
    mat_th = (mat_th+mat_th.T)/2
    
     # eigen decomposition
    print('decomposition and deconvolution...')
    [D,U] = np.linalg.eig(mat_th)
    
    
    lam_n=abs(min([min(D),0]))
    lam_p=abs(max([max(D),0]))
    
    m1=lam_p*(1-beta)/beta
    m2=lam_n*(1+beta)/beta
    m=max(m1,m2)
    
    
    #network deconvolution
    for i in range(D.shape[0]):
        D[i] = (D[i])/(m+D[i])
    mat_new1 = np.dot(U, np.dot(np.diag(D),np.linalg.inv(U)))
    
    # displying direct weights
    ind_edges = (mat_th>0)*1.0
    ind_nonedges = (mat_th==0)*1.0
    m1 = np.max(np.multiply(covmat,ind_nonedges))
    m2 = np.min(mat_new1)
    mat_new2 = np.multiply((mat_new1+max(m1-m2,0)), ind_edges)+ np.multiply(covmat,ind_nonedges)
    
    
    # linearly mapping the deconvolved matrix to be between 0 and 1
    
    m1 = np.min(mat_new2)
    m2 = np.max(mat_new2)
    mat_nd = np.divide((mat_new2-m1),(m2-m1))
    
    return mat_nd
   
   
def DCOV(subset_spk_ms, subset_ids, samp_gt_conn, method_params):
    na_train = util.Spiketrain(subset_spk_ms, subset_ids)
    na_train.set_trains()
    bin_size = method_params['covar_bin']
    
    
    
    if('kernel' in method_params.keys()):
                
        # get neotrains
        neo_t = na_train.to_neotrain()
        
        if(method_params['kernel']=='alpha'):
        # get smoothed data
            countmat = instantaneous_rate(neo_t, sampling_period=bin_size*pq.ms, kernel=AlphaKernel(1*pq.ms))
            countmat = np.array(countmat).T.astype(np.float16)
            print(countmat.shape)

        elif(method_params['kernel']=='gauss'):
            countmat = instantaneous_rate(neo_t, sampling_period=bin_size*pq.ms, kernel=GaussianKernel(1*pq.ms))
            countmat = np.array(countmat).T.astype(np.float16)
            print(countmat.shape)
        else:
            countmat= na_train.to_countmat(bin_size)
    else:
        countmat= na_train.to_countmat(bin_size)
    
    
    st_time = timeit.default_timer()
    res = compute_dCov(countmat, method_params, return_extra=False)
    
    elapsed_time = timeit.default_timer() - st_time
    
    # making gt tables
    nodes= np.unique(subset_ids)
    marked_edges= util.make_marked_edges(samp_gt_conn)
    
    
    true_con_mat = util.create_connectivity_matrix(nodes, marked_edges)
    
    sol = res['dCov']
    
    sol_zero_diag = np.copy(sol)
    result_mat = {}
    result_mat['true_con'] = true_con_mat
    result_mat['thres_con'] = true_con_mat # for now no interpretation, no thresholding mechanism was implemented.
    result_mat['score_con'] = np.abs(sol_zero_diag) 
    
    result_mat['compute_time'] = elapsed_time
    result_mat['sol_mat'] = sol
    result_mat['res'] = res
    result_mat['method_params']=method_params
    
    return result_mat


def kernel_handler(na_train, neo_t, method_params):
    bin_size = method_params['covar_bin']
    if('kernel' in method_params.keys()):
        if(method_params['kernel']=='alpha'):
        # get smoothed data
            countmat = instantaneous_rate(neo_t, sampling_period=bin_size*pq.ms, kernel=AlphaKernel(method_params['kernel_size']*pq.ms))
            countmat = np.array(countmat).T.astype(np.float16)
            print(countmat.shape)

        elif(method_params['kernel']=='gauss'):
            countmat = instantaneous_rate(neo_t, sampling_period=bin_size*pq.ms, kernel=GaussianKernel(method_params['kernel_size']*pq.ms))
            countmat = np.array(countmat).T.astype(np.float16)
            print(countmat.shape)
        else:
            countmat= na_train.to_countmat(bin_size)
    else:
        countmat= na_train.to_countmat(bin_size)
    return countmat

    
def DDC(subset_spk_ms, subset_ids, samp_gt_conn, method_params):
    na_train = util.Spiketrain(subset_spk_ms, subset_ids)
    na_train.set_trains()
    # get neotrains
    neo_t = na_train.to_neotrain()
    
    st_time_w_kernel = timeit.default_timer() # measures time including kernel convolution 
    if('frac' in method_params['method']):
        countmat = kernel_handler(na_train, neo_t, method_params)     
        st_time = timeit.default_timer() # measures time without kernel convolution
        res = compute_frac_DDC(countmat, method_params)
        
    elif(method_params['method']=='precision'):
        countmat = kernel_handler(na_train, neo_t, method_params)
        st_time = timeit.default_timer()
        res = compute_only_prec(countmat, method_params)
    else:
        countmat = kernel_handler(na_train, neo_t, method_params)
        st_time = timeit.default_timer()
        res = compute_DDC(countmat, method_params)
    
    elapsed_time = timeit.default_timer() - st_time
    elapsed_time_w_kernel = timeit.default_timer() - st_time_w_kernel
    
    # making gt tables
    nodes= np.unique(subset_ids)
    marked_edges= util.make_marked_edges(samp_gt_conn)
    
    
    true_con_mat = util.create_connectivity_matrix(nodes, marked_edges)
    
    sol = res['DDC']
    
    sol_zero_diag = np.copy(sol)
    result_mat = {}
    result_mat['true_con'] = true_con_mat
    result_mat['thres_con'] = true_con_mat # for now no interpretation, no thresholding mechanism was implemented.
    result_mat['score_con'] = np.abs(sol_zero_diag) 
    result_mat['compute_time'] = elapsed_time
    result_mat['compute_time_kernel'] = elapsed_time_w_kernel
    result_mat['sol_mat'] = sol
    result_mat['res'] = res
    result_mat['method_params']=method_params
    
    return result_mat
    
def compute_only_prec(countmat, method_params, thres=0, TR=1, return_extra=False):
    
    [N,T] = countmat.shape
    Cov  = np.cov(countmat)
    precision = np.linalg.inv(Cov)
    
    result={}
    result['precision']=precision
    result['S']=Cov
    return result
    
def compute_dCov(countmat, method_params, thres=0, TR=1, return_extra=False):
    
    [N,T] = countmat.shape
    Fx = countmat # non-linearity not used
    tmp = np.cov(Fx, countmat)
    B = tmp[:N, N:]
    
    #dcov
    dV = ((-1/2) * countmat[:, :-2] + (1/2)*countmat[:, 2:])/TR
    dV = np.hstack((np.mean(countmat,axis=1).reshape(countmat.shape[0],1), dV))    
    dV = np.hstack((dV, np.mean(countmat,axis=1).reshape(countmat.shape[0],1)))    
    tmp = np.cov(dV, countmat)
    dCov = tmp[:N, N:]

    result={}
    result['dCov']=dCov
    
    return result

def compute_DDC(countmat, method_params, thres=0, TR=1, return_extra=False):
    
    [N,T] = countmat.shape
    Fx = countmat # non-linearity not used
    
    tmp = np.cov(Fx, countmat)
    B = tmp[:N, N:]
    
    #dcov
    dV = ((-1/2) * countmat[:, :-2] + (1/2)*countmat[:, 2:])/TR
    dV = np.hstack((np.mean(countmat,axis=1).reshape(countmat.shape[0],1), dV))    
    dV = np.hstack((dV, np.mean(countmat,axis=1).reshape(countmat.shape[0],1)))    

    tmp = np.cov(dV, countmat)
    dCov = tmp[:N, N:]
    
    
    DDC = np.matmul(dCov, np.linalg.inv(B))

    result={}
    result['DDC']=DDC
    result['R(x)']=B
    result['dCov']=dCov
    
    return result

    
def compute_frac_DDC(countmat, method_params, thres=0, TR=1, return_extra=False):
    
    [N,T] = countmat.shape
    Fx = countmat
    
    tmp = np.cov(Fx, countmat)
    B = tmp[:N, N:]
    
    if(method_params['frac_order']<=1):
        print('computing fractional diff with dimension of ' + str(method_params['frac_order']))
        max_d = method_params['frac_order']
        f = Fracdiff(method_params['frac_order'], window=method_params['diff_window'])
        dV = f.fit_transform(countmat.T) # (sample x features)
        dV = dV.T
        
    else: # when fraction order is not explicitly given, we sample spike trains and perform adfuller test to find lowest order (starts from 0.01)
        
        idx = np.random.choice(np.arange(countmat.shape[0]), method_params['sample_n'], replace=False)
        rand_stp = countmat.shape[1] - method_params['sample_window']-1
        rand_stp = int(np.floor(rand_stp*np.random.rand(1)))
        
        
        sampled = countmat[idx, rand_stp:rand_stp+ method_params['sample_window']]
        # sampled = countmat[idx, :]
        
        print(sampled.shape)
        f = FracdiffStat(lower = 0.01, window = method_params['diff_window']).fit(sampled.T)
        
        print(f.d_)
        max_d = np.nanmax(f.d_)
        print(max_d)
        f = Fracdiff(max_d, window=method_params['diff_window'])
        dV = f.fit_transform(countmat.T) # (sample x features)
        dV = dV.T
    
        
    tmp = np.cov(dV, countmat)
    dCov = tmp[:N, N:]
    
    
    DDC = np.matmul(dCov, np.linalg.inv(B))

    result={}
    result['DDC']=DDC
    result['alpha']=max_d # fractional order 'beta' in the manuscript
    return result

          
   
   
    
def compute_covmat(count_mat, if_prec=False):
    
    cov_mat = np.cov(count_mat)
    if(if_prec):
        precision_mat = np.linalg.inv(cov_mat)      
        return cov_mat, precision_mat
    else:
        return cov_mat

def compute_corrmat(count_mat, if_prec=False):
    z_mat = scipy.stats.zscore(count_mat, axis=1)
    
    cov_mat = np.cov(z_mat)
    if(if_prec):
        precision_mat = np.linalg.inv(cov_mat)      
        return cov_mat, precision_mat
    else:
        return cov_mat

    