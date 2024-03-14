from sklearn.manifold import Isomap
from sklearn.decomposition import FastICA
import statsmodels.api as sm
import sys
from typing import Tuple, Optional, Dict, Callable, Union

# JAX SETTINGS
import jax
import jax.numpy as np
import jax.random as random
from jax.ops import index, index_add, index_update
from jax.scipy.special import ndtri
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI
from jax.scipy import linalg
from jax.scipy.stats import norm
from jax import grad, value_and_grad
from jax.experimental import optimizers as jax_opt
from funcs_LNC import *

# NUMPY SETTINGS
import numpy as onp
import time
onp.set_printoptions(precision=3, suppress=True)

# LOGGING SETTINGS
import logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
logger = logging.getLogger()
#logger.setLevel(logging.INFO)

@jax.jit
def rrModel(lam, X, y, ws):
    
    L = linalg.cho_factor(np.dot(X.T,X) + lam * np.diag(1 / ws))

    # weights
    weights = linalg.cho_solve(L, np.dot(X.T,y))

    # save the params

    y_hat = np.dot(X, weights)

    resids = y - y_hat

    # return the predictions
    return weights, resids, y_hat

@jax.jit
def linRegCoefStat(lam, X, y, ws):

    invMat = np.linalg.inv(np.dot(X.T, X)+ lam * np.diag(1 / ws))
    #L = linalg.cho_factor(np.dot(X.T,X) + lam * np.diag(1 / ws))

    # weights
    #weights = linalg.cho_solve(L, np.dot(X.T,y))
    weights = np.dot(invMat, np.dot(X.T,y))

    # save the params

    y_hat = np.dot(X, weights)

    resids = y - y_hat
    mses = mse(y, y_hat)

    var_b = mses*(invMat).diagonal()
    sd_b = np.sqrt(var_b)
    
    ts_b = weights[:,0] / sd_b
    return ts_b


@jax.jit
def getDataLoss_LNC_lin(smpl, loss_data):
    
     
    x, y, Z, U, U_ica,  K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds, idx_min, idx_x_c, idx_y_c, mixMat = loss_data    

    D_x_aux = D_x[smpl, :]
    D_x_aux = D_x_aux[:, smpl]
    K_x_aux = K_x[smpl, :]
    K_x_aux = K_x_aux[:, smpl]
    K_y_aux = K_y[smpl,:]
    K_y_aux = K_y_aux[:,smpl]
    K_u_aux = K_u[smpl,:]

    U_aux = U[smpl,:]
    U_ica_aux = U_ica[smpl,:]
    
                
    x_aux = x[smpl,]
    y_aux = y[smpl,]
    

    if Z is not None:
        Z_aux = Z[smpl,]
    else:
        Z_aux = None
        
           
    return x_aux, y_aux, Z_aux, U_aux, U_ica_aux, K_u_aux, D_x_aux, K_y_aux, None, K_x, idxs, beta_real, stds, idx_min, idx_x_c, idx_y_c, mixMat

def getIniPar_LNC_lin(reps, loss_data, pars, smplsParts): #N, m, reps, y, M
    
    
    _, _, lam, _ , _ = pars 
    x, y, Z, U, U_ica,  K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds, idx_min, idx_x_c, idx_y_c, mixMat = loss_data
    
    n = x.shape[0]
    

    # initialize z_x to x, z_y to y and z_c manifold of z_c
    # we actually have to initialize alpha_x, alpha_y and alpha_c so in each case
    # we have to work out what z = U*alpha -> cholesky decomposition of U
    
    #zx_ini = x
    #zy_ini = y 
    xy = onp.hstack([x, y])
    
    #embedding = Isomap(n_components=10)
    #xy_transformed = embedding.fit_transform(xy)
    #zc_ini = np.array(xy_transformed[:,3])[:,None]
    
    
    zc_ini = np.array(U_ica[:,idx_min][:,None])
    p_ica = U_ica.shape[1]

    alpha_x_ini = onp.zeros((p_ica, p_ica))
    alpha_x_ini[idx_x_c, idx_x_c] = 1
    alpha_x_ini = alpha_x_ini[:,idx_x_c]
    alpha_x_ini = np.array(alpha_x_ini)
    
    alpha_y_ini = onp.zeros((p_ica, p_ica))
    alpha_y_ini[idx_y_c, idx_y_c] = 1
    alpha_y_ini = alpha_y_ini[:,idx_y_c]
    alpha_y_ini = np.array(alpha_y_ini)


    zx_ini = (U_ica@alpha_x_ini)
    zy_ini = (U_ica@alpha_y_ini)

    p = U.shape[1]
    L = linalg.cho_factor(np.dot(U.T, U) + lam * np.eye(p))
    #alpha_x_ini = linalg.cho_solve(L, np.dot(U.T, zx_ini))
    #alpha_y_ini = linalg.cho_solve(L, np.dot(U.T, zy_ini))
    alpha_c_ini = linalg.cho_solve(L, np.dot(U.T, zc_ini))
    
    
    params = {
            'alpha_x': alpha_x_ini,
            'alpha_y': alpha_y_ini,
            'alpha_c': alpha_c_ini,
    }
    
    #onp.random.seed(seed=4)
    
    #zx_ini = (zx_ini - np.mean(zx_ini)) / (np.std(zx_ini))
    #zy_ini = (zy_ini - np.mean(zy_ini)) / (np.std(zy_ini))
    #zc_ini = (zc_ini - np.mean(zc_ini)) / (np.std(zc_ini))
    zx_ini = np.array(onp.apply_along_axis(stdrze, 0, zx_ini))
    zy_ini = np.array(onp.apply_along_axis(stdrze, 0, zy_ini))
    zc_ini = np.array(onp.apply_along_axis(stdrze, 0, zc_ini))
    

    sigma_x_med = 1 / np.median(D_x)
    sigma_y = 1 / np.median(D_y)
    

    D_zx = covariance_matrix(sqeuclidean_distance, zx_ini, zx_ini)
    sigma_zx_med = 1 / np.median(D_zx)
    
    D_zy= covariance_matrix(sqeuclidean_distance, zy_ini, zy_ini)
    sigma_zy_med = 1 / np.median(D_zy)
    
    D_zc = covariance_matrix(sqeuclidean_distance, zc_ini, zc_ini)
    sigma_zc_med = 1 / np.median(D_zc)
    
    
    
    K_zx = rbf_kernel_matrix({'gamma': sigma_zx_med}, zx_ini, zx_ini)
    K_zy = rbf_kernel_matrix({'gamma': sigma_zy_med}, zy_ini, zy_ini)
    K_zc = rbf_kernel_matrix({'gamma': sigma_zc_med}, zc_ini, zc_ini)
    K_yhat = K_x*K_zc*K_zy
    K_xhat = K_zc*K_zx

    
    smplPart1 = smplsParts[0]
    N_part1 = smplPart1.shape[0]
    y_part1 = y[smplPart1,]
    x_part1 = x[smplPart1,]
    K_yhat_part1 = K_yhat[smplPart1, :]
    K_yhat_part1 = K_yhat_part1[:, smplPart1]
    K_xhat_part1 = K_xhat[smplPart1, :]
    K_xhat_part1 = K_xhat_part1[:, smplPart1]
    ws = np.ones(N_part1)
    
    
    weights_y, resids_y, y_hat = krrModel(lam, K_yhat_part1, y_part1, ws)
    weights_x, resids_x, x_hat = krrModel(lam, K_xhat_part1, x_part1, ws)
    
    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    distsRx = covariance_matrix(sqeuclidean_distance, resids_x, resids_x)
    sigma_rx_med = 1 / np.quantile(distsRx, 0.5)
    distsRy = covariance_matrix(sqeuclidean_distance, resids_y, resids_y)
    sigma_ry_med = 1 / np.quantile(distsRy, 0.5)
    

    
    
    sigs_zx_h= sigma_zx_med * np.ones(reps)
    sigs_zy_h= sigma_zy_med * np.ones(reps)
    sigs_zc_h= sigma_zc_med * np.ones(reps)
    sigs_x_h= sigma_x_med * np.ones(reps)
    sigs_rx_h= sigma_rx_med * np.ones(reps)
    sigs_ry_h= sigma_rx_med * np.ones(reps)
    
    
    params['ln_sig_zx_h']= np.log(sigs_zx_h)
    params['ln_sig_zy_h']= np.log(sigs_zy_h)
    params['ln_sig_zc_h']= np.log(sigs_zc_h)
    params['ln_sig_x_h']= np.log(sigs_x_h)
    params['ln_sig_rx_h']= np.log(sigs_rx_h)
    params['ln_sig_ry_h']= np.log(sigs_ry_h)
    
    
    return params

@jax.jit
def getParamsForGrad_LNC_lin(params, rep, smpl):
  
    ln_sig_x_h = params["ln_sig_x_h"][rep]
    ln_sig_zx_h = params["ln_sig_zx_h"][rep]
    ln_sig_zy_h = params["ln_sig_zy_h"][rep]
    ln_sig_zc_h = params["ln_sig_zc_h"][rep]
    ln_sig_rx_h = params["ln_sig_rx_h"][rep]
    ln_sig_ry_h = params["ln_sig_ry_h"][rep]
    

    params_aux = params.copy()

    #alpha_x = params["alpha_x"]#[:, rep]
    #alpha_y = params["alpha_y"]#[:, rep]
    #alpha_c = params["alpha_c"]#[:, rep]
    #alpha_x = alpha_x[:,None]
    #alpha_y = alpha_y[:,None]
    #alpha_c = alpha_c[:,None]
    
    
    #params_aux['alpha_x'] = alpha_x
    #params_aux['alpha_y'] = alpha_y
    #params_aux['alpha_c'] = alpha_c
    
    params_aux["ln_sig_x_h"] = ln_sig_x_h
    params_aux["ln_sig_zx_h"] = ln_sig_zx_h
    params_aux["ln_sig_zy_h"] = ln_sig_zy_h
    params_aux["ln_sig_zc_h"] = ln_sig_zc_h
    params_aux["ln_sig_rx_h"] = ln_sig_rx_h
    params_aux["ln_sig_ry_h"] = ln_sig_ry_h
    

    return params_aux

def updateParams_LNC_lin(params, grad_params, smpl, iteration, rep, learning_rate):
     
    #alpha_x
    #idx_rows = smpl[:, None]
    p = params['alpha_x'].shape[0]
    idx_rows = np.linspace(0, p - 1, p, dtype=int)[:, None]
    idx_cols = np.linspace(0, p - 1, p, dtype=int)[:, None] 
    #idx_cols = np.array(rep)[None, None]
    idx = jax.ops.index[tuple([idx_rows, idx_cols])]
    A = params['alpha_x']#[tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_x']
    #params['alpha_x'] = index_update(params['alpha_x'], idx, A - B)
    #params['alpha_x'] = A - B
    
    #alpha_y
    p = params['alpha_y'].shape[0]
    idx_rows = np.linspace(0, p - 1, p, dtype=int)[:, None]
    idx_cols = np.linspace(0, p - 1, p, dtype=int)[:, None]
    idx = jax.ops.index[tuple([idx_rows, idx_cols])]
    A = params['alpha_y']#[tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_y']
    #params['alpha_y'] = index_update(params['alpha_y'], idx, A - B)
    #params['alpha_y'] = A - B
    
    #alpha_c
    p = params['alpha_y'].shape[0]
    idx_rows = np.linspace(0, p - 1, p, dtype=int)[:, None]
    idx_cols = np.linspace(0, 1 - 1, 1, dtype=int)[:, None]
    idx = jax.ops.index[tuple([idx_rows, idx_cols])]

    A = params['alpha_c']#[tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_c']
    #params['alpha_c'] = index_update(params['alpha_c'], idx, A - B)
    params['alpha_c'] = A - B
    

    gpars = [grad_params["ln_sig_x_h"],
    grad_params["ln_sig_zx_h"],
    grad_params["ln_sig_zy_h"],
    grad_params["ln_sig_zc_h"],
    grad_params["ln_sig_rx_h"],
    grad_params["ln_sig_ry_h"],
    grad_params["ln_sig_x_h"],
    grad_params["ln_sig_zx_h"],
    grad_params["ln_sig_zy_h"],
    grad_params["ln_sig_zc_h"],
    grad_params["ln_sig_rx_h"],
    grad_params["ln_sig_ry_h"]]
    

    if (onp.sum(onp.isnan(B))!=0) | (onp.sum(onp.isinf(B))!=0) | (onp.sum(onp.isnan(gpars))!=0):
        idx_nan, _ = onp.where(onp.isnan(B))
        print("nans in grad Z, iteration: ", iteration, " rep: ", rep)
        raise ValueError('Nans in gradient.')
            
    
    
    return None

@jax.jit
def kernel_mat_LNC_lin(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h):
    
    
    K_x_h = np.exp(-sig_x_h * D_x)
    
    K_zx_h = rbf_kernel_matrix({'gamma': sig_zx_h}, zx, zx)
    K_zy_h = rbf_kernel_matrix({'gamma': sig_zy_h}, zy, zy)
    K_zc_h = rbf_kernel_matrix({'gamma': sig_zc_h}, zc, zc)
    

    # x causes - to compare to rx
    K_ax_h = K_zx_h*K_zc_h
    # y causes - to compare to ry
    K_ay_h = K_x_h*K_zy_h*K_zc_h
    
    return K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h

# loss 
@jax.jit
def loss_LNC_lin(params, pars, loss_data, ws, alpha_x, alpha_y, alpha_c):
    
    beta, neta, lam, nu, lu = pars 
    
    
    
    x, y, Z, U, U_ica,  K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds, idx_min, idx_x_c, idx_y_c, mixMat = loss_data    

    zxc = U_ica@alpha_x
    zyc = U_ica@alpha_y
    zcc = U@alpha_c
    
    alpha_x = params["alpha_x"]
    alpha_y = params["alpha_y"]
    alpha_c = params["alpha_c"]

    alphap_x = mixMat@alpha_x
    alphap_y = mixMat@alpha_y

    
    px = alpha_x.shape[1]
    py = alpha_y.shape[1]
    p = alpha_c.shape[0]
    one_p = np.ones(p)[:,None]
    one_px = np.ones(px)[:,None]
    one_py = np.ones(py)[:,None] 
    weight_orth = (one_px.T@alphap_x.T@alpha_c)/px + (one_py.T@alphap_y.T@alpha_c)/py + (one_px.T@alphap_x.T@alphap_y@one_py)/(p*p)   
    weight_orth = weight_orth[0,0]

    zx = U_ica@alpha_x
    zy = U_ica@alpha_y
    zc = U@alpha_c
    
    #zx = (zx - np.mean(zx)) / (np.std(zx))
    #zy = (zy - np.mean(zy)) / (np.std(zy))
    #zc = (zc - np.mean(zc)) / (np.std(zc))
    #zx = np.array([stdrze(zx[:,i]) for i in range(zx.shape[1])]).T
    #zy = np.array([stdrze(zy[:,i]) for i in range(zy.shape[1])]).T     

    #zxc = (zxc - np.mean(zxc)) / (np.std(zxc))
    #zyc = (zyc - np.mean(zyc)) / (np.std(zyc))
    #zcc = (zcc - np.mean(zcc)) / (np.std(zcc))
    #zxc = np.array([stdrze(zxc[:,i]) for i in range(zxc.shape[1])]).T
    #zyc = np.array([stdrze(zyc[:,i]) for i in range(zyc.shape[1])]).T

    
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_zx_h = np.exp(params["ln_sig_zx_h"])
    sig_zy_h = np.exp(params["ln_sig_zy_h"])
    sig_zc_h = np.exp(params["ln_sig_zc_h"])
    sig_rx_h = np.exp(params["ln_sig_rx_h"])
    sig_ry_h = np.exp(params["ln_sig_ry_h"])
    
    K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h = kernel_mat_LNC_lin(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h)
    n = K_ax_h.shape[0]
    
    
    
    
    #X_x = zc
    #X_xc = zcc
    #X_y = np.hstack([x,zc])
    #X_yc = np.hstack([x,zcc])

    X_x = np.hstack([zx, zc])
    X_xc = np.hstack([zxc, zcc]) 
    X_x_cross = zy   

    X_y = np.hstack([x,zy, zc])
    X_yc = np.hstack([x,zyc, zcc])
    X_y_cross = zx
    
    ws = np.ones(X_x.shape[1])
    weights_x, resids_x, x_hat = rrModel(lam, X_x, x, ws)
    _, _, xc_hat = rrModel(lam, X_xc, x, ws)
    ws = np.ones(X_x_cross.shape[1])
    _, _, x_cross_hat = rrModel(lam, X_x_cross, x, ws)
    
    ws = np.ones(X_y.shape[1])
    weights_y, resids_y, y_hat = rrModel(lam, X_y, y, ws)
    _, _, yc_hat = rrModel(lam, X_yc,  y, ws)
    ws = np.ones(X_y_cross.shape[1])
    _, _, y_cross_hat = rrModel(lam, X_y_cross, x, ws)


    # stats for reg coeffs
    ws = np.ones(X_x.shape[1])
    stat_x = linRegCoefStat(np.array(0.00001), X_x, x, ws)
    stat_x = stat_x[zx.shape[1]]
    ws = np.ones(X_y.shape[1])
    stat_y = linRegCoefStat(np.array(0.00001), X_y, y, ws)
    stat_y = stat_y[zy.shape[1]+1]
    statss = 1-(stat_x+stat_y)/100
    
    
    
    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    #distsRx = covariance_matrix(sqeuclidean_distance, resids_x, resids_x)
    #sig_rx_h = 1 / np.quantile(distsRx, 0.5)
    distsRy = covariance_matrix(sqeuclidean_distance, resids_y, resids_y)
    sig_ry_h = 1 / np.quantile(distsRy, 0.5)
    


    #K_rx_h = rbf_kernel_matrix({'gamma': sig_rx_h}, resids_x, resids_x)
    K_ry_h = rbf_kernel_matrix({'gamma': sig_ry_h}, resids_y, resids_y)

    
    # additivity
    # hsic resids_x, (zx, zc)
    #hsic_rx_ax = hsic(K_rx_h, K_ax_h)
    # hsic resids_y, (x, zx, zc)
    #hsic_ry_ay = hsic(K_ry_h, K_ay_h)
    #hsic_resids = np.log(hsic_rx_ax) +  np.log(hsic_ry_ay)

    # indep causes: between (zx, zy, zc) and between zy and x
    hsic_zx_zy = hsic(K_zx_h, K_zy_h)
    #hsic_zx_zy = np.max(np.array([hsic_zx_zy, 0.02]))
    hsic_zx_zc = hsic(K_zx_h, K_zc_h)
    #hsic_zx_zc = np.max(np.array([hsic_zx_zc, 0.02]))
    hsic_zy_zc = hsic(K_zy_h, K_zc_h)
    #hsic_zy_zc = np.max(np.array([hsic_zy_zc, 0.02]))
    hsic_zy_x  = hsic(K_zy_h, K_x_h)
    #hsic_zy_x = np.max(np.array([hsic_zy_x, 0.02]))
    # conditional independence between zx and y given x.. gonna do this for the additive assumption here
    hsic_ry_zx = hsic(K_zx_h, K_ry_h)
    #hsic_ry_zx = np.max(np.array([hsic_ry_zx, 0.02]))
    
    hsic_indep =  np.log(hsic_zx_zc+0.02) + np.log(hsic_zy_zc+0.02) + np.log(hsic_zx_zy+0.02) + np.log(hsic_zy_x+0.02)   + np.log(hsic_ry_zx+0.02)
     
        
    # desirable hsics of x, y vs zc    
    hsic_x_zc = hsic(K_x_h, K_zc_h)
    hsic_y_zc = hsic(K_y, K_zc_h)
    hsic_xy_zc = np.log(hsic_x_zc) + np.log(hsic_y_zc)
    
    # mse residual
    mse_rx = mse(x, x_hat)
    mse_ry = mse(y, y_hat)
    mse_rx_cross = mse(x, x_cross_hat)
    mse_rx_cross = np.min(np.array([mse_rx_cross, 1]))
    mse_ry_cross = mse(y, y_cross_hat)
    mse_ry_cross = np.min(np.array([mse_ry_cross, 1]))
    mse_rxc = mse(x, xc_hat)
    mse_ryc = mse(y, yc_hat)
    
    
    mses = np.log(mse_rx) +  np.log(mse_ry)  - np.log(mse_rx_cross) -  np.log(mse_ry_cross) #+ np.log(mse_rxc) +  np.log(mse_ryc)
    #mses = np.log(statss)    

    # CI condition
    
    # Zs size
    zx_norm = np.dot(zx.T, zx)[1,1]
    zy_norm = np.dot(zy.T, zy)[1,1]
    zc_norm = np.dot(zc.T, zc)[1,1]
    z_norm = np.log(zx_norm) + np.log(zy_norm) + np.log(zc_norm)
    
    # alpha_c norm
    alphac_norm = np.dot(alpha_c.T, alpha_c)[1,1]
    
    # calcualte compute los
    loss_value =  beta * mses  + neta*hsic_indep   + nu*z_norm - lu*hsic_xy_zc + weight_orth
    
    #print("loss value dim: ", loss_value.shape)
    
    return loss_value[0]

dloss_LNC_lin = jax.grad(loss_LNC_lin, )
dloss_LNC_lin_jitted = jax.jit(dloss_LNC_lin)

# for reporting purposes give back all terms separatley
@jax.jit
def model_LNC_lin(params, lam, loss_data, K_t):

    
    x, y, Z, U, U_ica,  K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds, idx_min, idx_x_c, idx_y_c, mixMat = loss_data

    alpha_x = params["alpha_x"]
    alpha_y = params["alpha_y"]
    alpha_c = params["alpha_c"]

    alphap_x = mixMat@alpha_x
    alphap_y = mixMat@alpha_y
    
    px = alpha_x.shape[1]
    py = alpha_y.shape[1]
    p = alpha_c.shape[0]
    one_p = np.ones(p)[:,None]
    one_px = np.ones(px)[:,None]
    one_py = np.ones(py)[:,None] 
    weight_orth = (one_px.T@alphap_x.T@alpha_c)/px + (one_py.T@alphap_y.T@alpha_c)/py + (one_px.T@alphap_x.T@alphap_y@one_py)/(p*p)
    weight_orth = weight_orth[0,0]**2    

    zx = U_ica@alpha_x
    zy = U_ica@alpha_y
    zc = U@alpha_c

    
    #zx = (zx - np.mean(zx)) / (np.std(zx))
    #zy = (zy - np.mean(zy)) / (np.std(zy))
    #zc = (zc - np.mean(zc)) / (np.std(zc))
    #zx = np.array([stdrze(zx[:,i]) for i in range(zx.shape[1])]).T
    #zy = np.array([stdrze(zy[:,i]) for i in range(zy.shape[1])]).T

    
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_zx_h = np.exp(params["ln_sig_zx_h"])
    sig_zy_h = np.exp(params["ln_sig_zy_h"])
    sig_zc_h = np.exp(params["ln_sig_zc_h"])
    sig_rx_h = np.exp(params["ln_sig_rx_h"])
    sig_ry_h = np.exp(params["ln_sig_ry_h"])
    
    
    
    K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h = kernel_mat_LNC_lin(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h) 
    n = K_ax_h.shape[0]

    
    #X_x = zc
    #X_y = np.hstack([x,zc])
    
    X_x = np.hstack([zx, zc])
    X_y = np.hstack([x,zy, zc])
    X_x_cross = zy   
    X_y_cross = zx
    
    ws = np.ones(X_x_cross.shape[1])
    _, _, x_cross_hat = rrModel(lam, X_x_cross, x, ws)
    ws = np.ones(X_y_cross.shape[1])
    _, _, y_cross_hat = rrModel(lam, X_y_cross, x, ws)


    ws = np.ones(X_x.shape[1])
    weights_x, resids_x, x_hat = rrModel(lam, X_x, x, ws)
    ws = np.ones(X_y.shape[1])
    weights_y, resids_y, y_hat = rrModel(lam, X_y, y, ws)
    
    ws = np.ones(x.shape[1])
    beta_x, _, _ = rrModel(lam, x, y, ws)
    ws = np.ones(U.shape[1])
    beta_u, _, _ = rrModel(lam, U, y, ws)
    
    beta_ce = weights_y[0]
    beta_x = beta_x[0]
    beta_u = beta_u[0]

    # stats for reg coeffs
    ws = np.ones(X_x.shape[1])
    stat_x = linRegCoefStat(np.array(0.00001), X_x, x, ws)
    stat_x = stat_x[zx.shape[1]]
    ws = np.ones(X_y.shape[1])
    stat_y = linRegCoefStat(np.array(0.00001), X_y, y, ws)
    stat_y = stat_y[zy.shape[1]+1]
    
        
    x_std, y_std = stds

    beta_ce = beta_ce/x_std*y_std
    beta_x = beta_x/x_std*y_std
    beta_u = beta_u/x_std*y_std

    bias_beta = np.abs(beta_real-beta_ce)
    bias_beta_u = np.abs(beta_real-beta_u)
    bias_beta_x = np.abs(beta_real-beta_x)
    bias_beta_rel_u = bias_beta/bias_beta_u
    bias_beta_rel_x= bias_beta/bias_beta_x
    
    
    # right now we simply overwrite residual lengthscale sig_r_h to be median heuristic
    #distsRx = covariance_matrix(sqeuclidean_distance, resids_x, resids_x)
    #sig_rx_h = 1 / np.quantile(distsRx, 0.5)
    distsRy = covariance_matrix(sqeuclidean_distance, resids_y, resids_y)
    sig_ry_h = 1 / np.quantile(distsRy, 0.5)
    #K_rx_h = rbf_kernel_matrix({'gamma': sig_rx_h}, resids_x, resids_x)
    K_ry_h = rbf_kernel_matrix({'gamma': sig_ry_h}, resids_y, resids_y)

    
    # additivity
    # hsic resids_x, (zx, zc)
    #hsic_rx_ax = hsic(K_rx_h, K_ax_h)
    # hsic resids_y, (x, zx, zc)
    #hsic_ry_ay = hsic(K_ry_h, K_ay_h)
    #hsic_resids = np.log(hsic_rx_ax) +  np.log(hsic_ry_ay)

    
    # indep causes: between (zx, zy, zc) and between zy and x
    hsic_zx_zy = hsic(K_zx_h, K_zy_h)
    hsic_zx_zc = hsic(K_zx_h, K_zc_h)
    hsic_zy_zc = hsic(K_zy_h, K_zc_h)
    hsic_zy_x  = hsic(K_zy_h, K_x_h)
    # conditional independence between zx and y given x.. gonna do this for the additive assumption here
    hsic_ry_zx = hsic(K_zx_h, K_ry_h)
    hsic_indep = np.log(hsic_zx_zy) + np.log(hsic_zx_zc) + np.log(hsic_zy_zc) + np.log(hsic_zy_x) + np.log(hsic_ry_zx)
     
    
    # desirable hsics of x, y vs zc    
    hsic_x_zc = hsic(K_x_h, K_zc_h)
    hsic_y_zc = hsic(K_y, K_zc_h)  
    hsic_xy_zc = np.log(hsic_x_zc) + np.log(hsic_y_zc)
        
    # mse residual
    mse_rx = mse(x, x_hat)
    mse_ry = mse(y, y_hat)
    mse_rx_cross = mse(x, x_cross_hat)
    mse_rx_cross = np.min(np.array([mse_rx_cross, 1]))
    mse_ry_cross = mse(y, y_cross_hat)
    mse_ry_cross = np.min(np.array([mse_ry_cross, 1]))

    mses = np.log(mse_rx) +  np.log(mse_ry) - np.log(mse_rx_cross) -  np.log(mse_ry_cross) #+ np.log(mse_rxc) +  np.log(mse_ryc)
    
    
    

    # hsic (zs, zs_reals)
    
    
    idx_x, idx_y, idx_c = idxs
    
    zx_real = Z[:,idx_x]
    zy_real = Z[:,idx_y]
    zc_real = Z[:,idx_c]
    
    distsZx_real = covariance_matrix(sqeuclidean_distance, zx_real, zx_real)
    sig_zx_real = 1 / np.quantile(distsZx_real, 0.5)
    K_zx_real = rbf_kernel_matrix({'gamma': sig_zx_real}, zx_real, zx_real)
    
    distsZy_real = covariance_matrix(sqeuclidean_distance, zy_real, zy_real)
    sig_zy_real = 1 / np.quantile(distsZy_real, 0.5)
    K_zy_real = rbf_kernel_matrix({'gamma': sig_zy_real}, zy_real, zy_real)
    
    distsZc_real = covariance_matrix(sqeuclidean_distance, zc_real, zc_real)
    sig_zc_real = 1 / np.quantile(distsZc_real, 0.5)
    K_zc_real = rbf_kernel_matrix({'gamma': sig_zc_real}, zc_real, zc_real)
    
    
    hsic_zx = hsic(K_zx_h, K_zx_real)
    hsic_zy = hsic(K_zy_h, K_zy_real)
    hsic_zc = hsic(K_zc_h, K_zc_real)
    
    # correlations
    corr_zc = np.abs(corrcoef(zc_real, zc))
    
    #impurites
    imp_zx = hsic(K_zx_h, K_zc_real)
    imp_zy = hsic(K_zy_h, K_zc_real)
    
    
    
    monitor = {}
    monitor = {    
        #'hsic_rx_ax': hsic_rx_ax,
        #'hsic_ry_ay': hsic_ry_ay,
        #'hsic_resids': hsic_resids,
        'hsic_zx_zy': hsic_zx_zy,
        'hsic_zx_zc': hsic_zx_zc,
        'hsic_zy_zc': hsic_zy_zc,
        'hsic_zy_x': hsic_zy_x,
        'hsic_ry_zx': hsic_ry_zx,
        'hsic_indep': hsic_indep,
        'weight_orth': weight_orth,
        'mse_rx': mse_rx,
        'mse_ry': mse_ry,
        'mse_rx_cross': mse_rx_cross,
        'mse_ry_cross': mse_ry_cross,
        'stat_x': stat_x,
        'stat_y': stat_y,
        'mses': mses,
        'hsic_zx': hsic_zx,
        'hsic_zy': hsic_zy,
        'hsic_zc': hsic_zc,
        'corr_zc': corr_zc,
        'imp_zx': imp_zx,
        'imp_zy': imp_zy,
        'hsic_x_zc': hsic_x_zc,
        'hsic_y_zc': hsic_y_zc,
        'hsic_xy_zc': hsic_xy_zc,
        'beta_ce': beta_ce,
        'beta_u': beta_u,
        'beta_x': beta_x,
        'bias_beta': bias_beta,
        'bias_beta_u':bias_beta_u,
        'bias_beta_x':bias_beta_x,
        'bias_beta_rel_u':bias_beta_rel_u,
        'bias_beta_rel_x':bias_beta_rel_x
    }

    return monitor

def getIniMonitor_LNC_lin(epochs, report_freq, reps, parts): 
        num_reports = int(np.ceil(epochs / report_freq))+1 # initial report 
        print("num_reports: ", num_reports)
        monitors = {
            'loss': onp.zeros([num_reports, reps, parts]),
            #'hsic_rx_ax': onp.zeros([num_reports, reps, parts]),
            #'hsic_ry_ay': onp.zeros([num_reports, reps, parts]),
            #'hsic_resids': onp.zeros([num_reports, reps, parts]),
            'hsic_zx_zy': onp.zeros([num_reports, reps, parts]),
            'hsic_zx_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_zy_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_zy_x': onp.zeros([num_reports, reps, parts]),
            'hsic_ry_zx': onp.zeros([num_reports, reps, parts]),
            'hsic_indep': onp.zeros([num_reports, reps, parts]),
            'weight_orth': onp.zeros([num_reports, reps, parts]),	
            'hsic_x_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_y_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_xy_zc': onp.zeros([num_reports, reps, parts]),
            'mse_rx': onp.zeros([num_reports, reps, parts]),
            'mse_ry': onp.zeros([num_reports, reps, parts]),
            'mse_rx_cross': onp.zeros([num_reports, reps, parts]),
            'mse_ry_cross': onp.zeros([num_reports, reps, parts]),
            'mses': onp.zeros([num_reports, reps, parts]),
            'stat_x': onp.zeros([num_reports, reps, parts]),
            'stat_y': onp.zeros([num_reports, reps, parts]),
            'corr_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_zx': onp.zeros([num_reports, reps, parts]),
            'hsic_zy': onp.zeros([num_reports, reps, parts]),
            'hsic_zc': onp.zeros([num_reports, reps, parts]),
            'imp_zx': onp.zeros([num_reports, reps, parts]),
            'imp_zy': onp.zeros([num_reports, reps, parts]),
            'beta_ce': onp.zeros([num_reports, reps, parts]),
            'beta_u': onp.zeros([num_reports, reps, parts]),
            'beta_x': onp.zeros([num_reports, reps, parts]),
            'bias_beta': onp.zeros([num_reports, reps, parts]),
            'bias_beta_u':onp.zeros([num_reports, reps, parts]),
            'bias_beta_x':onp.zeros([num_reports, reps, parts]),
            'bias_beta_rel_u':onp.zeros([num_reports, reps, parts]),
            'bias_beta_rel_x':onp.zeros([num_reports, reps, parts])
            
        }
        return monitors


                    
def fillMonitor_LNC_lin(params, params_or, pars, loss_as_par, dloss_as_par_jitted, lossData, iteration, report_freq, rep, part, reps, monitors, K_t):
    
    print("enters filMonitor_LNC_lin")
    _, _, lam, _,_ = pars
          
    
    monitor = model_LNC_lin(params, lam, lossData, K_t)
    print("monitor")
    print(monitor)
    ws = np.ones(params["alpha_x"].shape[0])
    loss_val = loss_as_par(params, pars, lossData,ws, params["alpha_x"], params["alpha_y"], params["alpha_c"])
    indxRep = int(iteration / report_freq) 
    print("indxRep:", indxRep, " iteration: ", iteration, " rep: ", rep, " part: ", part)
    
    
    monitors['loss'][indxRep, rep, part] = loss_val
    #monitors['hsic_rx_ax'][indxRep, rep, part] = monitor['hsic_rx_ax']
    #monitors['hsic_ry_ay'][indxRep, rep, part] = monitor['hsic_ry_ay']
    #monitors['hsic_resids'][indxRep, rep, part] = monitor['hsic_resids']
    monitors['hsic_zx_zy'][indxRep, rep, part] = monitor['hsic_zx_zy']
    monitors['hsic_zx_zc'][indxRep, rep, part] = monitor['hsic_zx_zc']
    monitors['hsic_zy_zc'][indxRep, rep, part] = monitor['hsic_zy_zc']
    monitors['hsic_zy_x'][indxRep, rep, part] = monitor['hsic_zy_x']
    monitors['hsic_ry_zx'][indxRep, rep, part] = monitor['hsic_ry_zx']
    monitors['hsic_indep'][indxRep, rep, part] = monitor['hsic_indep']
    monitors['hsic_x_zc'][indxRep, rep, part] = monitor['hsic_x_zc']
    monitors['hsic_y_zc'][indxRep, rep, part] = monitor['hsic_y_zc']
    monitors['hsic_xy_zc'][indxRep, rep, part] = monitor['hsic_xy_zc']
    monitors['mse_rx'][indxRep, rep, part] = monitor['mse_rx']
    monitors['mse_ry'][indxRep, rep, part] = monitor['mse_ry']
    monitors['mse_rx_cross'][indxRep, rep, part]= monitor['mse_rx_cross']
    monitors['mse_ry_cross'][indxRep, rep, part]= monitor['mse_ry_cross']
    monitors['mses'][indxRep, rep, part] = monitor['mses']
    monitors['stat_x'][indxRep, rep, part] = monitor['stat_x']
    monitors['stat_y'][indxRep, rep, part] = monitor['stat_y']
    monitors['hsic_zx'][indxRep, rep, part] = monitor['hsic_zx']
    monitors['hsic_zy'][indxRep, rep, part] = monitor['hsic_zy']
    monitors['hsic_zc'][indxRep, rep, part] = monitor['hsic_zc']
    monitors['corr_zc'][indxRep, rep, part] = monitor['corr_zc']
    monitors['imp_zx'][indxRep, rep, part] = monitor['imp_zx']
    monitors['imp_zy'][indxRep, rep, part] = monitor['imp_zy']
    monitors['beta_ce'][indxRep, rep, part] = monitor['beta_ce']
    monitors['beta_x'][indxRep, rep, part] = monitor['beta_x']
    monitors['beta_u'][indxRep, rep, part] = monitor['beta_u']
    monitors['bias_beta'][indxRep, rep, part]= monitor["bias_beta"]
    monitors['bias_beta_u'][indxRep, rep, part]=monitor["bias_beta_u"]
    monitors['bias_beta_x'][indxRep, rep, part]=monitor["bias_beta_x"]
    monitors['bias_beta_rel_u'][indxRep, rep, part]=monitor["bias_beta_rel_u"]
    monitors['bias_beta_rel_x'][indxRep, rep, part]= monitor['bias_beta_rel_x']
    
    
    
    print("exits filMonitor_LNC_lin")                
                    
    return monitors

def getLatentZ_LNC_lin(params, loss_as_par, dloss_as_par_jitted, loss_data, pars, epochs, report_freq, reps, batch_size, batch_per_epoch, learning_rate, smplsParts, monitors, batches, batches2, K_t):
    x, _, _, _, _, _, _, _, _, _, _, _, _,_,_,_,_ = loss_data
    
    N = x.shape[0]
    p = params["alpha_x"].shape[0]
         
    print("report_freq: ", report_freq)
    parts = len(smplsParts)
    
    ms = onp.zeros([p, epochs+1, reps])
    vs = onp.zeros([p, epochs+1, reps])
    epochs2 = int(onp.floor(epochs/batch_per_epoch))
    loss_vals = onp.ones([epochs+1, reps])*onp.Inf
    loss_vals_epoch = onp.ones([epochs2, reps])*onp.Inf
    
    
    lrs = learning_rate*onp.ones(reps)
    adamOpt = False
    


    for iteration in range(epochs+1):
      
        # print("*********************")
        if (iteration % 50 == 0):
            print("iteration: ", iteration)
        #print("nans: ", onp.sum(onp.isnan(onp.array(params["Z"]))))

        epoch = (iteration-1) // batch_per_epoch
        batch = (iteration-1) % batch_per_epoch
        
        #epoch = bisect.bisect_left(cumbatches_per_epoch, iteration)-1
        #batch = iteration -cumbatches_per_epoch[epoch]-1
        
        #print("iteration: ", iteration)
        #print("epoch: ", epoch)
        #print("batch: ", batch)

        


        # get the gradient of the loss
        for rep in range(reps):
            #print("rep: ", rep)


            if(iteration == 0):
                smpl = onp.random.randint(low=0, high=N, size=batch_size)
                smpl2 = onp.random.randint(low=0, high=N, size=batch_size*2)
            else:
                #print(np.unique(np.hstack(batches[rep][epoch])).shape)
                #print(np.unique(np.hstack(batches2[rep][epoch])).shape)
                smpl = batches[rep][epoch][batch]
                smpl2 = batches2[rep][epoch][batch]
                

            #print("len smpl: ", len(smpl))
            #print("len smpl 2: ", len(smpl2))
             

            if iteration == 5:
                print("smpl: ", smpl[0:4])
            
            #smpl2 = onp.random.randint(low=0, high=N, size=batch_size)
            
            #smpl3 = onp.linspace(0,n-1,n, dtype=int)
            # sampling without replacemnt
            #smpl = onp.random.choice(a=n, size=batch_size, replace=False)

  
            loss_data_aux = getDataLoss_LNC_lin(smpl, loss_data)
            loss_data_aux2 = getDataLoss_LNC_lin(smpl2, loss_data)

  
            # equal weights
            ws = np.ones(2)
            

            # algorithmic independence forcing

            # random weights - so that E[y|x] alg indep of p(x)
            #ws = getWeightsAlgoIndep(loss_data, batch_size, smpl)
            #print("ws: ", ws[0:5])

            
            # prepare parameters for grad calculation (subsample)
            
            #params_aux = getParamsForGrad(params, rep, optType, smpl)
            params_aux = getParamsForGrad_LNC_lin(params, rep, smpl)
            params_aux2 = getParamsForGrad_LNC_lin(params, rep, smpl2)
            #params_aux3 = getParamsForGrad(params, rep, smpl3)
           
            
             
            if adamOpt:
                grad_params = grad(loss_as_par, argnums=0)(params_aux, pars, loss_data_aux, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
                if iteration > 0:
                    m = ms[smpl,iteration-1,rep][:,None]
                    v = vs[smpl,iteration-1,rep][:,None]
                    m, v = updateParamsAdam2_LNC_lin(params, grad_params, smpl, iteration, rep, lrs[rep], iteration-1, m, v)
                    ms[smpl,iteration,rep] = m[:,0]
                    vs[smpl,iteration,rep] = v[:,0]
            

            else:
                grad_params = dloss_as_par_jitted(params_aux, pars, loss_data_aux, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
                if iteration > 0:
                    updateParams_LNC_lin(params, grad_params, smpl, iteration, rep, lrs[rep])
                    
            ws = np.ones(batch_size*2)
            loss_vals[iteration, rep] = loss_as_par(params_aux2, pars, loss_data_aux2, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
            	

 
            if (iteration % report_freq == 0) & (iteration != 0):
                #print("report")
                print("iteration report: ", iteration)
                
                for part in range(parts):
                    
                    smplPart = smplsParts[part]
                    params_part = getParamsForGrad_LNC_lin(params, rep, smplPart)
                    #params_part = getParamsSmpl(params, smplPart)
                    loss_data_part = getDataLoss_LNC_lin(smplPart, loss_data)
                    monitors = fillMonitor_LNC_lin(params_part, params, pars, loss_as_par, dloss_as_par_jitted, loss_data_part, iteration, report_freq, rep, part, reps, monitors, K_t)
                               

                    #nPerPart = smplPart.shape[0]
                    #ws = np.ones(nPerPart)
                    #loss_val = loss_as_par(params_part, pars, loss_data_part, ws)
         
                              
    
    return params, monitors #, resids, bestResids, bestZ


def getLatentZ_wrapper_lin(x, y, Z, U, idxs, stds, beta_real, nm, pars, num_epochs, report_freq, num_reps, batch_size, learning_rate, job):
    print("nm:", nm)
    N = x.shape[0]
    
    D_x = covariance_matrix(sqeuclidean_distance, x, x)
    sigma_x_med = 1 / np.median(D_x)
    K_x = rbf_kernel_matrix({'gamma': sigma_x_med}, x, x)

    D_y = covariance_matrix(sqeuclidean_distance, y, y)
    sigma_y = 1 / np.median(D_y)
    K_y = rbf_kernel_matrix({'gamma': sigma_y}, y, y)
    
    D_u = covariance_matrix(sqeuclidean_distance, U, U)
    sigma_u = 1 / np.median(D_u)
    K_u = rbf_kernel_matrix({'gamma': sigma_u}, U, U)
    
    numComps = onp.min([Z.shape[1], Z.shape[0]-3])
    transformer = FastICA(n_components=numComps,random_state=0,whiten='unit-variance')
    U_ica = transformer.fit_transform(onp.array(U))
    print("U_ica shape: ", U_ica.shape)
    mixMat = transformer.components_
    mixMat  = mixMat.T
    pvalsy = [sm.OLS(onp.array(y), sm.add_constant(onp.hstack([x, U_ica[:,i][:,None]]))).fit().pvalues[2] for i in range(U_ica.shape[1])]
    pvalsx = [sm.OLS(onp.array(x), sm.add_constant(U_ica[:,i][:,None])).fit().pvalues[1] for i in range(U_ica.shape[1])]
    pvals = np.array([pvalsx, pvalsy])
    sumpvals = onp.apply_along_axis(np.sum, 0, pvals)
    indx_min = onp.argmin(sumpvals)
    print("indx_min: ", indx_min)
    idx_rest  = list(set(onp.arange(0, U_ica.shape[1])).difference(set([indx_min])))
    print("idx_rest: ", idx_rest)
    modMat_aux = sm.add_constant(onp.hstack([x, U_ica[:,indx_min][:,None] ,U_ica[:,idx_rest]] ))
    numRows = modMat_aux.shape[0]
    maxCols = onp.min([numRows, modMat_aux.shape[1]])
    #modMat_aux = modMat_aux[:, 0:maxCols]
    print("modMat shape: ", modMat_aux.shape)
    pvals_resty = sm.OLS(onp.array(y), modMat_aux).fit().pvalues
    pvals_resty = pvals_resty[3:(pvals_resty.shape[0])]
    modMat_aux = sm.add_constant(onp.hstack([U_ica[:,indx_min][:,None], U_ica[:,idx_rest]] ))
    numRows = modMat_aux.shape[0]
    maxCols = onp.min([numRows, modMat_aux.shape[1]])
    #modMat_aux = modMat_aux[:, 0:maxCols]
    print("modMat shape: ", modMat_aux.shape)
    pvals_restx = sm.OLS(onp.array(x), modMat_aux).fit().pvalues
    pvals_restx = pvals_restx[2:(pvals_restx.shape[0])]

    print("pvals_resty: ", pvals_resty)
    print("pvals_restx: ", pvals_restx)
    idx_y_c, = onp.where( (pvals_resty<pvals_restx)&(pvals_resty<0.05))
    idx_y_c = onp.array(idx_rest)[idx_y_c]
    if idx_y_c.shape[0]==0:
        idx_y_c, = onp.where( (pvals_resty<pvals_restx))
        idx_y_c = onp.array(idx_rest)[idx_y_c]
    if idx_y_c.shape[0]==0:
        idx_y_c = onp.argsort(pvals_resty)[0:2]
        idx_y_c = onp.array(idx_rest)[idx_y_c]

    print("idx_y_c: ", idx_y_c)
    idx_x_c, = onp.where( (pvals_resty>pvals_restx)&(pvals_restx<0.05))
    idx_x_c = onp.array(idx_rest)[idx_x_c]
    if idx_x_c.shape[0]==0:
        idx_x_c, = onp.where( (pvals_resty>pvals_restx))
        idx_x_c = onp.array(idx_rest)[idx_x_c]
    if idx_x_c.shape[0]==0:
        idx_x_c = onp.argsort(pvals_restx)[0:2]
        idx_x_c = onp.array(idx_rest)[idx_x_c]
    print("idx_x_c: ", idx_x_c)

    onp.random.seed(seed=job)
    
    maxMonitor = 1000
    parts = int(onp.ceil(N/maxMonitor))
    smplsParts = onp.random.choice(parts, size=N)
    

    smplsParts = [myWhere(smplsParts==i) for i in range(parts)]
    #print("smplsParts: ", smplsParts)

    
    lossData = x, y, Z, U, U_ica, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds, indx_min, idx_x_c, idx_y_c, mixMat
    

    onp.random.seed(seed=job+3)
    params = getIniPar_LNC_lin(num_reps, lossData, pars, smplsParts)
    print("alpha x shape: ",params["alpha_x"].shape)
    
    res = []
    


    onp.random.seed(seed=job+123)
    batch_per_epoch = int(onp.ceil(N / batch_size))
    epochs2 = int(onp.ceil(num_epochs / batch_per_epoch))
    num_iters = int(epochs2*batch_per_epoch)
    batches = [[get_batches_one_epoch(batch_per_epoch, batch_size, N) for i in range(epochs2)] for j in range(num_reps)]
    #batches2 = [[get_batches_one_epoch(batch_per_epoch, batch_size, N) for i in range(epochs2)] for j in range(num_reps)]
    batches2 = [[ [onp.random.randint(low=0, high=N, size=batch_size*2) for k in range(batch_per_epoch)] for i in range(epochs2)] for j in range(num_reps)]
    print("N: ", N)
    print("batch size: ", batch_size)
    print("batch per_epoch: ", batch_per_epoch)
    print("epochs2: ", epochs2)
    print("epochs eff: ", len(batches[0]))
    print("num_iters: ", num_iters)

    report_freq2 = num_iters // 1

    path = getIniMonitor_LNC_lin(num_iters, report_freq, num_reps, parts)
    


    loss_as_par = loss_LNC_lin
    dloss_as_par_jitted = dloss_LNC_lin_jitted
    

        
        
    start = time.process_time()
    params, path = getLatentZ_LNC_lin(params, loss_as_par, dloss_as_par_jitted, loss_data=lossData, pars=pars,epochs=num_iters, report_freq=report_freq,
                                                         reps=num_reps, batch_size=batch_size, batch_per_epoch=batch_per_epoch, learning_rate=learning_rate, smplsParts=smplsParts, monitors=path, batches=batches, batches2=batches2, K_t=None)
    res.append(path)
    
    res = {k: onp.concatenate([r[k] for r in res], axis=0) for k in res[0].keys()}
    
    params["zc"] = U@params["alpha_c"]

    print("results")
    res = {"params": params, "path": res}
    #print(res["path"])
    

    return res








