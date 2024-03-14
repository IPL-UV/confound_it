from sklearn.manifold import Isomap
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

def myWhere(x):
       res,  = onp.where(x)
       return res

def norml(x):
    return (x-onp.min(x))/(onp.max(x)-onp.min(x))-0.5
def norml_mat(x):
    return onp.apply_along_axis(norml,0,x)

@jax.jit
def normalize(m):
    o = np.argsort(m)
    mp = np.argsort(o)
    #sns.distplot(mp)
    min_mp = np.min(mp)-0.001
    max_mp = np.max(mp)+0.001
    mpp = (mp-min_mp)/(max_mp-min_mp)
    #sns.distplot(mpp)
    #print(np.min(mpp), np.max(mpp))
    mppp = ndtri(mpp)
    return(mppp)

def stdrze(x):
    return (x-onp.mean(x))/onp.std(x)
def stdrze_mat(x):
    return onp.apply_along_axis(stdrze,0,x)


@jax.jit
def corrcoef(x, y):
    x_stdr = (x-np.mean(x))/np.std(x)
    y_stdr = (y-np.mean(y))/np.std(y)
    res = (np.dot(x_stdr.T,y_stdr)/x_stdr.shape[0])
    #print("corrcoef",res.shape)
    return res[0]

# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x, y):
    return np.sum((x - y) ** 2)


# RBF Kernel
@jax.jit
def rbf_kernel(params, x, y):
    return np.exp(- params['gamma'] * sqeuclidean_distance(x, y))


# Covariance Matrix
def covariance_matrix(kernel_func, x, y):
    mapx1 = jax.vmap(
        lambda x, y: kernel_func(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(
        lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


# Covariance Matrix
def rbf_kernel_matrix(params, x, y):
    mapx1 = jax.vmap(lambda x, y: rbf_kernel(params, x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)

@jax.jit
def krrModel(lam, K, y, ws):
    # cho factor the cholesky
    # L = linalg.cho_factor(K + lam * np.eye(K.shape[0]))
    #L = linalg.cho_factor(K + lam * np.diag(1 / ws))
    #n = K.shape[0]
    L = linalg.cho_factor(K + lam * np.diag(1 / ws))

    # weights
    weights = linalg.cho_solve(L, y)

    # save the params

    y_hat = np.dot(K, weights)

    resids = y - y_hat

    # return the predictions
    return weights, resids, y_hat

@jax.jit
def krrModel_lin(lam, K, x, y, ws):
    # cho factor the cholesky
    
    I = np.diag(1 / ws)
    B1 = linalg.inv(K + lam * I)
    B2 = I + np.dot(K, B1)
    beta_num = np.dot(np.dot(x.T, B2), y)
    beta_den= lam + np.dot(np.dot(x.T, B2), x)
    beta = beta_num / beta_den
    
    # weights
    #weights = linalg.cho_solve(L, y-x*beta)
    weights = np.dot(B1, y-x*beta)

    # save the params

    y_hat = np.dot(K, weights) + beta*x

    resids = y - y_hat

    # return the predictions
    return weights, beta, resids, y_hat

@jax.jit
def centering(K):
    n_samples = K.shape[0]
    logging.debug(f"N: {n_samples}")
    logging.debug(f"I: {np.ones((n_samples, n_samples)).shape}")
    H = np.eye(K.shape[0], ) - (1 / n_samples) * np.ones((n_samples, n_samples))
    return np.dot(np.dot(H, K), H)


# Normalized Hsic - from kernels
@jax.jit
def hsic(K_x, K_z):
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)

# Normalized Hsic - from features using rbf kernels
@jax.jit
def hsicRBF(x, z):
    distsX = covariance_matrix(sqeuclidean_distance, x, x)
    sigma = 1 / np.median(distsX)
    K_x = rbf_kernel_matrix({'gamma': sigma}, x, x)
    distsZ = covariance_matrix(sqeuclidean_distance, z, z)
    sigma = 1 / np.median(distsZ)
    K_z = rbf_kernel_matrix({'gamma': sigma}, z, z)
    K_x = centering(K_x)
    K_z = centering(K_z)
    return np.sum(K_x * K_z) / np.linalg.norm(K_x) / np.linalg.norm(K_z)

@jax.jit
def mse(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


@jax.jit
def getDataLoss_LNC(smpl, loss_data):
    
    x, y, Z, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds = loss_data 
    
    

    D_x_aux = D_x[smpl, :]
    D_x_aux = D_x_aux[:, smpl]
    K_x_aux = K_x[smpl, :]
    K_x_aux = K_x_aux[:, smpl]
    #K_zmani_aux = K_zmani[smpl,:]
    #K_zmani_aux = K_zmani_aux[:,smpl]
    K_y_aux = K_y[smpl,:]
    K_y_aux = K_y_aux[:,smpl]
    K_u_aux = K_u[smpl,:]
    #K_u_aux = K_u_aux[:,smpl]
                
    x_aux = x[smpl,]
    y_aux = y[smpl,]
    

    if Z is not None:
        Z_aux = Z[smpl,]
    else:
        Z_aux = None
        
            
    return x_aux, y_aux, Z_aux, K_u_aux, D_x_aux, K_y_aux, None, K_x, idxs, beta_real, stds


def getIniPar_LNC(reps, loss_data, pars, smplsParts): #N, m, reps, y, M
    
    
    _, _, lam, _ , _ = pars 
    x, y, Z, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds = loss_data
    
    n = x.shape[0]
    

    # initialize z_x to x, z_y to y and z_c manifold of z_c
    # we actually have to initialize alpha_x, alpha_y and alpha_c so in each case
    # we have to work out what z = U*alpha -> cholesky decomposition of U
    
    zx_ini = x
    zy_ini = y 
    
    embedding = Isomap(n_components=10)
    xy = onp.hstack([x, y])
    xy_transformed = embedding.fit_transform(xy)
    zc_ini = np.array(xy_transformed[:,3])[:,None]
    
    
    #L = linalg.cho_factor(np.dot(U.T, U) + 0.0001 * np.eye(m))
    #alpha_x = linalg.cho_solve(L, np.dot(U.T, zx_ini))
    #alpha_y = linalg.cho_solve(L, np.dot(U.T, zy_ini))
    #alpha_c = linalg.cho_solve(L, np.dot(U.T, zc_ini))
    
    
    ws = np.ones(n)
    alpha_x, _, _ = krrModel(lam, K_u, zx_ini, ws)
    alpha_x = alpha_x * np.ones(reps)
    alpha_y, _, _ = krrModel(lam, K_u, zy_ini, ws)
    alpha_y = alpha_y * np.ones(reps)
    alpha_c, _, _ = krrModel(lam, K_u, zc_ini, ws)
    alpha_c = alpha_c * np.ones(reps)
    
    #alpha_x = onp.random.normal(size=(n,reps))#[:,None]
    #alpha_y = onp.random.normal(size=(n,reps))#[:,None]
    #alpha_c = onp.random.normal(size=(n,reps))#[:,None]
    
    
    params = {
            'alpha_x': alpha_x,
            'alpha_y': alpha_y,
            'alpha_c': alpha_c,
    }
    
    #onp.random.seed(seed=4)
    
    zx_ini = np.array(onp.apply_along_axis(normalize, 0, zx_ini))
    zy_ini = np.array(onp.apply_along_axis(normalize, 0, zy_ini))
    zc_ini = np.array(onp.apply_along_axis(normalize, 0, zc_ini))
    

    
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
    
    

    #qs = [0.3, 0.9]
    qs = [0.5]
    indxs = np.arange(len(qs)).tolist()
    indxs = [[i, j, k] for i in indxs for j in indxs for k in indxs]
    indxs = np.array(indxs).T.tolist()
    qs_zx_f = np.array(qs)[np.array(indxs[0])]
    qs_zy_f = np.array(qs)[np.array(indxs[1])]
    qs_zc_f = np.array(qs)[np.array(indxs[2])]
    #qs_zx_f = np.array([0.3])
    #qs_zy_f = np.array([0.3])
    #qs_zc_f = np.array([0.9])
    print("qs_zx_f: ", qs_zx_f)
    print("qs_zy_f: ", qs_zy_f)
    print("qs_zc_f: ", qs_zc_f)
    
    sigs_zx_f= 1/np.quantile(D_zx, qs_zx_f)
    sigs_zy_f= 1/np.quantile(D_zy, qs_zy_f)
    sigs_zc_f= 1/np.quantile(D_zc, qs_zc_f)
    
    
    sigs_x_f= sigma_x_med * np.ones(reps)
    
    sigs_zx_h= sigma_zx_med * np.ones(reps)
    sigs_zy_h= sigma_zy_med * np.ones(reps)
    sigs_zc_h= sigma_zc_med * np.ones(reps)
    sigs_x_h= sigma_x_med * np.ones(reps)
    sigs_rx_h= sigma_rx_med * np.ones(reps)
    sigs_ry_h= sigma_rx_med * np.ones(reps)
    
    params['ln_sig_zx_f']= np.log(sigs_zx_f)
    params['ln_sig_zy_f']= np.log(sigs_zy_f)
    params['ln_sig_zc_f']= np.log(sigs_zc_f)
    params['ln_sig_x_f']= np.log(sigs_x_f)
    
    params['ln_sig_zx_h']= np.log(sigs_zx_h)
    params['ln_sig_zy_h']= np.log(sigs_zy_h)
    params['ln_sig_zc_h']= np.log(sigs_zc_h)
    params['ln_sig_x_h']= np.log(sigs_x_h)
    params['ln_sig_rx_h']= np.log(sigs_rx_h)
    params['ln_sig_ry_h']= np.log(sigs_ry_h)
    
    
    return params

@jax.jit
def kernel_mat_LNC(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h, sigs_f):
    # sigs_f = np.hstack([sig_x_f, sig_zx_f, sig_zy_f, sig_zc_f])
    sig_x_f = sigs_f[0]
    sig_zx_f = sigs_f[1]
    sig_zy_f = sigs_f[2]
    sig_zc_f = sigs_f[3]
    
    K_x_f = np.exp(-sig_x_f * D_x)
    K_zx_f = rbf_kernel_matrix({'gamma': sig_zx_f}, zx, zx)
    K_zy_f = rbf_kernel_matrix({'gamma': sig_zy_f}, zy, zy)
    K_zc_f = rbf_kernel_matrix({'gamma': sig_zc_f}, zc, zc)
    

    K_x_h = K_x_f #np.exp(-sig_x_h * D_x)
    K_zx_h = K_zx_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    K_zy_h = K_zy_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    K_zc_h = K_zc_f #rbf_kernel_matrix({'gamma': sig_z_h}, z, z)
    # x causes - to compare to rx
    K_ax_h = K_zx_h+K_zc_h
    # y causes - to compare to ry
    K_ay_h = K_x_h+K_zy_h+K_zc_h
    #K_a_f = 2 * K_x_f + 2 * K_z_f + K_xz_f + K_zx_f
    K_ax_f = K_zc_h#+K_zx_f
    K_ay_f = K_zc_h#+K_zy_h

    return K_ax_f, K_ay_f, K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h

# loss 
@jax.jit
def loss_LNC(params, pars, loss_data, ws, alpha_x, alpha_y, alpha_c):
    
    beta, neta, lam, nu, lu = pars 
    
    
    x, y, Z, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds = loss_data
    
    zxc = K_u@alpha_x
    zyc = K_u@alpha_y
    zcc = K_u@alpha_c
    
    alpha_x = params["alpha_x"]
    alpha_y = params["alpha_y"]
    alpha_c = params["alpha_c"]
    
    zx = K_u@alpha_x
    zy = K_u@alpha_y
    zc = K_u@alpha_c
    
    zx = (zx - np.mean(zx)) / (np.std(zx))
    zy = (zy - np.mean(zy)) / (np.std(zy))
    zc = (zc - np.mean(zc)) / (np.std(zc))
    
    zxc = (zxc - np.mean(zxc)) / (np.std(zxc))
    zyc = (zyc - np.mean(zyc)) / (np.std(zyc))
    zcc = (zcc - np.mean(zcc)) / (np.std(zcc))

    
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_zx_h = np.exp(params["ln_sig_zx_h"])
    sig_zy_h = np.exp(params["ln_sig_zy_h"])
    sig_zc_h = np.exp(params["ln_sig_zc_h"])
    sig_rx_h = np.exp(params["ln_sig_rx_h"])
    sig_ry_h = np.exp(params["ln_sig_ry_h"])
    


    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_zx_f = np.exp(params["ln_sig_zx_f"])
    sig_zy_f = np.exp(params["ln_sig_zy_f"])
    sig_zc_f = np.exp(params["ln_sig_zc_f"])
    
    
    sigs_f = np.hstack([sig_x_f, sig_zx_f, sig_zy_f, sig_zc_f])
    K_ax_f, K_ay_f, K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h = kernel_mat_LNC(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h, sigs_f)
    K_ax_fc, K_ay_fc, _, _, _, _, _, _ = kernel_mat_LNC(D_x, x, zxc, zyc, zcc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h, sigs_f)
    n = K_ax_f.shape[0]
    
    weights_x, resids_x, x_hat = krrModel(lam, K_ax_f, x, ws)
    _, _, xc_hat = krrModel(lam, K_ax_fc, x, ws)
    
    weights_y, _, resids_y, y_hat = krrModel_lin(lam, K_ay_f, x, y, ws)
    _, _, _, yc_hat = krrModel_lin(lam, K_ay_fc, x, y, ws)
    
    
    
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
    hsic_zx_zy = np.max(np.array([hsic_zx_zy, 0.02]))
    hsic_zx_zc = hsic(K_zx_h, K_zc_h)
    hsic_zx_zc = np.max(np.array([hsic_zx_zc, 0.02]))
    hsic_zy_zc = hsic(K_zy_h, K_zc_h)
    hsic_zy_zc = np.max(np.array([hsic_zy_zc, 0.02]))
    hsic_zy_x  = hsic(K_zy_h, K_x_h)
    hsic_zy_x = np.max(np.array([hsic_zy_x, 0.02]))
    # conditional independence between zx and y given x.. gonna do this for the additive assumption here
    hsic_ry_zx = hsic(K_zx_h, K_ry_h)
    hsic_ry_zx = np.max(np.array([hsic_ry_zx, 0.02]))
    
    hsic_indep = np.log(hsic_zx_zy) + np.log(hsic_zx_zc) + np.log(hsic_zy_zc) + np.log(hsic_zy_x) + np.log(hsic_ry_zx)
     
        
    # desirable hsics of x, y vs zc    
    hsic_x_zc = hsic(K_x_h, K_zc_h)
    hsic_y_zc = hsic(K_y, K_zc_h)
    hsic_xy_zc = np.log(hsic_x_zc) + np.log(hsic_y_zc)
    
    # mse residual
    mse_rx = mse(x, x_hat)
    mse_ry = mse(y, y_hat)
    mse_rxc = mse(x, xc_hat)
    mse_ryc = mse(y, yc_hat)
    
    
    mses = np.log(mse_rx) +  np.log(mse_ry) + np.log(mse_rxc) +  np.log(mse_ryc)
    
    # CI condition
    
    # Zs size
    zx_norm = np.dot(zx.T, zx)[1,1]
    zy_norm = np.dot(zy.T, zy)[1,1]
    zc_norm = np.dot(zc.T, zc)[1,1]
    z_norm = np.log(zx_norm) + np.log(zy_norm) + np.log(zc_norm)
    
    # alpha_c norm
    alphac_norm = np.dot(alpha_c.T, alpha_c)[1,1]
    
    # calcualte compute los
    loss_value =  beta * mses  + neta*hsic_indep   + nu*z_norm - lu*hsic_xy_zc
    
    return loss_value[0,0]

dloss_LNC = jax.grad(loss_LNC, )
dloss_LNC_jitted = jax.jit(dloss_LNC)

# for reporting purposes give back all terms separatley
@jax.jit
def model_LNC(params, lam, loss_data, K_t):

    x, y, Z, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds = loss_data

    alpha_x = params["alpha_x"]
    alpha_y = params["alpha_y"]
    alpha_c = params["alpha_c"]
    
    zx = K_u@alpha_x
    zy = K_u@alpha_y
    zc = K_u@alpha_c
    
    zx = (zx - np.mean(zx)) / (np.std(zx))
    zy = (zy - np.mean(zy)) / (np.std(zy))
    zc = (zc - np.mean(zc)) / (np.std(zc))

    
    sig_x_h = np.exp(params["ln_sig_x_h"])
    sig_zx_h = np.exp(params["ln_sig_zx_h"])
    sig_zy_h = np.exp(params["ln_sig_zy_h"])
    sig_zc_h = np.exp(params["ln_sig_zc_h"])
    sig_rx_h = np.exp(params["ln_sig_rx_h"])
    sig_ry_h = np.exp(params["ln_sig_ry_h"])
    


    sig_x_f = np.exp(params["ln_sig_x_f"])
    sig_zx_f = np.exp(params["ln_sig_zx_f"])
    sig_zy_f = np.exp(params["ln_sig_zy_f"])
    sig_zc_f = np.exp(params["ln_sig_zc_f"])
    
    
    sigs_f = np.hstack([sig_x_f, sig_zx_f, sig_zy_f, sig_zc_f])
    K_ax_f, K_ay_f, K_ax_h, K_ay_h, K_x_h, K_zx_h, K_zy_h, K_zc_h = kernel_mat_LNC(D_x, x, zx, zy, zc, sig_x_h, sig_zx_h, sig_zy_h, sig_zc_h, sigs_f) 
    n = K_ax_f.shape[0]

    ws = np.ones(n)
    #weights_y, resids_y, y_hat = krrModel(lam, K_ay_f, y, ws)
    weights_y, beta, resids_y, y_hat = krrModel_lin(lam, K_ay_f, x, y, ws)
    _, beta_u, _, _ = krrModel_lin(lam, K_u, x, y, ws)
    print("K_x.shape",K_x.shape)
    _, beta_x, _, _ = krrModel_lin(lam, K_x, x, y, ws)
    weights_x, resids_x, x_hat = krrModel(lam, K_ax_f, x, ws)
    
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
    mses = np.log(mse_rx) +  np.log(mse_ry)

    
    

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
        'mse_rx': mse_rx,
        'mse_ry': mse_ry,
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

@jax.jit
def getParamsForGrad_LNC(params, rep, smpl):

    
    
    ln_sig_x_h = params["ln_sig_x_h"][rep]
    ln_sig_zx_h = params["ln_sig_zx_h"][rep]
    ln_sig_zy_h = params["ln_sig_zy_h"][rep]
    ln_sig_zc_h = params["ln_sig_zc_h"][rep]
    ln_sig_rx_h = params["ln_sig_rx_h"][rep]
    ln_sig_ry_h = params["ln_sig_ry_h"][rep]
    


    ln_sig_x_f = params["ln_sig_x_f"][rep]
    ln_sig_zx_f = params["ln_sig_zx_f"][rep]
    ln_sig_zy_f = params["ln_sig_zy_f"][rep]
    ln_sig_zc_f = params["ln_sig_zc_f"][rep]
    
    
  
    params_aux = params.copy()

    alpha_x = params["alpha_x"][:, rep]
    alpha_y = params["alpha_y"][:, rep]
    alpha_c = params["alpha_c"][:, rep]
    alpha_x = alpha_x[:,None]
    alpha_y = alpha_y[:,None]
    alpha_c = alpha_c[:,None]
    
    
    params_aux['alpha_x'] = alpha_x
    params_aux['alpha_y'] = alpha_y
    params_aux['alpha_c'] = alpha_c
    
    params_aux["ln_sig_x_h"] = ln_sig_x_h
    params_aux["ln_sig_zx_h"] = ln_sig_zx_h
    params_aux["ln_sig_zy_h"] = ln_sig_zy_h
    params_aux["ln_sig_zc_h"] = ln_sig_zc_h
    params_aux["ln_sig_rx_h"] = ln_sig_rx_h
    params_aux["ln_sig_ry_h"] = ln_sig_ry_h
    

    params_aux["ln_sig_x_f"] = ln_sig_x_f
    params_aux["ln_sig_zx_f"] = ln_sig_zx_f
    params_aux["ln_sig_zy_f"] = ln_sig_zy_f
    params_aux["ln_sig_zc_f"] = ln_sig_zc_f


    return params_aux

def updateParams_LNC(params, grad_params, smpl, iteration, rep, learning_rate):
     
    #idx_rows = smpl[:, None]
    n = params['alpha_x'].shape[0]
    idx_rows = np.linspace(0, n - 1, n, dtype=int)[:, None]
    idx_cols = np.array(rep)[None, None]
    idx = jax.ops.index[tuple([idx_rows, idx_cols])]
    
    #alpha_x
    A = params['alpha_x'][tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_x']
    params['alpha_x'] = index_update(params['alpha_x'], idx, A - B)
    
    #alpha_y
    A = params['alpha_y'][tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_y']
    params['alpha_y'] = index_update(params['alpha_y'], idx, A - B)
    
    #alpha_c
    A = params['alpha_c'][tuple([idx_rows, idx_cols])]
    B = learning_rate * grad_params['alpha_c']
    params['alpha_c'] = index_update(params['alpha_c'], idx, A - B)
    
    gpars = [grad_params["ln_sig_x_h"],
    grad_params["ln_sig_zx_h"],
    grad_params["ln_sig_zy_h"],
    grad_params["ln_sig_zc_h"],
    grad_params["ln_sig_rx_h"],
    grad_params["ln_sig_ry_h"],
    grad_params["ln_sig_x_f"],
    grad_params["ln_sig_zx_f"],
    grad_params["ln_sig_zy_f"],
    grad_params["ln_sig_zc_f"],
    grad_params["ln_sig_x_h"],
    grad_params["ln_sig_zx_h"],
    grad_params["ln_sig_zy_h"],
    grad_params["ln_sig_zc_h"],
    grad_params["ln_sig_rx_h"],
    grad_params["ln_sig_ry_h"],
    grad_params["ln_sig_x_f"],
    grad_params["ln_sig_zx_f"],
    grad_params["ln_sig_zy_f"],
    grad_params["ln_sig_zc_f"]]
    

    if (onp.sum(onp.isnan(B))!=0) | (onp.sum(onp.isinf(B))!=0) | (onp.sum(onp.isnan(gpars))!=0):
        idx_nan, _ = onp.where(onp.isnan(B))
        print("nans in grad Z, iteration: ", iteration, " rep: ", rep)
        raise ValueError('Nans in gradient.')
            
    
    
    return None

def getIniMonitor_LNC(epochs, report_freq, reps, parts): 
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
            'hsic_x_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_y_zc': onp.zeros([num_reports, reps, parts]),
            'hsic_xy_zc': onp.zeros([num_reports, reps, parts]),
            'mse_rx': onp.zeros([num_reports, reps, parts]),
            'mse_ry': onp.zeros([num_reports, reps, parts]),
            'mses': onp.zeros([num_reports, reps, parts]),
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

def fillMonitor_LNC(params, params_or, pars, loss_as_par, dloss_as_par_jitted, lossData, iteration, report_freq, rep, part, reps, monitors, K_t):
    
    
    _, _, lam, _,_ = pars
          
    
    monitor = model_LNC(params, lam, lossData, K_t)
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
    monitors['mses'][indxRep, rep, part] = monitor['mses']
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
    
    
    
                    
                    
    return monitors


def getLatentZ_LNC(params, loss_as_par, dloss_as_par_jitted, loss_data, pars, epochs, report_freq, reps, batch_size, batch_per_epoch, learning_rate, smplsParts, monitors, batches, batches2, K_t):
    N = params["alpha_x"].shape[0]
         
    print("report_freq: ", report_freq)
    parts = len(smplsParts)
    
    ms = onp.zeros([N, epochs+1, reps])
    vs = onp.zeros([N, epochs+1, reps])
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

  
            loss_data_aux = getDataLoss_LNC(smpl, loss_data)
            loss_data_aux2 = getDataLoss_LNC(smpl2, loss_data)

  
            # equal weights
            ws = np.ones(batch_size)
            

            # algorithmic independence forcing

            # random weights - so that E[y|x] alg indep of p(x)
            #ws = getWeightsAlgoIndep(loss_data, batch_size, smpl)
            #print("ws: ", ws[0:5])

            
            # prepare parameters for grad calculation (subsample)
            
            #params_aux = getParamsForGrad(params, rep, optType, smpl)
            params_aux = getParamsForGrad_LNC(params, rep, smpl)
            params_aux2 = getParamsForGrad_LNC(params, rep, smpl2)
            #params_aux3 = getParamsForGrad(params, rep, smpl3)
           
            
             
            if adamOpt:
                grad_params = grad(loss_as_par, argnums=0)(params_aux, pars, loss_data_aux, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
                if iteration > 0:
                    m = ms[smpl,iteration-1,rep][:,None]
                    v = vs[smpl,iteration-1,rep][:,None]
                    m, v = updateParamsAdam2_LNC(params, grad_params, smpl, iteration, rep, lrs[rep], iteration-1, m, v)
                    ms[smpl,iteration,rep] = m[:,0]
                    vs[smpl,iteration,rep] = v[:,0]
            

            else:
                grad_params = dloss_as_par_jitted(params_aux, pars, loss_data_aux, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
                if iteration > 0:
                    updateParams_LNC(params, grad_params, smpl, iteration, rep, lrs[rep])

            ws = np.ones(batch_size*2)
            loss_vals[iteration, rep] = loss_as_par(params_aux2, pars, loss_data_aux2, ws, params_aux["alpha_x"], params_aux["alpha_y"], params_aux["alpha_c"])
            	

 
            if (iteration % report_freq == 0) & (iteration != 0):
                #print("report")
                print("iteration report: ", iteration)
                
                for part in range(parts):
                    
                    smplPart = smplsParts[part]
                    params_part = getParamsForGrad_LNC(params, rep, smplPart)
                    #params_part = getParamsSmpl(params, smplPart)
                    loss_data_part = getDataLoss_LNC(smplPart, loss_data)
                    monitors = fillMonitor_LNC(params_part, params, pars, loss_as_par, dloss_as_par_jitted, loss_data_part, iteration, report_freq, rep, part, reps, monitors, K_t)
                               

                    #nPerPart = smplPart.shape[0]
                    #ws = np.ones(nPerPart)
                    #loss_val = loss_as_par(params_part, pars, loss_data_part, ws)
         
                              
                  
   
    return params, monitors #, resids, bestResids, bestZ


def get_batches_one_epoch(batch_per_epoch, batch_size, n):
    #batches_one_epoch = onp.random.choice(batch_per_epoch, size=n) # equal expected number of points
    batches_one_epoch = onp.random.choice(n, size=n, replace=False)//batch_size # equal number of points
    #batches_one_epoch = onp.random.choice(n, size=n, replace=True)//batch_size # equal number of expected points
    batches_one_epoch = [myWhere(batches_one_epoch == i) for i in range(batch_per_epoch)]
    if len(batches_one_epoch[len(batches_one_epoch)-1]):
        comple = batch_size - len(batches_one_epoch[len(batches_one_epoch)-1])
        batches_one_epoch[len(batches_one_epoch)-1] = np.hstack([batches_one_epoch[len(batches_one_epoch)-1], np.array(onp.random.randint(low=0, high=n, size=comple))])
    return batches_one_epoch


def getLatentZ_wrapper(x, y, Z, U, idxs, stds, beta_real, nm, pars, num_epochs, report_freq, num_reps, batch_size, learning_rate, job):
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
    
    
    onp.random.seed(seed=job)
    
    maxMonitor = 1000
    parts = int(onp.ceil(N/maxMonitor))
    smplsParts = onp.random.choice(parts, size=N)
    

    smplsParts = [myWhere(smplsParts==i) for i in range(parts)]

    lossData = x, y, Z, K_u, D_x, K_y, D_y, K_x, idxs, beta_real, stds 
    
    

    onp.random.seed(seed=job+3)
    params = getIniPar_LNC(num_reps, lossData, pars, smplsParts)
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

    path = getIniMonitor_LNC(num_iters, report_freq, num_reps, parts)
    


    loss_as_par = loss_LNC
    dloss_as_par_jitted = dloss_LNC_jitted
    

        
        
    start = time.process_time()
    params, path = getLatentZ_LNC(params, loss_as_par, dloss_as_par_jitted, loss_data=lossData, pars=pars,epochs=num_iters, report_freq=report_freq,
                                                         reps=num_reps, batch_size=batch_size, batch_per_epoch=batch_per_epoch, learning_rate=learning_rate, smplsParts=smplsParts, monitors=path, batches=batches, batches2=batches2, K_t=None)
    res.append(path)
    
    res = {k: onp.concatenate([r[k] for r in res], axis=0) for k in res[0].keys()}
    

    print("results")
    res = {"params": params, "path": res}
    #print(res["path"])
    

    return res



