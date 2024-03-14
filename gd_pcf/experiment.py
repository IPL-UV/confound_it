from typing import Dict
import numpy as onp
import jax.numpy as np
from funcs_LNC import *
from funcs_LNC_lin import *
from processResults import *
import json
import sys
import os
# allows for arguments
#import argparse




def main_experiment(x: np.ndarray, y: np.ndarray, Z: np.ndarray, U: np.ndarray, idxs: tuple, beta_real:np.ndarray, stds:np.ndarray,  beta: np.ndarray, neta: np.ndarray, lam: np.ndarray, nu: np.ndarray, lu: np.ndarray,lr: float, name: str, epchs:int, bs:int, reps:int, job:int) -> Dict:
    


    
    pars = (beta, neta,lam,nu, lu)

    start = time.process_time()
    res = getLatentZ_wrapper_lin(x, y, Z, U, idxs, stds, beta_real, name, pars, epchs, epchs, reps, bs, lr, job)
    print(time.process_time() - start)  #

    


    res = {"Z": res}

    return res



if __name__ == "__main__":
    # load dataset
    # dataset = ...

    server = str(sys.argv[1])
    print("server: ",server)
    if server == "erc":
        print("erc")
        folder = "/home/emiliano/discoveringLatentConfounders/data/data_to_try/13:45:44_14092023/"

    if server == "myLap":
        print("myLap")
        folder = "/home/emiliano/Documents/ISP/postdoc/discoveringLatentConfounders/data/data_noisyproxy/"
    
    
    #myfile = "rep1_gaussian1exponential1_independent_20_1_50_50.csv"

    #myfile = "5_exponential1gaussian1_dependent_10_1_100.csv"
     
    #beta = np.array(1.0) # mse
    #neta = np.array(1.0) # indep zs
    #lam = np.array(0.01) # krr
    #nu = np.array(1.0) # z norm
    #lu = np.array(1.0) # CI indep
    
    #epchs = 1000
    #reps = 1
    #bs = 100
    #lr = 0.001
    #job = 1  
    #nm = "test"
    
   

    pars = {"lambda": [0.01],
            "beta": [1.0],
            "neta": [1.0],
            "nu": [1.0],
            "lu": [1.0],
            "lr": [0.001],
            "epchs": [1000],
            "bs":[100],
            "reps":[1]}

    datasetTab = getDataSetTab(folder, pars)
    print("datasetTab shape: ", datasetTab.shape)
    job = 1
    print(f"Starting job: {job}")
    indx_set = job
    myfile = datasetTab["fileNames"][indx_set]
    lam = datasetTab["lambda"][indx_set]
    beta = datasetTab["beta"][indx_set]
    neta = datasetTab["neta"][indx_set]
    nu = datasetTab["nu"][indx_set]
    lu = datasetTab["lu"][indx_set]
    lr = datasetTab["lr"][indx_set]
    epchs = datasetTab["epchs"][indx_set]
    bs = datasetTab["bs"][indx_set]
    reps = datasetTab["reps"][indx_set]

    x, y, Z, U, meta, idxs, beta_real, stds = readFile(folder, myfile)

    nm = myfile.split(".")[0]

    
    # run experiment
    results = main_experiment(x, y, Z, U, idxs, beta_real, stds, beta, neta, lam, nu, lu,lr, nm, epchs, bs, reps, job)
    print(results["Z"])
    print("finished")
    # save to somewhere
    # save_shit
