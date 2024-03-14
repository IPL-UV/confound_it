from typing import Dict
import numpy as onp
import pandas as pd
import sys
print(sys.version)
import jax.numpy as np
import json
from funcs_LNC import *
from funcs_LNC_lin import *
from processResults import *
import bisect
import pickle
import os.path

from experiment import main_experiment
# allows for arguments
import argparse

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)




def load_dataset(dataset_num: int = 0, server: str="myLap") -> (np.ndarray, str):
    # ======================
    # GET THE DATASET
    # ======================

    """
    1) take the job
        (dataset_num)

    2) match the number of the id to the dataset

    3) load the dataset to memory

    4) check it's numpy array

    4) return
    """
    print("enter load_dataset NEW")
    # Read in and prepare files
    if server == "erc":
        repos = "/home/emiliano/discoveringLatentConfounders/data/data_noisyproxy2/"
        


    if server == "myLap":
        repos = "/home/emiliano/causaLearner/data/"


    # declare parmeters

    
    pars = {"lambda": [0.01],
            "beta": [1.0],
            "neta": [1.0],
            "nu": [1.0],
            "lu": [0.0],
            "lr": [0.001],
            "epchs": [3000],
            "bs":[100],
            "reps":[1]}
    

    datasetTab = getDataSetTab(repos, pars)
    print("datasetTab shape: ", datasetTab.shape)
   
    job = dataset_num
    print(f"Starting job: {job}")


    indx_set = job - 1
    file = datasetTab["fileNames"][indx_set]
    lam = datasetTab["lambda"][indx_set]
    beta = datasetTab["beta"][indx_set]
    neta = datasetTab["neta"][indx_set]
    nu = datasetTab["nu"][indx_set]
    lu = datasetTab["lu"][indx_set]
    lr = datasetTab["lr"][indx_set]
    epchs = datasetTab["epchs"][indx_set]
    bs = datasetTab["bs"][indx_set]
    reps = datasetTab["reps"][indx_set]
    pars = {"lambda": lam, "beta": beta, "neta": neta, "nu":nu,"lu":lu, "lr": lr, "epchs":epchs, "bs":bs, "reps":reps}

    x, y, Z, U, meta, idxs, beta_real, stds = readFile(repos, file)
    
    nm = file.split(".")[0]
    

    dataInfo = {"dataset":nm}

    print("dataset: ", nm)

    
    return nm, x, y, Z, U, meta, idxs, beta_real, stds, pars, dataInfo  # load shit

def main(args):

    job = int(args.job) + int(args.offset)
    # load dataset from job array id
    print("load")
    nm, x, y, Z, U, meta, idxs, beta_real, stds, pars, dataInfo = load_dataset(dataset_num=job, server=args.server)
    print("nm: ", nm)
    print("meta: ", meta)
    print("U shape data", U.shape)
    print("pars: ", pars)
    

    N = x.shape[0]
    maxMonitor = 5000
    parts = int(onp.ceil(N/maxMonitor))
    print("parts: ", parts)
    


    # do stuffs (Latent Noise-KRR over the data)
    print("getLatenZs etc")
    beta = np.array([pars["beta"]])
    neta = np.array([pars["neta"]])
    lam = np.array([pars["lambda"]])
    nu = np.array([pars["nu"]])
    lu = np.array([pars["lu"]])
    lr = float(pars["lr"])
    epchs = int(pars["epchs"])
    bs = int(pars["bs"])
    reps = int(pars["reps"])

    
    #batch_size2 = int(onp.floor(onp.min([onp.max([30, bs/1000*N]), 300])))
    batch_size2 = int(onp.floor(onp.min([onp.max([25, bs/1000*N]), 300])))
    epochs2 = int(onp.max([onp.ceil((50*N)/batch_size2), 500]))
    print("epochs2: ", epochs2)
    print("batch_size2: ", batch_size2)


    # save shit (to json)
    print("save")

    if args.server == "erc":
        reposResults = "/home/emiliano/discoveringLatentConfounders/results/"

    if args.server == "myLap":
        reposResults = "/home/emiliano/ISP/proyectos/latentNoise_krr/results/"


    #save_shit(results, name=f"{args.save}_results_{args.job}.json")
    fileRes = reposResults+"LNC_"+str(job)+".pkl"
    #with open(fileRes, 'w') as outfile:
    #    json.dump(results, outfile)
    
    pars = {"lambda": lam,
            "beta": beta,
            "neta": neta,
            "nu": nu,
            "lu": lu,
            "lr": lr,
            "epchs": epchs,
            "bs":batch_size2,
            "reps":reps}


    if os.path.isfile(fileRes):
        print("File exist")
        results = pickle.load( open( fileRes, "rb" ) )
        results["pars"] = pars
        results["dataInfo"] = dataInfo
        results["meta"] = meta
        print(results)
    	# sample usage
        save_object(results, fileRes)
    else:
        print("File not exist")
        #results = main_experiment(x, y, Z, U, idxs, beta_real, stds, beta, neta, lam, nu, lu,lr, nm, epochs2, batch_size2, reps, job)
        results = main_experiment(x, y, Z, U, idxs, beta_real, stds, beta, neta, lam, nu, lu,lr, nm, epchs, batch_size2, reps, job)
        results["pars"] = pars
        results["dataInfo"] = dataInfo
        results["meta"] = meta
        #print(results)
    	# sample usage
        save_object(results, fileRes)
    
    return "bla"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments LNKRR.")

    # FOR THE JOB ARRRAY
    parser.add_argument("-j", "--job", default=0, type=int, help="job array for dataset")
    parser.add_argument("-o", "--offset", default=0, type=int, help="which job to begin after")
    parser.add_argument("-s", "--save", default="0", type=str, help="version string")
    parser.add_argument("-v", "--server", default="myLap", type=str, help="server to run in")
    # run experiment
    
    args = parser.parse_args()
    print(args)
    print("run experiment")
    results = main(args)
    print("finished")

