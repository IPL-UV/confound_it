#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# SBATCH --nodelist=nodo01,nodo02,nodo04,nodo07,nodo08,nodo09,nodo10,nodo11,nodo12
#SBATCH --exclude=nodo17,nodo13,nodo15,nodo20,nodo18,nodo19,nodo21,nodo22,nodo23,nodo24,nodo08
#SBATCH --workdir=/home/emiliano/discoveringLatentConfounders/confoundIt_py/src/
#SBATCH --job-name=LNC
#SBATCH --output=/home/emiliano/discoveringLatentConfounders/log/LNC_%A_%a.out
#SBATCH --array=1-2000%100

# initialize conda environment on ERC
module load Anaconda3
# activate environment for your script!
source activate /home/emiliano/miniconda3/envs/lnkrr_py38
#source activate lnkrr_py38

# DO STUFF
srun --ntasks 1 --nodes 1  python -u slurm_script_PC.py --save v0 --job $SLURM_ARRAY_TASK_ID --offset 0 --server erc
