# Recovering latent confounders from high-dimensional proxy variables

This repository contains the code to replicate the experiments and examples from 
_Recovering latent confounders from high-dimensional proxy variables_ 



## requirements

The scripts need R (for the simualtion experiments) and pyhton (for the real-world examples) installed. 

The following R packages need to be installed:
- ggplot2
- reshape2
- pbapply
- optparse
- fastICA
- mdatools

The following python libraries need to be available:

 - numpy
 - pandas
 - sklearn
 - jax 

## reproduce the simulation experiments 

To run the simulation experiments and produce the results we provide in the `simulations/` folder:

- `generate_data.R` a script which generate the synthetic data (run `Rscript generate_data.R --help` to list all the options).
- `run_methods.R` a script which generate the synthetic data (run `Rscript run_methods.R --help` to list all the options).
- `evaluate.R` a script which evaluate the results  (run `Rscript evaluate.R --help` to list all the options).
- `get_plots.R` a script to produce the final plots.  

The exact simualtion and experiments can be reproduced with the commands in the `experiments.sh` file.  


## reproduce GD-PCF method and experiments

To apply the GD-PCF on the simulation experiments and produce the results we provide the gd-pcf/folder:
- funcs_LNC_lin.py  functions that implement the GD-PCF method with linear assumptions
- funcs_LNC.py function that implement the GD-PCF method without assuming linearity.
- experiment.py script to apply GD-PCF on one dataset.
- slurm_script.py calls the experiment.py script for a dataset "job", parametrized by a slurm script that calls it
- processResults.py functions for gathering the results of GD-PCF for individual datasets which are stored as pickle files and obtaining performance measures.
- LNC_job.sh slurm script for running GD-PCF on different datasets in parallel 
