# CALVADOS-MDP

This repository contains codes, raw data and parameter files that used to reproduce the results in ${paper}.

## Layout
`BLOCKING/`: tools for assessing convergence of Molecular Dynamics simulations (https://github.com/fpesceKU/BLOCKING);

`expPREs`: paramagnetic relaxation enhancement (PRE) NMR data for optimization;

`extract_relax`: Initial conformations of multi domain proteins for simulations;

`domains.yaml`: Domain boundaries of used multi domain proteins in this study;

## Use
To use this repository, please download and unzip it. 
Then create a conda environment using the following command:

``conda env create -f environment.yaml``

and activate the environment:

```conda activate Calvados_test```

If you want to submit optimization jobs, please check the details in `submit_ray.py`;

If you want to submit slab simulation tasks, please check the details in `submit_slab.py`;