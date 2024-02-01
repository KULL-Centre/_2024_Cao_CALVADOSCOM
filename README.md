# CALVADOS 3

This repository contains codes, raw data and parameter files that used to reproduce the results in ${paper}.

## Layout
`data/`: raw data used for making figures in the paper; 

`paper_multidomainCALVADOS/`: figures (pdf format) in the paper; 

`src/`: all codes, starting conformations of multi-doamin proteins and experimental data used to reproduce results contained in the paper;

`src/BLOCKING/`: tools for assessing convergence of Molecular Dynamics simulations (https://github.com/fpesceKU/BLOCKING);

`src/expPREs/`: paramagnetic relaxation enhancement (PRE) NMR data for optimization;

`src/extract_relax`: Initial conformations of multi-domain proteins for simulations;

`src/domains.yaml`: Domain boundaries of used multi-domain proteins in this study;

`src/residues_pub.csv`: parameters file containing CALVADOS 1, CALVADOS 2 and CALVADOS 3;

`figures.ipynb`: jupyter version of plotting;



## Use
To use this repository, please download and unzip it. 
Then create a conda environment using the following command:

``conda env create -f environment.yaml``

and activate the environment:

```conda activate Calvados_test```

If you want to submit optimization jobs, please check the details in `src/submit_ray.py`;

If you want to submit slab simulation tasks, please check the details in `src/submit_slab.py`;