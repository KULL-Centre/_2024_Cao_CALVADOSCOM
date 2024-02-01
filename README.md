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

To use this repository:
1. download and unzip it; 
2. create a conda environment with the following command. (alternative: install packages separately by `pip` or `conda`);

    ``conda env create -f environment.yaml``

    then activate the environment:

    ``conda activate CALVADOS3``

4. install `pulchra` (https://www.pirx.com/pulchra/) and assign its absolute path to `path2pulchra` in `src/submit_ray.py`;

## Run single-chain simulations with CALVADOS 3:

Follow 1-4 if your protein is a multi-domain protein; just follow 1 and 4 if it is a intrinsically disordered protein;
1. decide a proper protein name (`pro_name`) to avoid conflict with existing proteins;
2. insert the protein structure file (must be `.pdb`, the `resSeq` should start from `1`) into `/src/extract_relax` directory with a nomenclature as `${pro_name}_rank0_relax.pdb`;
3. determine the domain boundaries of the protein. Each domain is restrained separately. For example, 
```
Ubq2 has two domains; one is from 11th to 82th residue; the other one is from 87th to 158th residue (counting from 1).
Ubq2:
  - [11,82]
  - [87,158]
```
```
S4FL has two domains; one is from 15th to 138th residue; 
the other one has 3 parts because we are excluding long loops between them. 3 parts are still restrained together.  
S4FL:
  - [15,138]
  - [[287,294],[323,466],[492,542]]
```
4. add two new lines in `rawdata.py` with formats of
```
fasta_${pro_name} = "${protein_sequence_oneletter}"
proteins.loc['${pro_name}'] = dict(temp=${experimental_temperate}, expRg=${experimental_Rg}, expRgErr=${experimental_RgErr}, pH=${experimental_pH}, fasta=list(fasta_${pro_name}), ionic=${experimental_ionic)}
```
The new lines should be in `initIDPsRgs` function under `if validate:` condition if it is a intrinsically disordered protein; 

The new lines should be in `initMultiDomainsRgs` function under `if validate:` condition if it is a multi-domain protein.
5. modify `/src/submit_ray.py` as you want and submit jobs by
```
python3 submit_ray.py
```
## Run multi-chain simulations with CALVADOS 3:
1. make sure the single-chain simulations of your interested proteins have been finished;
2. pick the most compact conformation from single-chain trajectories and save it under `src/extract_relax` as `${pro_name}_${CG}_ini.pdb`. `CG` is the coarse-grained method (CA, COM or SCCOM);
3. add experimental data to `src/csat_calvados2_test.csv` (if IDP) or `src/csat_MDPs_test.csv` (if MDP);
4. modify `/src/submit_slab.py` as you want and submit jobs by
```
python3 submit_slab.py
```
## Reproduce optimization results:
Please check the details in `src/submit_ray.py`;