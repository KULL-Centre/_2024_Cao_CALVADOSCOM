########################################################################################################################
#                                   import modules                                                                     #
########################################################################################################################
from utils_slab import *
import yaml
from jinja2 import Template
import subprocess
import pandas as pd
import os
########################################################################################################################
#                                    general simulation details                                                        #
########################################################################################################################
envname = "CALVADOS3"  # your own conda environment name
batch_sys = "ROBUST"  # your computing server
home_folder = "yourhomedir/_2024_Cao_CALVADOSCOM/src"  # absolute path to "submit_slab.py"
cwd_dict = {batch_sys: f"{home_folder}"}
CoarseGrained = "COM"  # COM, CA; CoarseGrained strategy;
k_restraint = 700  # unit:KJ/(mol*nm^2); prior default value: 700; force constant of elastic network model;
cycles = [0]
dataset_replicas = [1]  # used to discriminate from other runs;
cwd = cwd_dict[batch_sys]  # current working directory
initial_type = "C3"  # which forcefield you want to use, C3 means CALVADOS3
IDP = False  # simulate IDPs or MDPs
# don't need to change
slab = True
cutoff = 2.0  # cutoff for the nonionic interactions in slab simulations, default is 2 nm
compact_ini = True  # use the most compact conformation sampled from single-chain simulations
k_eq = 0.01  # unit:KJ/(mol*nm^2) for linear restraints towards box center in z
replicas = 1  # nums of replica for each sequence
gpu = True
eps_factor=0.2
gpu_id = 0
runtime = 20  # hours (overwrites steps if runtime is > 0)
simulation_time = 500  # ns
equi_time = 20  # ns
scale = int(simulation_time/equi_time)
nframes = 4000  # total number of frames to keep for each replica (exclude discarded frames)
N_steps = simulation_time*100000
N_save = int(N_steps/nframes)
########################################################################################################################
#                              multidomain simulation details                                                    #
########################################################################################################################
# kb = 8.31451E-3  # unit:KJ/(mol*K);
# protein and parameter list
Usecheckpoint = False
Threads = 18  # doesn't matter if gpu is used
group_list = ""
submit_job = Template("""#!/bin/bash
{{group_list}}
#SBATCH --job-name={{record}}
#SBATCH --partition=qgpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH -t 24:00:00
#SBATCH -o {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{name}}.out
#SBATCH -e {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{name}}.err

module purge
conda activate {{envname}}

cd {{cwd}}

{{equi_command}}
python3 -u {{cwd}}/simulate_slab.py --config {{fconfig}}
{{resubmit_command}}
""")
if batch_sys=="Computerome":
    group_list = "#PBS -W group_list=ku_10001 -A ku_10001"
    submit_job = Template("""#!/bin/bash
{{group_list}}
#PBS -N {{record}}
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l mem=130gb
#PBS -l walltime=48:00:00
#PBS -o {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{name}}.out
#PBS -e {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{name}}.err

module purge
conda activate {{envname}}

cd {{cwd}}

{{equi_command}}
python3 -u {{cwd}}/simulate_slab.py --config {{fconfig}}
{{resubmit_command}}
""")
########################################################################################################################
#                                             submit simulations                                                       #
########################################################################################################################
for dataset_replica in dataset_replicas:
    dataset = f"slab{initial_type}_{CoarseGrained}_cutoff{cutoff}_{dataset_replica}"
    records = []
    if not os.path.isfile(f"{cwd}/{dataset}/allproteins.pkl"):
        allproteins = pd.DataFrame(columns=['temp', 'expRg', 'expRgErr', 'Rg', 'rgarray', 'eff', 'chi2_rg', 'weights', 'pH', 'ionic', 'fasta'], dtype=object)
    else:
        allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl").astype(object)
    if IDP:
        df_csat = pd.read_csv(f"{cwd}/csat_calvados2_test.csv").set_index("Unnamed: 0")
    else:
        df_csat = pd.read_csv(f"{cwd}/csat_MDPs_test.csv").set_index("Unnamed: 0")
    for record in df_csat.index:
        prot = df_csat.loc[record]
        allproteins.loc[record] = dict(temp=prot.temp, expRg=None, expRgErr=None, pH=prot.pH, fasta=list(prot.fasta), ionic=prot.ionic)
        records.append(record)

    fdomains = f'{cwd}/{dataset}/domains.yaml'
    if not os.path.isdir(f"{cwd}/{dataset}"):
        os.system(f"mkdir -p {cwd}/{dataset}")
    os.system(f"cp {cwd}/domains.yaml {fdomains}")
    for record in records:
        chains = 100  # 100, 150, 200
        if record in ["hnRNPA1S"]:
            chains = 150
        os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")  # specify lambda initial values
        allproteins.to_pickle(f'{cwd}/{dataset}/allproteins.pkl')
        allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))
        # simulate
        for cycle in cycles:
            # create_parameters
            create_parameters(cwd, dataset, initial_type)
            if cycle == cycles[0]:
                dependency = ""
            else:
                dependency = f"#SBATCH --dependency=afterok:{jobid}" if batch_sys == 'Deic' else f"#PBS -W depend=afterok:{jobid}"
            prot = allproteins.loc[record]
            print(record)
            name = record.split("@")[0]
            if cycle == cycles[-1]:
                resubmit_command = ""
            else:
                resubmit_command = f"{'qsub' if batch_sys=='Computerome' else 'sbatch'} {cwd}/{dataset}/{record}/{cycle+1}/{name}_sim.pbs"
            N_res = prot.N
            if not IDP:
                if record in ["hnRNPA1S", "FL_FUS"]:
                    L = 20
                elif record in ["SNAP_FUS_PLDY2F_RBDR2K"]:
                    L = 29
                else:
                    L = 25
                isIDP = False
                path2fasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
                input_pae = ""  # decide_best_pae(cwd, name)
                path2pdb = f'{cwd}/extract_relax/{f"{name}_{CoarseGrained}_ini" if compact_ini else f"{name}_rank0_relax"}.pdb'  # af2 predicted structure
                use_pdb = True
                use_hnetwork = True
                use_ssdomains = True
                Lz = determineLz(cwd, dataset, record, name, cycle, 0, path2pdb, compact_ini, CoarseGrained, get_ssdomains(name, fdomains), chains, L, isIDP)
                print(Lz)
            else:
                if record=="Ddx4WT":
                    L = 17  # x and y, nm
                    Lz = 300  # y, nm
                else:
                    L = 15  # x and y, nm
                    Lz = 150  # y, nm
                isIDP = True
                path2fasta = ""  # no fasta needed if pdb provided
                input_pae = None
                path2pdb = ""  # af2 predicted structure
                use_pdb = False
                use_hnetwork = False
                use_ssdomains = False
            if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}"):
                os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}")
            for replica in range(replicas):
                equi_command = ""
                do_equi = False
                equi_dcd_readable = True
                try:
                    MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/equilibrium_{replica}.dcd")
                except Exception:
                    do_equi = True
                    equi_dcd_readable = False
                if equi_dcd_readable and len(MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/equilibrium_{replica}.dcd")) != int(nframes):
                    do_equi = True
                if cycle!=0:
                    do_equi = False
                if do_equi:
                    N_steps /= scale
                    N_save /= scale
                    config_equi_filename = f'config_equi.yaml'
                    config_equi_data = dict(cwd=cwd, name=name, dataset=dataset,
                        initial_type=initial_type, path2fasta=path2fasta, temp=float(prot.temp), pH=float(prot.pH), eps_factor=eps_factor,
                        ionic=float(prot.ionic), cycle=cycle, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab,
                        use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains, k_eq=k_eq,
                        use_ssdomains=use_ssdomains, replica=replica, input_pae=input_pae, k_restraint=k_restraint,
                        runtime=runtime, gpu_id=gpu_id, overwrite=True, N_res=int(N_res), Threads=Threads, chains=chains,
                        CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, seq=prot.fasta,
                        steps=int(N_steps), gpu=gpu, equilibrium=do_equi, record=record, Lz=Lz, compact_ini=compact_ini)
                    yaml.dump(config_equi_data, open(f'{cwd}/{dataset}/{record}/{cycle}/{config_equi_filename}', 'w'))
                    equi_command = f"python3 -u {cwd}/simulate_slab.py --config {cwd}/{dataset}/{record}/{cycle}/{config_equi_filename}"
                    N_steps *= scale
                    N_save *= scale
                    do_equi = False
                config_sim_filename = f'config_sim.yaml'
                config_sim_data = dict(cwd=cwd, name=name, dataset=dataset, initial_type=initial_type,
                                   path2fasta=path2fasta, temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab,
                                   use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains, k_eq=k_eq,
                                   use_ssdomains=use_ssdomains, replica=replica, input_pae=input_pae, k_restraint=k_restraint,
                                   runtime=runtime, gpu_id=gpu_id, overwrite=True, N_res=int(N_res), Threads=Threads, chains=chains,
                                   CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, Lz=Lz, pH=float(prot.pH), eps_factor=eps_factor,
                    seq=prot.fasta, steps=int(N_steps), gpu=gpu, equilibrium=do_equi, record=record, compact_ini=compact_ini)
                yaml.dump(config_sim_data, open(f'{cwd}/{dataset}/{record}/{cycle}/{config_sim_filename}', 'w'))
                # slab simulation needs more memory
                simrender_dict = dict(cwd=cwd, dataset=dataset, name=name, cycle=cycle, envname=envname,
                    fconfig=f"{cwd}/{dataset}/{record}/{cycle}/{config_sim_filename}", equi_command=equi_command,
                group_list=group_list, record=record, dependency=dependency, resubmit_command=resubmit_command)
                with open(f"{cwd}/{dataset}/{record}/{cycle}/{name}_sim.pbs", 'w') as submit:
                    submit.write(submit_job.render(simrender_dict))
                if cycle == cycles[0]:
                    proc = subprocess.run(["sbatch", f"{cwd}/{dataset}/{record}/{cycle}/{name}_sim.pbs"],capture_output=True)
                    print(proc)
                    jobid = int(proc.stdout.split(b' ')[-1].split(b'\\')[0])
