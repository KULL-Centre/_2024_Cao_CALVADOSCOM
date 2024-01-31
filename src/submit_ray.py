########################################################################################################################
#                                   import modules                                                                     #
########################################################################################################################
from utils import *
import subprocess
from rawdata import *
import os
import MDAnalysis
from protein_repo import get_ssdomains
########################################################################################################################
# general simulation details (please read this block carefully if you want to run optimizations) #
########################################################################################################################
env_name = "CALVADOSCOM"  # your own conda environment name
batch_sys = "Deic"  # your computing server
home_folder = "yourhomedir/_2024_Cao_CALVADOSCOM/src"  # absolute path to "submit_ray.py" 
cwd_dict = {batch_sys: f"{home_folder}/{env_name}"}
initial_types_dict = {
"CALVADOS2CA": "C2",
"CALVADOS2COM": "C2",
"CALVADOS2SCCOM": "C2",
"IDPs_MDPsCOM": "0.5",
"IDPs_MDPsSCCOM": "0.5"
}  # which forcefield to use; C2 means CALVADOS2; 0.5 means optimization procedure starting from 0.5
cpu_max = 32  # the maximum of requested number of cpu: let's simulate 20 replicas for each sequence; then 4 cpus for each replica; so 32 cpus could handel 32/4=8 replicas;
CoarseGrained = "COM"  # COM, CA, SCCOM; CoarseGrained strategy;
validate = False  # optimization or validation
thetas = [0.08]  # prior default value: 0.08
k_restraint = 700  # unit:KJ/(mol*nm^2); prior default value: 700; force constant of elastic network model;
cycles = [0]  # optimization cycles, 0-based
dataset_replicas = [1]  # used to do parallel optimization runs;
cutoffs = [2.2]  # cutoff for the nonionic interactions, nm
purpose = "IDPs_MDPs"  # IDPs_MDPs means optimization including IDPs and MDPs, CALVADOS2 means using CALVADOS2 to simulate proteins;


rebalancechi2_rg= False
include_PREloss = True
lambda_oneMax = False  # the limitation that lambda should not exceed 1.0
replicas = 20  # nums of replica for each sequence
gpu = False
eps_factor=0.2
gpu_id = 0
discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded when merging replicas
nframes = 200  # total number of frames to keep for each replica (exclude discarded frames)
slab = False  # slab simulation parameters
Usecheckpoint = False
path2pulchra = "/groups/sbinlab/fancao/pulchra"
########################################################################################################################
#                                             submit simulations                                                       #
########################################################################################################################

# please replace keywords of `submission_1`, `submission_2` and `submission_3` with your own desired configurations;
validate_str = "_validate" if validate else ""
for dataset_replica in dataset_replicas:
    for theta in thetas:
        for cutoff in cutoffs:
            dataset = f"{purpose}{CoarseGrained}"
            cwd = cwd_dict[batch_sys]  # current working directory
            initial_type = initial_types_dict[dataset]
            dataset = f"{dataset}_{cutoff}_{theta}_{dataset_replica}{validate_str}"
            if validate:
                cutoff = 2.0  # 2.0 nm for validation
            fdomains = f'{cwd}/{dataset}/domains.yaml'
            if not os.path.isdir(f"{cwd}/{dataset}"):
                os.system(f"mkdir -p {cwd}/{dataset}")
            if validate:  # you need first run validate=False and then validate=True to make sure those files can be copied;
                os.system(f"cp {cwd}/{dataset[:-len(validate_str)]}/MultiDomainsRgs.pkl {cwd}/{dataset}")
                os.system(f"cp {cwd}/{dataset[:-len(validate_str)]}/allproteins.pkl {cwd}/{dataset}")
                os.system(f"cp {cwd}/{dataset[:-len(validate_str)]}/proteinsPRE.pkl {cwd}/{dataset}")
            os.system(f"cp {cwd}/domains.yaml {fdomains}")
            for cycle in cycles:
                if cycle == 0:
                    if initial_type=="C1" or initial_type=="C2":
                        os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")  # specify lambda initial values
                    if initial_type == "0.5" or initial_type == "Ran":
                        os.system(f"cp {cwd}/residues_-1.csv {cwd}/{dataset}")  # use 0.5 or random
                    create_parameters(cwd, dataset, cycle, initial_type)
                elif validate:
                    if initial_type=="C1" or initial_type=="C2":
                        os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")  # specify lambda initial values
                    if initial_type == "0.5" or initial_type == "Ran":
                        os.system(f"cp {cwd}/{dataset[:-len(validate_str)]}/residues_{cycle-1}.csv {cwd}/{dataset}")  # use 0.5 or random
                proteinsPRE = initProteinsPRE()
                IDPsRgs = initIDPsRgs(validate=validate)
                IDPsRgs_names = list(IDPsRgs.index)
                MultiDomainsRgs = initMultiDomainsRgs(validate=validate)
                multidomain_names = list(MultiDomainsRgs.index)
                proteinsRgs = pd.concat((IDPsRgs, MultiDomainsRgs), sort=True)
                IDPsRgs = proteinsRgs.loc[IDPsRgs_names]
                MultiDomainsRgs = proteinsRgs.loc[multidomain_names]
                allproteins = pd.concat((proteinsPRE, IDPsRgs, MultiDomainsRgs), sort=True)
                if not validate:
                    proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
                    IDPsRgs.to_pickle(f'{cwd}/{dataset}/IDPsRgs.pkl')
                    MultiDomainsRgs.to_pickle(f'{cwd}/{dataset}/MultiDomainsRgs.pkl')
                    allproteins.to_pickle(f'{cwd}/{dataset}/allproteins.pkl')
                else:
                    proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE_test.pkl')
                    IDPsRgs.to_pickle(f'{cwd}/{dataset}/IDPsRgs_test.pkl')
                    MultiDomainsRgs.to_pickle(f'{cwd}/{dataset}/MultiDomainsRgs_test.pkl')
                    allproteins.to_pickle(f'{cwd}/{dataset}/allproteins_test.pkl')
                allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))
                N_res = []
                for record in allproteins.index:
                    name = record.split("@")[0]
                    if record in multidomain_names:
                        domain_len = 0
                        for domain in get_ssdomains(name, fdomains, output=False):
                            domain_len += len(domain)
                        N_res.append(allproteins.loc[record].N - domain_len)
                    else:
                        N_res.append(allproteins.loc[record].N)
                allproteins["N_res"] = N_res
                allproteins = allproteins.sort_values('N_res', ascending=False)
                # simulate
                jobid_2 = []
                name_idx = 0
                for record, prot in allproteins.iterrows():
                    jobid_1 = []
                    name = record.split("@")[0]
                    do_merge = False
                    name_dcd_readable = True
                    try:
                        MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd")
                    except Exception:
                        do_merge = True
                        name_dcd_readable = False
                    if name_dcd_readable and len(MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd"))!=int(replicas * nframes):
                        do_merge = True
                    if do_merge:
                        print(record)
                        N_res = prot.N
                        L = int(np.ceil((N_res - 1) * 0.38 + 4))
                        if record in multidomain_names:
                            isIDP = False
                            path2fasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
                            input_pae = ""  # decide_best_pae(cwd, name)
                            path2pdb = f'{cwd}/extract_relax/{name}_rank0_relax.pdb'  # af2 predicted structure
                            use_pdb = True
                            use_hnetwork = True
                            use_ssdomains = True
                            domain_len = 0
                            for domain in get_ssdomains(name, fdomains, output=False):
                                domain_len += len(domain)
                            N_res = int(allproteins.loc[record].N - domain_len)
                            N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
                            cost = (allproteins.loc[record].N ** 2 + domain_len * 15) * N_save / 1000
                        else:
                            isIDP = True
                            path2fasta = ""  # no fasta needed if pdb provided
                            input_pae = None
                            path2pdb = ""  # af2 predicted structure
                            use_pdb = False
                            use_hnetwork = False
                            use_ssdomains = False
                            N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
                            cost = allproteins.loc[record].N ** 2 * N_save / 1000
                        if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}"):
                            os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}")

                        N_steps = (nframes + discard_first_nframes) * int(N_save)
                        Threads, node = determineThreadsnode(batch_sys, cost)  # Threads for each replica
                        collected_replicas = []
                        replicas_list4MD = []
                        for replica in range(replicas):
                            dcd_readable = True
                            do_MD = False
                            try:
                                MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{replica}.dcd")
                            except Exception:
                                dcd_readable = False
                                do_MD = True
                            if dcd_readable and len(MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{replica}.dcd")) != int(nframes + discard_first_nframes):
                                do_MD = True
                            if do_MD:
                                if len(replicas_list4MD)+1 <= cpu_max//Threads:
                                    replicas_list4MD.append(replica)
                                else:
                                    collected_replicas.append(replicas_list4MD)
                                    replicas_list4MD = []
                                    replicas_list4MD.append(replica)
                        collected_replicas.append(replicas_list4MD)
                        for replicas_list4MD_idx, replicas_list4MD in enumerate(collected_replicas):
                            if len(replicas_list4MD)!=0:
                                config_sim_filename = f'config_sim{replicas_list4MD_idx}.yaml'
                                config_sim_data = dict(cwd=cwd, name=name, dataset=dataset,
                                   path2fasta=path2fasta, temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, pH=float(prot.pH),
                                   replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab,
                                   use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains,
                                   use_ssdomains=use_ssdomains, input_pae=input_pae, k_restraint=k_restraint, record=record,
                                   gpu_id=gpu_id%4, Threads=Threads, overwrite=True, N_res=N_res,
                                   CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, eps_factor=eps_factor,
                                   initial_type=initial_type, seq=prot.fasta, steps=N_steps, gpu=gpu, replicas=replicas,
                                   discard_first_nframes=discard_first_nframes,validate=validate, nframes=nframes)
                                write_config(cwd, dataset, record, cycle, config_sim_data, config_filename=config_sim_filename)
                                if len(cycles)!=1 and cycle!=cycles[0]:
                                    sim_dependency = f"#SBATCH --dependency=afterok:{optimize_jobid}" if batch_sys == 'Deic' else f"#PBS -W depend=afterok:{optimize_jobid}"
                                else:
                                    sim_dependency = ""
                                requested_resource = f"1:ppn={len(replicas_list4MD)*Threads}"
                                simrender_dict = dict(cwd=cwd, dataset=dataset, record=record, cycle=f'{cycle}',
                                    requested_resource=requested_resource, node=node, sim_dependency=sim_dependency,
                                    config_sim_filename=config_sim_filename, Threads=Threads, mem=f"{len(replicas_list4MD)*2}",
                                requested_cpunum=len(replicas_list4MD)*Threads, replicas_list4MD_idx=replicas_list4MD_idx,
                                )

                                with open(f"{cwd}/{dataset}/{record}/{cycle}/{record}_sim{replicas_list4MD_idx}.sh", 'w') as submit:
                                    submit.write(submission_1.render(simrender_dict))
                                proc = subprocess.run(['sbatch', f"{cwd}/{dataset}/{record}/{cycle}/{record}_sim{replicas_list4MD_idx}.sh"],capture_output=True)
                                # os.system("sleep 0.5")
                                print(proc)
                                jobid_1.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
                        # merge
                        if len(jobid_1) != 0:
                            jobs = ""
                            for job in jobid_1:
                                jobs += f":{job}"
                            merge_dependency = f"#SBATCH --dependency=afterok{jobs}" if batch_sys == 'Deic' else f"#PBS -W depend=afterok{jobs}"
                        else:
                            merge_dependency = ""
                        config_mer_filename = f'config_merge.yaml'
                        # copy of config_sim_data
                        config_merge_data = dict(cwd=cwd, name=name, dataset=dataset, path2fasta=path2fasta, temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, pH=float(prot.pH), replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab, use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains, use_ssdomains=use_ssdomains, input_pae=input_pae, k_restraint=k_restraint, record=record, gpu_id=gpu_id % 4, Threads=Threads, overwrite=True, N_res=N_res, CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, eps_factor=eps_factor, initial_type=initial_type, seq=prot.fasta, steps=N_steps, gpu=gpu, replicas=replicas, discard_first_nframes=discard_first_nframes, validate=validate, nframes=nframes)
                        write_config(cwd, dataset, record, cycle, config_merge_data, config_filename=config_mer_filename)
                        mergerender_dict = dict(cwd=cwd, dataset=dataset, record=record, cycle=f'{cycle}',
                            requested_resource="1:ppn=1", node=node, merge_dependency=merge_dependency,
                            config_mer_filename=config_mer_filename, mem=f"{Threads*2}")
                        with open(f"{cwd}/{dataset}/{record}/{cycle}/{record}_merge.sh", 'w') as submit:
                            submit.write(submission_2.render(mergerender_dict))
                        proc = subprocess.run(["sbatch", f"{cwd}/{dataset}/{record}/{cycle}/{record}_merge.sh"], capture_output=True)
                        print(proc)
                        jobid_2.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
                    print(f'Simulating {record}: {jobid_1}')

                if not validate:
                    # optimize
                    if len(jobid_2) != 0:
                        jobs = ""
                        for job in jobid_2:
                            jobs += f":{job}"
                        opt_dependency = f"#SBATCH --dependency=afterok{jobs}" if batch_sys == 'Deic' else f"#PBS -W depend=afterok{jobs}"
                    else:
                        opt_dependency = ""
                    config_filename = f'config_opt{cycle}_{theta}.yaml'
                    config_data = dict(cwd=cwd, log_path="LOG", dataset=dataset, cycle=cycle, num_cpus=10 if batch_sys=='Deic' else 38, cutoff=cutoff,
                                       theta=theta, fdomains=fdomains, initial_type=initial_type, lambda_oneMax=lambda_oneMax, include_PREloss=include_PREloss,
                    rebalancechi2_rg=rebalancechi2_rg)
                    yaml.dump(config_data, open(f'{cwd}/{dataset}/{config_filename}','w'))
                    optrender_dict = dict(cwd=cwd, dataset=dataset, opt_dependency=opt_dependency,proteins=' '.join(proteinsPRE.index),
                                          cycle=f'{cycle}', path2config=f"{cwd}/{dataset}/{config_filename}", path2pulchra=path2pulchra,
                    )
                    with open(f"{cwd}/{dataset}/opt_{cycle}.sh", 'w') as submit:
                        submit.write(submission_3.render(optrender_dict))
                    proc = subprocess.run(["sbatch",f"{cwd}/{dataset}/opt_{cycle}.sh"],capture_output=True)
                    print(proc)
                    optimize_jobid = int(proc.stdout.split(b' ')[-1].split(b'\\')[0])
