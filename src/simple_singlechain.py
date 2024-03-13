"""
This script is not designed for any HPC platform. It can be used for a quick single-chain simulation for your protein;
Just modify the basic settings in Block 2, you can get a coarse-grained trajectory with CALVADOS3
by running "python3 simple_singlechain.py";
Make sure this script is always in the src/ directory because it is using functions from other files in src/;
"""
########################################################################################################################
#                                   1. import modules                                                                  #
########################################################################################################################
from utils import *
from rawdata import *
import os
from protein_repo import get_ssdomains

########################################################################################################################
#                2. you only need to change parameters below to customize your protein simulation                      #
########################################################################################################################
cwd = "/projects/prism/people/ckv176/_2024_Cao_CALVADOSCOM/src"  # absolute path to "simple_singlechain.py", this script

# add IDPs or MDPs;
record = "GS48"  # protein name
temp = 293  # unit: K
pH = 7.5
ionic = 0.15  # unit: M
fasta = list("SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSWGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYFSDNVYITADKQKNGIKANFKIRHNIEDGGVQLADHYQQNTPIGDGPVLLPDNHYLSTQSKLSKDPNEKRDHMVLLEFVTAAGITLGMDELYKEGLSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSGSKLMVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDLQFSPWILVPHIGYGFHQYLPYPDGMSPFQAAMVDGSGYQVHRTMQFEDGASLTVNYRYTYEGSHIKGEAQVKGTGFPADGPVMTNSLTAADWCRSKKTYPNDKTIISTFKWSYTTGNGKRYRSTARTTYTFAKPMAANYLKNQPMYVFRKTELKHSKTELNFKEWQKAFTD")
replicas = 20  # nums of simulation replica for your protein

# is it a IDP or MDP?
isIDP = False
if not isIDP:
    domain_boundaries = {record: [[1,226], [352,566]]}  # define domain boundaries; for more information, please check step3 in "Run single-chain simulations with CALVADOS 3" from https://github.com/KULL-Centre/_2024_Cao_CALVADOSCOM/blob/main/README.md
    # the resSeq in 'path2pdb' file must start from 1 !!!!! the code won't check it itself.
    path2pdb = '/home/ckv176/_2024_Cao_CALVADOSCOM/src/extract_relax/GS48_rank0_relax.pdb'  # absolute path to atomistic structure

# customize your desired simulation time, unit: ns;
customized_simulation_time = 5
if customized_simulation_time == None:  # if None, script will use the default simulation duration depending on the IDR sequence length;
    # you don't have to change this block
    nframes = 200  # total number of frames to keep for each replica (exclude discarded frames)
    discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded for each replica
else:
    # you have to modify this block if you don't set 'customized_simulation_time' to 'None'
    interval = 0.01  # time interval to save each frame, unit: ns

########################################################################################################################
#          3. no need to change parameters below unless you know exactly what you are doing                            #
########################################################################################################################
initial_type = "C3"  # which forcefield to use; C3 means CALVADOS3;
CoarseGrained = "COM"  # COM: Center of mass; CA: Cα; CoarseGrained strategy;
k_restraint = 700  # unit:KJ/(mol*nm^2); prior default value: 700; force constant of elastic network model;
purpose = "CALVADOS3"  # CALVADOS3 means using CALVADOS3 to simulate proteins;
dataset_replica = 1
cutoff = 2.0  # 2.0 nm for production
cycle = 0  # optimization cycles, 0-based, only for optimization
gpu = False  # gpu acceleration
eps_factor = 0.2
gpu_id = 0
Usecheckpoint = False
slab = False

########################################################################################################################
#                                            4. submit simulations                                                     #
########################################################################################################################
dataset = f"{purpose}{CoarseGrained}_{cutoff}_{dataset_replica}"
if not os.path.isdir(f"{cwd}/{dataset}"):
    os.system(f"mkdir -p {cwd}/{dataset}")
os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")
create_parameters(cwd, dataset, cycle, initial_type)
# simulate
name = record.split("@")[0]
L = int(np.ceil((len(fasta) - 1) * 0.38 + 4))  # cubic box size setting, unit: nm
fdomains = f'{cwd}/{dataset}/domain_simple.yaml'
if not isIDP:
    yaml.dump(domain_boundaries, open(fdomains, 'w'))
    input_pae = ""  # decide_best_pae(cwd, name)
    use_pdb = True
    use_hnetwork = True
    use_ssdomains = True
    if customized_simulation_time == None:
        domain_len = 0
        for domain in get_ssdomains(name, fdomains, output=False):
            domain_len += len(domain)
        N_res = int(len(fasta) - domain_len)
        N_save = 3000 if N_res < 100 else int(np.ceil(3e-4 * N_res ** 2) * 1000)  # interval
        N_steps = (nframes + discard_first_nframes) * int(N_save)
    else:
        N_steps = int(customized_simulation_time * 1E5)
        total_frames = int(customized_simulation_time / interval)
        discard_first_nframes = int(total_frames * 0.05)
        nframes = int(total_frames - discard_first_nframes)
        N_save = int(interval * 1E5)
else:
    input_pae = None
    path2pdb = ""
    use_pdb = False
    use_hnetwork = False
    use_ssdomains = False
    if customized_simulation_time == None:
        N_save = 3000 if len(fasta) < 100 else int(np.ceil(3e-4 * len(fasta) ** 2) * 1000)  # interval
        N_steps = (nframes + discard_first_nframes) * int(N_save)
    else:
        N_steps = int(customized_simulation_time * 1E5)
        total_frames = int(customized_simulation_time / interval)
        discard_first_nframes = int(total_frames * 0.05)
        nframes = int(total_frames - discard_first_nframes)
        N_save = int(interval * 1E5)

if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}"):
    os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}")

replicas_list4MD = list(range(replicas))
config_sim_filename = f'config_sim.yaml'
config_sim_data = dict(cwd=cwd, name=name, dataset=dataset, temp=temp, ionic=ionic, cycle=cycle, pH=pH,
   replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab,
   use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains,
   use_ssdomains=use_ssdomains, input_pae=input_pae, k_restraint=k_restraint, record=record,
   gpu_id=gpu_id%4, Threads=1, overwrite=True, N_res=len(fasta),
   CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, eps_factor=eps_factor,
   initial_type=initial_type, seq=fasta, steps=N_steps, gpu=gpu, replicas=replicas,
   discard_first_nframes=discard_first_nframes,validate=False, nframes=nframes)

ray.init(num_cpus=len(replicas_list4MD), include_dashboard=False)
for replica in replicas_list4MD:
    config_sim_data["replica"] = replica
    yaml.dump(config_sim_data, open(f"{cwd}/{dataset}/{record}/{cycle}/config_{replica}.yaml", 'w'))
ray.get([simulate_simple.remote(yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_{replica}.yaml", 'r'))) for replica in replicas_list4MD])

config_merge_data = dict(cwd=cwd, name=name, dataset=dataset, temp=temp, ionic=ionic,
                         cycle=cycle, pH=pH, replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L,
                         wfreq=int(N_save), slab=slab, use_pdb=use_pdb, path2pdb=path2pdb,
                         use_hnetwork=use_hnetwork, fdomains=fdomains, use_ssdomains=use_ssdomains,
                         input_pae=input_pae, k_restraint=k_restraint, record=record, gpu_id=gpu_id % 4, Threads=1,
                         overwrite=True, N_res=len(fasta), CoarseGrained=CoarseGrained, isIDP=isIDP,
                         Usecheckpoint=Usecheckpoint, eps_factor=eps_factor, initial_type=initial_type, seq=fasta,
                         steps=N_steps, gpu=gpu, replicas=replicas, discard_first_nframes=discard_first_nframes,
                         validate=False, nframes=nframes)
centerDCD_simple(config_merge_data)
# you will get a trajectory ${record}.dcd and a corresponding topology file ${record}.pdb after performing codes above;
# the trajectory file ${record}.dcd contains all the trajectories from each replica;
# trajectory file ${record}.dcd can be used for post analysis, like calculating Rg (see below);

# calculate simulated Rg
df = load_parameters(cwd, dataset, cycle, initial_type).set_index("three")
t = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd",
                f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")  # nm
residues = [res.name for res in t.top.atoms]
masses = df.loc[residues,'MW'].values
masses[0] += 2
masses[-1] += 16
# calculate the center of mass
cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
# calculate residue-cm distances
si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
# calculate rg
rgarray = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
np.save(f"{cwd}/{dataset}/{record}/{cycle}/Rg_traj.npy", rgarray)