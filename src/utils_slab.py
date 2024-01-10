from matplotlib.colors import LogNorm
import sys
import os
sys.path.append('./BLOCKING')
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from main import BlockAnalysis
from MDAnalysis import transformations
import numpy as np
import statsmodels.regression.linear_model as sm
import time
# from DEERPREdict.PRE import PREpredict
# from pulchra import fix_topology
from collections import defaultdict
import MDAnalysis
from jinja2 import Template
from protein_repo import *
import yaml
from simtk import openmm, unit
import mdtraj as md
import itertools
# from rawdata import calcChi2, reweightRg
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
element_mass_dict = {"C": 12.011, "N": 14.007, "O": 15.999, "H": 1.0080, "S": 32.06}
aa = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
               "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
               "HID", "ASH", "HIE", "HIP", "HSD", "HSE", "HSP"]  # irregular aa

sasa_max = {
"GLY": 86.2,
"ALA": 109.3,
"ARG": 255.5,  # +1 net charge
"ASN": 171.5,
"ASP": 170.2,  # COO-, because in CALVADOS2, it has -1 net charge
"CYS": 140.1,
"GLN": 189.2,
"GLU": 198.6,  # COO-, because in CALVADOS2, it has -1 net charge
"HIS": 193.8,  # not charged
"ILE": 173.3,
"LEU": 181.6,
"LYS": 215.0,  # +1 net charge
"MET": 193.0,
"PHE": 205.8,
"PRO": 134.3,
"SER": 132.4,
"THR": 148.8,
"TRP": 253.1,
"TYR": 230.5,
"VAL": 148.4,
}  # Angstroms^2, extended GGXGG (G, glycine; X any residue)

#SBATCH --exclusive
submission_1 = Template(
"""#!/bin/bash
#SBATCH --job-name={{name}}_sim{{cycle}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={{requested_resource}}
#SBATCH --partition={{node}}
{{sim_dependency}}
#SBATCH --mem={{mem}}GB
#SBATCH -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_out_sim
#SBATCH -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_err_sim

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

python3 {{cwd}}/sim_replicas.py --config_sim {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{config_sim_filename}}""")

submission_2 = Template(
"""#!/bin/bash
#SBATCH --job-name=merge_{{cycle}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=sbinlab_gpu
{{mer_dependency}}
#SBATCH --mem=10GB
#SBATCH -o {{cwd}}/{{dataset}}/{{cycle}}_merge_out
#SBATCH -e {{cwd}}/{{dataset}}/{{cycle}}_merge_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

python3 {{cwd}}/merge_replicas_basic.py --config {{cwd}}/{{dataset}}/{{config_mer_filename}}""")


submission_3 = Template(
"""#!/bin/bash
#SBATCH --job-name=opt_{{cycle}}
#SBATCH --nodes=1
#SBATCH --partition=sbinlab
#SBATCH --mem=487GB
#SBATCH --cpus-per-task=62
{{opt_dependency}}
#SBATCH -o {{cwd}}/{{dataset}}/{{cycle}}_out
#SBATCH -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /groups/sbinlab/fancao/.bashrc

conda activate calvados

declare -a proteinsPRE_list=({{proteins}})
# python3 {{cwd}}/utils.py

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python3 {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --cycle {{cycle}} --num_cpus 62 --pulchra /groups/sbinlab/fancao/pulchra
done

python3 {{cwd}}/optimize.py  --path2config {{path2config}}""")

submission_4 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_sim{{cycle}}
#PBS -l nodes={{requested_resource}}:{{node}}
#PBS -l walltime=48:00:00
{{sim_dependency}}
#PBS -l mem={{mem}}gb
#PBS -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_out_sim
#PBS -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_err_sim

source /home/people/fancao/.bashrc
conda activate calvados

python3 {{cwd}}/sim_replicas.py --config_sim {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{config_sim_filename}}""")

submission_5 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_sim{{cycle}}
#PBS -l nodes={{requested_resource}}:{{node}}
#PBS -l walltime=48:00:00
{{sim_dependency}}
#PBS -l mem={{mem}}gb
#PBS -o {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_out_sim
#PBS -e {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{cycle}}_err_sim


sleep {{name_idx}}
# source /home/people/fancao/.bashrc
conda activate calvados

module purge
module load tools
module load cuda/toolkit/9.2.148
module load gcc/12.2.0
module load gromacs/2020.1

thishost=`uname -n | awk -F. '{print $1.}'`
echo ${thishost}>{{cwd}}/{{dataset}}/{{name}}/{{cycle}}/thishost.txt
# ip_prefix=`hostname -i`  # wrong
redis_password=$(uuidgen)
echo ${redis_password}
ray start --head --redis-password=$redis_password --num-cpus {{Threads}} --port=0 > {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/port.txt
res=$(cat {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/port.txt)
leftstring="start --address='"
rightstring="' --redis-password"
res=${res#*${leftstring}}
res=${res%${rightstring}*}
ip_prefix=${res%:*}
port=${res#*:}

suffix=:${port}
ip_head=$ip_prefix$suffix
jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`
echo ${jobnodes}>{{cwd}}/{{dataset}}/{{name}}/{{cycle}}/jobnodes.txt

# shellcheck disable=SC2206
declare -a nodeslist=(${jobnodes})
# shellcheck disable=SC2068
# shellcheck disable=SC2034
for node in ${nodeslist[@]:1}
do
pbsdsh -h ${node} -v {{cwd}}/startWorkerNode.sh $ip_head $redis_password {{Threads}} &
sleep 10
done

echo "Initializing worker nodes complete"
# make sure all workers are connected
# even though WorkerNodes do not complete initialization on time,
# as long as they can be initialized in the end, the pooled jobs will be assigned to those nodes
sleep 10   

cd {{cwd}} || exit
python3 -u {{cwd}}/simulate_saxs.py --ip_head $ip_head --pw $redis_password --config {{cwd}}/{{dataset}}/{{name}}/{{cycle}}/{{config_simerge_filename}}""")

submission_6 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N merge_{{cycle}}
#PBS -l nodes=1:ppn=1:thinnode
#PBS -l walltime=2:00:00
#PBS -l mem=10gb
{{mer_dependency}}
#PBS -o {{cwd}}/{{dataset}}/{{cycle}}_merge_out
#PBS -e {{cwd}}/{{dataset}}/{{cycle}}_merge_err

source /home/people/fancao/.bashrc

conda activate calvados

python3 {{cwd}}/merge_replicas_basic.py --config {{cwd}}/{{dataset}}/{{config_mer_filename}}""")

submission_7 = Template(
"""#!/bin/bash
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N opt_{{cycle}}
#PBS -l nodes=1:ppn=38:thinnode
#PBS -l walltime=45:00:00
#PBS -l mem=170gb
{{opt_dependency}}
#PBS -o {{cwd}}/{{dataset}}/{{cycle}}_out
#PBS -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /home/people/fancao/.bashrc

conda activate calvados

declare -a proteinsPRE_list=({{proteins}})

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python3 {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --cycle {{cycle}} --num_cpus 38 --pulchra /home/projects/ku_10001/people/fancao/pulchra
done

python3 {{cwd}}/optimize_saxs.py  --path2config {{path2config}}""")


def euclidean(a_matrix, b_matrix):
    # Using matrix operation to calculate Euclidean distance
    """                     b
       [[2.23606798  1.          5.47722558]
    a   [7.07106781  4.69041576  3.        ]
        [12.20655562 9.8488578   6.4807407 ]]"""
    d1 = -2 * np.dot(a_matrix, b_matrix.T)
    d2 = np.sum(np.square(a_matrix), axis=1, keepdims=True)
    d3 = np.sum(np.square(b_matrix), axis=1)
    dist = np.sqrt(d1 + d2 + d3)
    return dist

def nested_dictlist():
    return defaultdict(list)

def processIon(aa) -> str:  # Dealing with protonation conditions
    if aa in ['ASH']:
        return 'ASP'
    if aa in ['HIE', 'HID', 'HIP', 'HSD', 'HSE', 'HSP']:
        return 'HIS'
    return aa

def space_data(li, interval=20):  # interval
    y = []
    for i in range(len(li)):
        if i % interval == 0:
            y.append(li[i])
    return y

def write_config(cwd, dataset, name, cycle,config_data,config_filename):
    with open(f'{cwd}/{dataset}/{name}/{cycle}/{config_filename}','w') as stream:
        yaml.dump(config_data,stream)

def set_harmonic_network(N,dmap,pae_inv,yu,ah,n_chains=1,ssdomains=None,cs_cutoff=0.9,k_restraint=700.):
    cs = openmm.openmm.HarmonicBondForce()
    for n_chain in range(n_chains):
        begin = n_chain * N  # 0
        end = n_chain * N + N  # n_residues
        for i in range(begin, end-2):  # 0-based
            for j in range(i+2,end):  # 0-based
                residue_i = i - begin
                residue_j = j - begin
                if ssdomains != None:  # use fixed domain boundaries for network
                    ss = False
                    for ssdom in ssdomains:
                        if residue_i+1 in ssdom and residue_j+1 in ssdom:
                            ss = True
                    if ss:  # both residues in structured domains
                        if dmap[residue_i,residue_j] < cs_cutoff:  # nm
                            cs.addBond(i, j, dmap[residue_i, residue_j] * unit.nanometer,
                                       k_restraint * unit.kilojoules_per_mole / (unit.nanometer ** 2))
                            yu.addExclusion(i, j)
                            ah.addExclusion(i, j)
                elif isinstance(pae_inv, np.ndarray):  # use alphafold PAE matrix for network
                    k = k_restraint * pae_inv[residue_i,residue_j]**2
                    if k > 0.0:
                        cs.addBond(i,j, dmap[residue_i,residue_j]*unit.nanometer,
                                    k*unit.kilojoules_per_mole/(unit.nanometer**2))
                        yu.addExclusion(i, j)
                        ah.addExclusion(i, j)
                else:
                    raise

    cs.setUsesPeriodicBoundaryConditions(True)
    return cs, yu, ah

def set_interactions(system, residues, prot, calvados_version, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N, n_chains=1,
                     CoarseGrained="CA", dismatrix=None, isIDP=True, fdomains=None):
    print("dismatrix.shape", dismatrix.shape)
    hb = openmm.openmm.HarmonicBondForce()
    # interactions
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    if calvados_version in [1, 2]:
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    elif calvados_version == 3:  # interactions scaled aromatics + R + H
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=sqrt(l1*l2)+m1*m2*0.8; shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    elif calvados_version == 4:  # scaled charges
        ah = openmm.openmm.CustomNonbondedForce(
            energy_expression + '; s=0.5*(s1+s2); l=sqrt(l1*l2)+m1*m2*0.5; shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    else:
        raise

    ah.addGlobalParameter('eps', lj_eps * unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc', float(cutoff) * unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    if calvados_version in [3, 4]:
        ah.addPerParticleParameter('m')

    print('rc', cutoff * unit.nanometer)

    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa', yukawa_kappa / unit.nanometer)
    yu.addGlobalParameter('shift', np.exp(-yukawa_kappa * 4.0) / 4.0 / unit.nanometer)
    yu.addPerParticleParameter('q')

    for j in range(n_chains):
        # print("j", j)
        begin = j * N  # 0
        end = j * N + N  # n_residues
        for a, e in zip(prot.fasta, yukawa_eps):
            yu.addParticle([e * unit.nanometer * unit.kilojoules_per_mole])
            if calvados_version in [3, 4]:
                m = 1.0 if a in ['R', 'H', 'F', 'Y', 'W'] else 0.0
                ah.addParticle([residues.loc[a].sigmas * unit.nanometer, residues.loc[a].lambdas * unit.dimensionless,
                                m * unit.dimensionless])
            else:
                ah.addParticle([residues.loc[a].sigmas * unit.nanometer, residues.loc[a].lambdas * unit.dimensionless])

        for i in range(begin, end - 1):  # index starts from 0
            residue_idx = i - begin  # index starts from 0, 0+N...
            # print("residue_idx", residue_idx)
            if CoarseGrained=="CA" or isIDP:
                hb.addBond(i, i + 1, 0.38 * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
            else:  # COM or COE or SCCOM
                incomplete = True
                while incomplete:
                    try:
                        ssdomains = get_ssdomains(prot.name.split("@")[0], fdomains, output=False)
                    except Exception:
                        os.system("rm /home/fancao/core*")
                        os.system("rm /home/fancao/IDPs_multi/core*")
                        os.system("sleep 1")
                    else:
                        incomplete = False
                resSeqinDomain = []
                for ssdomain in ssdomains:
                    resSeqinDomain += ssdomain
                if residue_idx+1 in resSeqinDomain or residue_idx+1+1 in resSeqinDomain:
                    hb.addBond(i, i + 1, dismatrix[residue_idx][residue_idx+1] * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
                else:
                    hb.addBond(i, i + 1, 0.38 * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))
            yu.addExclusion(i, i + 1)
            ah.addExclusion(i, i + 1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4 * unit.nanometer)
    ah.setCutoffDistance(cutoff * unit.nanometer)
    return hb, yu, ah

def add_particles(system,residues,prot,n_chains=1):
    for i_chain in range(n_chains):
        system.addParticle((residues.loc[prot.fasta[0]].MW+2)*unit.amu)
        for a in prot.fasta[1:-1]:
            system.addParticle(residues.loc[a].MW*unit.amu)
        system.addParticle((residues.loc[prot.fasta[-1]].MW+16)*unit.amu)
    return system

def build_box(Lx,Ly,Lz):
    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = Lx * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = Ly * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    return a, b, c

def build_topology(fasta,n_chains=1):
    # build CG topology
    top = md.Topology()
    for i_chain in range(n_chains):
        chain = top.add_chain()
        for resname in fasta:
            residue = top.add_residue(resname, chain)
            top.add_atom(resname, element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top.add_bond(chain.atom(i),chain.atom(i+1))
    return top

def p2c(r, phi):
    """
    polar to cartesian
    """
    return (r * np.cos(phi), r * np.sin(phi))

def xy_spiral_array(n, delta=0, arc=.38, separation=.7):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i in range(n):
        coords.append(list(p2c(r, phi))+[0])
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta

def geometry_from_pdb(cwd, dataset, record, name, cycle, replica, pdb, compact_ini, CoarseGrained="CA", ssdomains=None):
    """ positions in nm """
    # pos is used to calculate equilibrium distances
    # ini_pos is used to set initial coordinates
    pdb2predic = f"{cwd}/extract_relax/{name}_rank0_relax.pdb"
    u = mda.Universe(pdb2predic)
    ini_CA = u.select_atoms("name CA")
    # ini_CA.write(f"{cwd}/{dataset}/{record}/{cycle}/ini_CA.pdb")
    if CoarseGrained == "CA":
        cas = u.select_atoms('name CA')  # in-place
        center_of_mass = cas.center_of_mass()
        cas.translate(-center_of_mass)
        pos = cas.positions / 10.  # nm, shape: (n, 3)
    elif CoarseGrained == "COM":  # COM
        pos = []
        resSeqinDomain = []
        for ssdomain in ssdomains:
            resSeqinDomain += ssdomain
        for i in range(len(u.residues)):  # i starts from 0
            if i+1 in resSeqinDomain:  # COM
                pos.append(u.residues[i].atoms.center_of_mass())
            else:  # CA
                pos.append(np.squeeze(u.residues[i].atoms.select_atoms("name CA").positions))
        center_of_mass = u.atoms.center_of_mass()
        pos = (np.array(pos) - center_of_mass) / 10.  # nm
    elif CoarseGrained == "COE":  # COE
        residue_coes = residueCOE(pdb)
        pos = []
        resSeqinDomain = []
        for ssdomain in ssdomains:
            resSeqinDomain += ssdomain
        assert len(u.residues) == len(residue_coes)
        for i in range(len(u.residues)):  # i starts from 0
            if i + 1 in resSeqinDomain:  # COE
                pos.append(residue_coes[i])
            else:  # CA
                pos.append(np.squeeze(u.residues[i].atoms.select_atoms("name CA").positions))
        center_of_mass = u.atoms.center_of_mass()
        pos = (np.array(pos) - center_of_mass) / 10.  # nm
    else:  # SCCOM
        with open(f"{cwd}/{dataset}/{record}/{cycle}/make_ndx{replica}.txt", 'w') as file:
            file.write("a HA\n")
            file.write("\"SideChain\" &! \"HA\"\n")  # note that GLY has two HA atoms, for now take the average of these two atoms
            file.write("q\n")
        os.system(f"cat {cwd}/{dataset}/{record}/{cycle}/make_ndx{replica}.txt | gmx make_ndx -f {pdb2predic} -o {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax{replica}.ndx")
        os.system(f"echo 11 | gmx editconf -f {pdb2predic} -o {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax_SC{replica}.pdb -n {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax{replica}.ndx")
        u_SC = mda.Universe(f"{cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax_SC{replica}.pdb")
        pos = []
        resSeqinDomain = []
        for ssdomain in ssdomains:
            resSeqinDomain += ssdomain
        for i in range(len(u.residues)):  # i starts from 0
            if i + 1 in resSeqinDomain:  # SCCOM
                pos.append(u_SC.residues[i].atoms.center_of_mass())
            else:  # CA
                pos.append(np.squeeze(u.residues[i].atoms.select_atoms("name CA").positions))
        center_of_mass = u.atoms.center_of_mass()
        pos = (np.array(pos) - center_of_mass) / 10.  # nm

    if compact_ini:
        ini_conformation = md.load_pdb(pdb)  # nm
        ini_pos = np.squeeze(ini_conformation.xyz)
        center_of_mass = np.mean(ini_pos,axis=0)  # actually it is center of geometry, nm
        ini_pos -= center_of_mass
        center_of_mass *= 10  # Å
    else:
        ini_pos = pos

    return ini_pos, pos, center_of_mass/10.  # nm

def slab_xy(L,margin):
    xy = np.empty(0)
    xy = np.append(xy,np.random.rand(2)*(L-margin)-(L-margin)/2).reshape((-1,2))
    for x,y in np.random.rand(1000,2)*(L-margin)-(L-margin)/2:
        x1 = x-L if x>0 else x+L
        y1 = y-L if y>0 else y+L
        if np.all(np.linalg.norm(xy-[x,y],axis=1)>.7):
            if np.all(np.linalg.norm(xy-[x1,y],axis=1)>.7):
                if np.all(np.linalg.norm(xy-[x,y1],axis=1)>.7):
                    xy = np.append(xy,[x,y]).reshape((-1,2))
        if xy.shape[0] == 100:
            break
    n_chains = xy.shape[0]
    return xy, n_chains

def slab_dimensions(N):
    if N > 350:
        L = 25.
        Lz = 300.
        margin = 8
        Nsteps = int(2e7)
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
        Nsteps = int(6e7)
    else:
        L = 15.
        Lz = 150.
        margin = 2
        Nsteps = int(6e7)
    return L, Lz, margin, Nsteps

def create_parameters(flib, dataset, initial_type, tune4W=0):
    if initial_type == "C1":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues["CALVADOS1"]
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    if initial_type == "C2":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues["CALVADOS2"]
        if tune4W!=0:
            residues.loc["W", "lambdas"] = tune4W
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    if initial_type in ["CALVADOS_COM"]:
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues[initial_type]
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    print("Properties used:\n", residues)


def load_parameters(flib, dataset, cycle, calvados_version, initial_type):
    # if calvados_version in [1,2]:
    if initial_type in ["C1", "C2", "CALVADOS_COM"]:
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one', drop=False)
    if initial_type in ["0.5", "Ran"]:
        residues = pd.read_csv(f'{flib}/{dataset}/residues_-1.csv').set_index('one', drop=False)

    print("Properties used:\n", residues)
    return residues

def load_pae(input_pae, cutoff=0.25):
    """ pae as pkl file (AF2 format) """
    pae = pd.read_pickle(input_pae)['predicted_aligned_error']/10. + .0001  # (N,N) N is the num of res, nm
    print(pae)
    # pae = np.where(pae == 0, 1, pae)  # avoid division by zero (for i = j)
    pae_inv = 1 / pae  # inverse pae
    pae_inv = np.where(pae_inv > cutoff, pae_inv, 0)
    return pae_inv

def calcAHenergy(cwd,dataset, df,name, cycle):
    term_1 = np.load(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_1.npy')
    term_2 = np.load(f'{cwd}/{dataset}/{name}/{cycle}/energy_sums_2.npy')
    unique_pairs = np.load(f'{cwd}/{dataset}/{name}/{cycle}/unique_pairs.npy')
    lambdas = (df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)/2
    return term_1+np.nansum(lambdas*term_2,axis=1)

def determineThreadsnode(use_newdata, batch_sys, cost):
    if use_newdata:
        if cost <= 5E5:
            Threads = 2
            node = "sbinlab_ib" if batch_sys=='Deic' else 'thinnode'
        elif cost <= 1E7:
            Threads = 4
            node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'
        else:
            Threads = 6
            node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'
    else:
        if cost <= 1.5E5:
            Threads = 2
            node = "sbinlab_ib" if batch_sys=='Deic' else 'thinnode'
        elif cost <= 3E5:
            Threads = 4
            node = "sbinlab_ib" if batch_sys=='Deic' else 'thinnode'
        elif cost <= 1E7:
            Threads = 6
            node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'
        else:
            Threads = 10
            node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'

    return Threads, node

def residueCOE(pdbpath):
    # pdbpath = "/home/people/fancao/IDPs_multi/extract_relax/GS0_rank0_relax.pdb"
    electron_num_dict = {"C": 6, "N": 7, "O": 8, "H": 1, "S": 16}
    current_resSeq = -999
    residue_coes = []
    atoms_cors_dict = defaultdict(list)
    with open(pdbpath, 'r') as file:
        for line in file.readlines():
            record = line.strip()
            ATOM = record[:4].strip()
            if ATOM != "ATOM":  # Detect ATOM start line
                continue
            resName = processIon(record[17:20].strip())  # PRO, Treated protonation conditions
            if resName not in aa:
                continue
            # pdbinfo["serial"].append(int(record[6:11].strip()))  # 697
            name = record[12:16].strip()  # CA
            # pdbinfo["resName"].append(resName)  # PRO, Treated protonation conditions
            resSeq = int(record[22:26].strip())  # 3
            if current_resSeq==-999:
                current_resSeq = resSeq
            x = float(record[30:38].strip())  # Å
            y = float(record[38:46].strip())
            z = float(record[46:54].strip())
            if current_resSeq!=resSeq:
                # print(atoms_cors_dict)
                # element_mass_dict
                total_e = 0
                coeX = 0
                coeY = 0
                coeZ = 0
                for atom_name in atoms_cors_dict.keys():
                    total_e += electron_num_dict[atom_name[0]]
                    coeX += atoms_cors_dict[atom_name][0] * electron_num_dict[atom_name[0]]
                    coeY += atoms_cors_dict[atom_name][1] * electron_num_dict[atom_name[0]]
                    coeZ += atoms_cors_dict[atom_name][2] * electron_num_dict[atom_name[0]]
                residue_coes.append(np.array([coeX,coeY,coeZ])/total_e)
                atoms_cors_dict = defaultdict(list)
                current_resSeq = resSeq
            atoms_cors_dict[name] = [x,y,z]
    total_e = 0
    coeX = 0
    coeY = 0
    coeZ = 0
    for atom_name in atoms_cors_dict.keys():
        total_e += electron_num_dict[atom_name[0]]
        coeX += atoms_cors_dict[atom_name][0] * electron_num_dict[atom_name[0]]
        coeY += atoms_cors_dict[atom_name][1] * electron_num_dict[atom_name[0]]
        coeZ += atoms_cors_dict[atom_name][2] * electron_num_dict[atom_name[0]]
    residue_coes.append(np.array([coeX, coeY, coeZ]) / total_e)
    # residue_coes = np.array(residue_coes)/10.  # nm
    return residue_coes

def determineMask(name, multidomain_names, fasta, fdomains, pairs):
    # exclude all ssdomain and bonds
    # this method is more reasonable, but much slower than simple loops and requests too much memory;
    """mask = np.ones(int(((N - 1) * N) / 2), dtype=bool)
    for domain in ssdomains:
        folded = np.array(list(itertools.combinations(range(domain[0] - 1, domain[-1]), 2)))
        mask &= ~(pairs[:, None] == folded).all(-1).any(-1)
    mask &= np.squeeze(np.abs(pairs[:, 0] - pairs[:, 1]) > 1)"""
    if name in multidomain_names:
        N = len(fasta)  # number of residues
        ssdomains = get_ssdomains(name, fdomains)  # start with 1
        # mask residues within the same domain and bonds, checked
        mask = []
        for i in range(N):
            for j in range(i + 1, N):
                mask_tmp = True
                if np.abs(i - j) == 1:
                    mask_tmp = False
                elif ssdomains != None:  # use fixed domain boundaries for network
                    for ssdom in ssdomains:
                        if i + 1 in ssdom and j + 1 in ssdom:
                            mask_tmp = False
                mask.append(mask_tmp)
        mask = np.array(mask)
    else:  # IDP
        mask = np.abs(pairs[:, 0] - pairs[:, 1]) > 1  # exclude bonds
    return mask

def energy_details(cwd, dataset, name, cycle, fdomains, rc=2.4):
    time_total = time.time()
    df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
    multidomain_names = pd.read_pickle(f"{cwd}/{dataset}/MultiDomainsRgs.pkl").index
    traj = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd", f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")
    # traj = traj[:2]
    fasta = [res.name for res in traj.top.atoms]
    pairs = traj.top.select_pairs('all', 'all')  # index starts with 0
    mask = determineMask(name, multidomain_names, fasta, fdomains, pairs)
    pairs = pairs[mask]
    d = md.compute_distances(traj, pairs)  # (n_traj, n_pairs)
    d[d > rc] = np.inf  # cutoff
    n1 = np.zeros(d.shape, dtype=np.int8)
    n2 = np.zeros(d.shape, dtype=np.int8)
    resSeq = [str(i + 1) for i in range(len(fasta))]
    resSeq_fasta = np.core.defchararray.add(resSeq, fasta)
    pairs = np.array(list(itertools.combinations(fasta, 2)))
    resSeq_fasta_pairs = np.array(list(itertools.combinations(resSeq_fasta, 2)))
    pairs = pairs[mask]
    resSeq_fasta_pairs = resSeq_fasta_pairs[mask]
    sigmas = 0.5 * (df.loc[pairs[:, 0]].sigmas.values + df.loc[pairs[:, 1]].sigmas.values)
    lambdas = (df.loc[pairs[:, 0]].lambdas.values + df.loc[pairs[:, 1]].lambdas.values) / 2

    for i, sigma in enumerate(sigmas):
        mask = d[:, i] > np.power(2., 1. / 6) * sigma
        # n1 is to calculate the constant terms, within 2^(1/6)*sigma
        n1[:, i][~mask] = 1
        # n2 is to calculat all shifted energy, within cutoff
        n2[:, i][np.isfinite(d[:, i])] = 1
    pairs = np.core.defchararray.add(pairs[:, 0], pairs[:, 1])
    resSeq_fasta_pairs = np.core.defchararray.add(resSeq_fasta_pairs[:, 0], resSeq_fasta_pairs[:, 1])
    sigmas6 = np.power(sigmas, 6)
    sigmas12 = np.power(sigmas6, 2)
    eps = 0.2 * 4.184
    # n3 is to calculat: 2^(1/6)*sigma<d<cutoff
    n3 = np.where(n1 == 1, 0, 1) & n2
    tmp = pd.DataFrame({"pairs": pairs,
                        "resSeq_fasta_pairs": resSeq_fasta_pairs,
                        "d": [i for i in d.T],
                        "d<=2^(1/6)*sigma": [i for i in n1.T],
                        "2^(1/6)*sigma<d<cutoff": [i for i in n3.T],
                        "d-12": [i for i in np.power(d, -12.).T],
                        "d-6": [i for i in np.power(d, -6.).T],
                        "sigmas6": sigmas6,
                        "sigmas12": sigmas12,
                        "lambdas": lambdas,
                        }
                       ).groupby("pairs", group_keys=True).apply(lambda x: x)
    tmp["energy"] = tmp[
        ["d-12", "d-6", "sigmas6", "sigmas12", "lambdas", "d<=2^(1/6)*sigma", "2^(1/6)*sigma<d<cutoff"]].apply(
        lambda x: eps * (x["d<=2^(1/6)*sigma"] * (
                4 * (x["sigmas12"] * x["d-12"] - x["sigmas6"] * x["d-6"]) - 4 * x["lambdas"] * (
                x["sigmas12"] * np.power(rc, -12) - x["sigmas6"] * np.power(rc, -6)) + (1 - x["lambdas"])) + x[
                             "2^(1/6)*sigma<d<cutoff"] * (4 * x["lambdas"] * (
                x["sigmas12"] * x["d-12"] - x["sigmas6"] * x["d-6"] - x["sigmas12"] * np.power(rc, -12) + x[
            "sigmas6"] * np.power(rc, -6)))), axis=1)
    print(tmp["energy"].sum())
    tmp.to_pickle(f"{cwd}/{dataset}/{name}/{cycle}/energy_details.pkl")
    time_end = time.time()
    target_seconds = time_end - time_total  # total used time
    print(f"total optimization used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")

def resetexpRgErr(proteinsRgs):
    tmp = proteinsRgs.expRgErr.values/proteinsRgs.expRg.values
    proteinsRgs["expRgErr"] = tmp.mean() * proteinsRgs.expRg.values
    return proteinsRgs

def balance_chi2_rg(IDPsRgs, MultiDomainsRgs, proteinsRgs):
    multidomain_names = list(MultiDomainsRgs.index)
    IDP_names = list(IDPsRgs.index)
    chi2_rg = (proteinsRgs.loc[multidomain_names]["chi2_rg"].mean()+proteinsRgs.loc[IDP_names]["chi2_rg"].mean())/2
    return chi2_rg

def build_xygrid(N,box,z=0.):
    """ Grid for xy slabs """
    if np.sqrt(N) % 1 > 0:
        b = 2
    else:
        b = 1
    nx = int(np.sqrt(N)) + b # nx spots in x dim
    ny = int(np.sqrt(N)) + b # ny spots in x dim

    dx = box[0] / nx
    dy = box[1] / ny

    xy = []
    x, y = 0., 0.
    ct = 0
    for n in range(N):
        ct += 1
        xy.append([x,y,z])
        if ct == ny:
            y = 0
            x += dx
            ct = 0
        else:
            y += dy
    xy = np.array(xy)
    return xy

def cal_COM(cors: np.array,  # (n,3)
            mass_weights: np.array):
    assert len(cors) == len(mass_weights)
    return np.sum((cors.T * mass_weights).T, axis=0) / mass_weights.sum()

def residueCOM(pdbpath):
    # might have a bit differences compared with other methods, but it is correct
    # pdbpath = "/home/people/fancao/IDPs_multi/extract_relax/GS0_rank0_relax.pdb"
    current_resSeq = -999
    residue_coms = []
    atoms_cors_dict = defaultdict(list)
    with open(pdbpath, 'r') as file:
        for line in file.readlines():
            record = line.strip()
            ATOM = record[:4].strip()
            if ATOM != "ATOM":  # Detect ATOM start line
                continue
            resName = processIon(record[17:20].strip())  # PRO, Treated protonation conditions
            if resName not in aa:
                continue
            name = record[12:16].strip()  # CA
            resSeq = int(record[22:26].strip())  # 3
            if current_resSeq==-999:
                current_resSeq = resSeq
            x = float(record[30:38].strip())  # Å
            y = float(record[38:46].strip())
            z = float(record[46:54].strip())
            if current_resSeq != resSeq:
                total_mass = 0
                comX = 0
                comY = 0
                comZ = 0
                for atom_name in atoms_cors_dict.keys():
                    total_mass += element_mass_dict[atom_name[0]]
                    comX += atoms_cors_dict[atom_name][0] * element_mass_dict[atom_name[0]]
                    comY += atoms_cors_dict[atom_name][1] * element_mass_dict[atom_name[0]]
                    comZ += atoms_cors_dict[atom_name][2] * element_mass_dict[atom_name[0]]
                residue_coms.append(np.array([comX, comY, comZ]) / total_mass)
                atoms_cors_dict = defaultdict(list)
                current_resSeq = resSeq
            atoms_cors_dict[name] = [x, y, z]
    total_mass = 0
    comX = 0
    comY = 0
    comZ = 0
    for atom_name in atoms_cors_dict.keys():
        total_mass += element_mass_dict[atom_name[0]]
        comX += atoms_cors_dict[atom_name][0] * element_mass_dict[atom_name[0]]
        comY += atoms_cors_dict[atom_name][1] * element_mass_dict[atom_name[0]]
        comZ += atoms_cors_dict[atom_name][2] * element_mass_dict[atom_name[0]]
    residue_coms.append(np.array([comX, comY, comZ]) / total_mass)
    # print(residue_coms)
    return residue_coms

def place_chains(pos, chains, L, isIDP):
    # the origin is at the point of box in openMM
    x_interval = np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0]))+1
    x_allows = int((L // x_interval))
    y_interval = np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1]))+1
    y_allows = int((L // y_interval))
    n_Zplane = int(x_allows*y_allows)  # num of molecules allowed for each Z plane
    if isIDP:
        z_interval = 1.47  # nm
    else:
        z_interval = np.ceil(np.abs(np.max(pos[:, 2]) - np.min(pos[:, 2])))+1
    pos_chains = []
    print(x_allows, y_allows, n_Zplane, chains//n_Zplane)
    added_num = 0
    for zplane in range((chains//n_Zplane)+1):
        for yplane in range(y_allows):
            for xplane in range(x_allows):
                if added_num<chains:
                    # not fully symmetric, because it is translation
                    interval_vec = np.array([x_interval*xplane, y_interval*yplane, z_interval*zplane])
                    pos_chains+=(pos+interval_vec).tolist()
                    added_num += 1
    pos_chains = np.array(pos_chains)
    pos_chains_COG = np.mean(pos_chains, axis=0)
    pos_chains -= pos_chains_COG
    return pos_chains, pos_chains_COG  # nm

def force2center(box, k):
    rcent_expr = 'k*abs(periodicdistance(x,y,z,x,y,z0))'
    rcent = openmm.openmm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k', k * unit.kilojoules_per_mole / unit.nanometer)
    rcent.addGlobalParameter('z0', box[2] / 2. * unit.nanometer)  # center of box in z
    rcent.setName("rcent")
    return rcent

def calc_zpatch(z,h):
    cutoff = np.min(h)
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = []
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x  # accumulating the number of atoms bigger than cutoff
            zwindow.append(z[ix])  # add bins where values bigger than cutoff
            hwindow.append(x)  # add values bigger than cutoff
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def center_slab(cwd, dataset, record, cycle, replica, path2traj):
    u = MDAnalysis.Universe(f'{cwd}/{dataset}/{record}/{cycle}/top_{replica}.pdb',  # Å
        path2traj,in_memory=True)
    n_frames = len(u.trajectory)
    ag = u.atoms
    n_atoms = ag.n_atoms
    lz = u.dimensions[2]  # Å
    print("lz: ", lz)
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    with MDAnalysis.Writer(f'{cwd}/{dataset}/{record}/wrapped.dcd',n_atoms) as W:
        for t,ts in enumerate(u.trajectory):
            # shift max density to center
            zpos = ag.positions.T[2]  # (n_atoms,)
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]  # maximum in zmax
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)  # find the continuous subset that gives the biggest number of atoms
            zmid = np.average(zpatch,weights=hpatch)  # weighted
            ag.translate(np.array([0,0,-zmid+0.5*lz]))  # centered on weighted center
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            hs[t] = h
            W.write(ag)
    np.save(f'{cwd}/{dataset}/{record}/{record}.npy',hs,allow_pickle=False)
    return hs, z

def calcProfiles_rc(cwd, dataset, record, cycle):
    config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml", 'r'))
    L = config["L"]
    model = record
    value = pd.DataFrame(index=[record], dtype=object)
    error = value.copy()
    nskip = 1200
    df_proteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl").astype(object)
    proteins = [record]
    Lz = config["Lz"]
    initial_type = config['initial_type']
    CoarseGrained = config['CoarseGrained']
    save_interval = 0.5  # ns
    for i, m in enumerate(proteins):
        if os.path.isfile(f"{cwd}/{dataset}/{record}/{record}.npy"):
            h = np.load(f"{cwd}/{dataset}/{record}/{record}.npy")
            fasta = df_proteins.loc[m].fasta
            N = len(fasta)
            conv = 1 / 6.022 / N / L / L / 0.1 * 1e4
            h = h[nskip:] * conv  # mM/L of every bin
            print(h.shape)
            lz = h.shape[1] + 1
            edges = np.arange(-lz / 2., lz / 2., 1) / 10
            dz = (edges[1] - edges[0]) / 2.
            z = edges[:-1] + dz
            profile = lambda x, a, b, c, d: .5 * (a + b) + .5 * (b - a) * np.tanh((np.abs(x) - c) / d)
            residuals = lambda params, *args: (args[1] - profile(args[0], *params))
            hm = np.mean(h, axis=0)
            z1 = z[z > 0]
            h1 = hm[z > 0]
            z2 = z[z < 0]
            h2 = hm[z < 0]
            p0 = [1, 1, 1, 1]
            res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0] * 4, [100] * 4))
            res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0] * 4, [100] * 4))
            print(res1.x)
            print(res2.x)
            cutoffs1 = [res1.x[2] - .5 * res1.x[3], -res2.x[2] + .5 * res2.x[3]]
            cutoffs2 = [res1.x[2] + 6 * res1.x[3], -res2.x[2] - 6 * res2.x[3]]

            if np.abs(cutoffs2[1] / cutoffs2[0]) > 2:
                print('WRONG', m, model, cutoffs1, cutoffs2)
                print(res1.x, res2.x)
            if np.abs(cutoffs2[1] / cutoffs2[0]) < 0.5:
                print('WRONG', m, model, cutoffs1, cutoffs2)
                print(res1.x, res2.x)
                plt.plot(z1, h1)
                plt.plot(z2, h2)
                plt.plot(z1, profile(z1, *res1.x), color='tab:blue')
                plt.plot(z2, profile(z2, *res2.x), color='tab:orange')
                cutoffs2[0] = -cutoffs2[1]
                print(cutoffs2)

            bool1 = np.logical_and(z < cutoffs1[0], z > cutoffs1[1])  # dense region
            bool2 = np.logical_or(z > cutoffs2[0], z < cutoffs2[1])  # dilute region

            dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, h)  # averaged over dilute region, (n_raj-nskip,)
            np.save(f"{cwd}/{dataset}/{record}/dilarray.npy", dilarray)
            denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, h)  # averaged over dense region, (n_raj-nskip,)
            dil = hm[bool2].mean()
            den = hm[bool1].mean()

            block_dil = BlockAnalysis(dilarray)
            block_den = BlockAnalysis(denarray)
            block_dil.SEM()
            block_den.SEM()

            value.loc[m, model + '_dil'] = block_dil.av  # same result to np.mean(dilarray)
            value.loc[m, model + '_den'] = block_den.av

            error.loc[m, model + '_dil'] = block_dil.sem
            error.loc[m, model + '_den'] = block_den.sem
            fig, ax = plt.subplots(1, 2, figsize=(8, 4))
            ax[0].plot(z, hm)
            # ax[0].set_title("Concentration profiles")
            # ax[1].set_title("Evolution of concentration")
            for c1, c2 in zip(cutoffs1, cutoffs2):
                ax[0].axvline(c1, color='gray')
                ax[0].axvline(c2, color='black')
            ax[0].set(xlabel='z [nm]', ylabel='Concentration [mM]')
            ax[1].set(xlabel='z [nm]', ylabel='Time [μs]')
            im = ax[1].imshow(h, cmap=plt.cm.Blues, origin='lower', norm=LogNorm(vmin=1e-2, vmax=1e2), aspect='auto')
            ax[1].set_yticks([(i + 1) * 1000 for i in range(h.shape[0] // 1000)], [f"{(i + 1) * save_interval}" for i in
                range(h.shape[0] // 1000)])
            ax[1].set_xticks([Lz * 10 / 2 - 500, Lz * 10 / 2 + 500], ["-50", "+50"])
            plt.colorbar(im)
            fig.suptitle(f"{initial_type}_{CoarseGrained} {record}\n Csat: {np.round(value.loc[m, model + '_dil'] * 1000)} μM")
            plt.savefig(f"{cwd}/{dataset}/{record}/{record}.png", dpi=300)
        else:
            print('DATA NOT FOUND FOR', m, model)
    print(value)
    value.to_pickle(f"{cwd}/{dataset}/{record}/value_{record}.pkl")
    error.to_pickle(f"{cwd}/{dataset}/{record}/error_{record}.pkl")

def backup_traj(traj_pool:str, traj_pre:str, top:str):
    if os.path.isfile(traj_pool):
        traj_backup = md.load_dcd(traj_pool, top)
        traj_backup = md.join([traj_backup, md.load_dcd(traj_pre, top)])
        traj_backup.save_dcd(traj_pool)
    else:
        os.system(f"cp {traj_pre} {traj_pool}")

def determineLz(cwd, dataset, record, name, cycle, replica, path2pdb, compact_ini, CoarseGrained, ssdomains, chains, L, isIDP):
    ini_pos, pos, center_of_mass = geometry_from_pdb(cwd, dataset, record, name, cycle, replica, path2pdb, compact_ini, CoarseGrained=CoarseGrained, ssdomains=ssdomains)  # nm
    ini_pos, center_of_mass = place_chains(ini_pos + center_of_mass, chains, L, isIDP)
    Lz = int(np.round(np.abs(np.max(ini_pos[:,2])-np.min(ini_pos[:,2]))*2, decimals=-1))
    return Lz


if __name__ == '__main__':
    # determineLz("/home/fancao/IDPs_multi", "slabC2_COM_cutoff2.0_1", "hnRNPA1S_backup", "hnRNPA1S", 1, 0, "/home/fancao/IDPs_multi/extract_relax/hnRNPA1S_COM_ini.pdb", True, "COM", get_ssdomains("hnRNPA1S", "/home/fancao/IDPs_multi/domains.yaml"), 150, 20, False)
    pass