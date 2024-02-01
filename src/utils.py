import os.path
import time
from DEERPREdict.PRE import PREpredict
import ray
from pulchra import fix_topology
from collections import defaultdict
import MDAnalysis
from jinja2 import Template
from protein_repo import *
import yaml
from simtk import openmm, unit
import mdtraj as md
import itertools
from rawdata import calcChi2, reweightRg
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
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
#SBATCH --job-name={{record}}_sim{{cycle}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task={{requested_cpunum}}
#SBATCH --partition={{node}}
{{sim_dependency}}
#SBATCH --mem={{mem}}GB
#SBATCH -t 24:00:00
#SBATCH -o {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{cycle}}_out_sim_{{replicas_list4MD_idx}}
#SBATCH -e {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{cycle}}_err_sim_{{replicas_list4MD_idx}}

source /groups/sbinlab/fancao/.bashrc

conda activate Calvados_MDP
module load astro
module load gcc/9.3-offload

python3 {{cwd}}/simulate.py --config {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{config_sim_filename}}""")

submission_2 = Template(
"""#!/bin/bash
#SBATCH --job-name={{record}}_merge_{{cycle}}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=sbinlab_ib2
#SBATCH -t 24:00:00
#SBATCH --mem={{mem}}GB
{{merge_dependency}}
#SBATCH -o {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{cycle}}_merge_out
#SBATCH -e {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{cycle}}_merge_err

source /groups/sbinlab/fancao/.bashrc

conda activate Calvados_MDP
module load astro
module load gcc/9.3-offload

python3 {{cwd}}/merge_replicas.py --config {{cwd}}/{{dataset}}/{{record}}/{{cycle}}/{{config_mer_filename}}""")


submission_3 = Template(
"""#!/bin/bash
#SBATCH --job-name=opt_{{cycle}}
#SBATCH --nodes=1
#SBATCH --partition=sbinlab
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=10
{{opt_dependency}}
#SBATCH -o {{cwd}}/{{dataset}}/{{cycle}}_out
#SBATCH -e {{cwd}}/{{dataset}}/{{cycle}}_err

source /groups/sbinlab/fancao/.bashrc

conda activate Calvados_MDP
module load astro
module load gcc/9.3-offload

declare -a proteinsPRE_list=({{proteins}})

for name in ${proteinsPRE_list[@]}
do
cp -r {{cwd}}/expPREs/$name/expPREs {{cwd}}/{{dataset}}/$name
python3 {{cwd}}/pulchra.py --cwd {{cwd}} --dataset {{dataset}} --name $name --cycle {{cycle}} --num_cpus 10 --pulchra {{path2pulchra}}
done

python3 {{cwd}}/optimize.py  --path2config {{path2config}}""")

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

def write_config(cwd, dataset, record, cycle,config_data,config_filename):
    with open(f'{cwd}/{dataset}/{record}/{cycle}/{config_filename}','w') as stream:
        yaml.dump(config_data,stream)

def set_harmonic_network(N,dmap,pae_inv,yu,ah,ssdomains=None,cs_cutoff=0.9,k_restraint=700.):
    cs = openmm.openmm.HarmonicBondForce()
    for i in range(N-2):
        for j in range(i+2,N):
            if ssdomains != None:  # use fixed domain boundaries for network
                ss = False
                for ssdom in ssdomains:
                    if i+1 in ssdom and j+1 in ssdom:
                        ss = True
                if ss:  # both residues in structured domains
                    if dmap[i,j] < cs_cutoff:  # nm
                        cs.addBond(i, j, dmap[i, j] * unit.nanometer,
                                   k_restraint * unit.kilojoules_per_mole / (unit.nanometer ** 2))
                        yu.addExclusion(i, j)
                        ah.addExclusion(i, j)
            elif isinstance(pae_inv, np.ndarray):  # use alphafold PAE matrix for network
                k = k_restraint * pae_inv[i,j]**2
                if k > 0.0:
                    cs.addBond(i,j, dmap[i,j]*unit.nanometer,
                                k*unit.kilojoules_per_mole/(unit.nanometer**2))
                    yu.addExclusion(i, j)
                    ah.addExclusion(i, j)
            else:
                raise
    cs.setUsesPeriodicBoundaryConditions(True)
    return cs, yu, ah

def set_interactions(system, residues, prot, lj_eps, cutoff, yukawa_kappa, yukawa_eps, N, n_chains=1,
                     CoarseGrained="CA", dismatrix=None, isIDP=True, fdomains=None):
    hb = openmm.openmm.HarmonicBondForce()
    # interactions
    energy_expression = 'select(step(r-2^(1/6)*s),4*eps*l*((s/r)^12-(s/r)^6-shift),4*eps*((s/r)^12-(s/r)^6-l*shift)+eps*(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression + '; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/rc)^12-(0.5*(s1+s2)/rc)^6')
    ah.addGlobalParameter('eps', lj_eps * unit.kilojoules_per_mole)
    ah.addGlobalParameter('rc', float(cutoff) * unit.nanometer)
    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    yu = openmm.openmm.CustomNonbondedForce('q*(exp(-kappa*r)/r-shift); q=q1*q2')
    yu.addGlobalParameter('kappa', yukawa_kappa / unit.nanometer)
    yu.addGlobalParameter('shift', np.exp(-yukawa_kappa * 4.0) / 4.0 / unit.nanometer)
    yu.addPerParticleParameter('q')

    for j in range(n_chains):
        begin = j * N  # 0
        end = j * N + N  # n_residues
        for a, e in zip(prot.fasta, yukawa_eps):
            yu.addParticle([e * unit.nanometer * unit.kilojoules_per_mole])
            ah.addParticle([residues.loc[a].sigmas * unit.nanometer, residues.loc[a].lambdas * unit.dimensionless])

        for i in range(begin, end - 1):  # index starts from 0
            if CoarseGrained=="CA" or isIDP:
                hb.addBond(i, i + 1, 0.38 * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))

            else:  # COM or COE
                ssdomains = get_ssdomains(prot.name.split("@")[0], fdomains, output=False)
                resSeqinDomain = []
                for ssdomain in ssdomains:
                    resSeqinDomain += ssdomain
                if i+1 in resSeqinDomain or i+1+1 in resSeqinDomain:
                    hb.addBond(i, i + 1, dismatrix[i][i+1] * unit.nanometer, 8033.0 * unit.kilojoules_per_mole / (unit.nanometer ** 2))

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

def geometry_from_pdb(cwd, dataset, name, record, cycle, replica, pdb, CoarseGrained="CA", ssdomains=None):
    """ positions in nm """
    u = mda.Universe(pdb)
    ini_CA = u.select_atoms("name CA")
    ini_CA.write(f"{cwd}/{dataset}/{record}/{cycle}/ini_CA.pdb")
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
        os.system(f"cat {cwd}/{dataset}/{record}/{cycle}/make_ndx{replica}.txt | gmx make_ndx -f {cwd}/extract_relax/{name}_rank0_relax.pdb -o {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax{replica}.ndx")
        os.system(f"echo 11 | gmx editconf -f {cwd}/extract_relax/{name}_rank0_relax.pdb -o {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax_SC{replica}.pdb -n {cwd}/{dataset}/{record}/{cycle}/{name}_rank0_relax{replica}.ndx")
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

    return pos, center_of_mass/10.  # nm

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

def create_parameters(flib, dataset, cycle, initial_type):
    if initial_type == "C1":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues["CALVADOS1"]
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    if initial_type == "C2":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues["CALVADOS2"]
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    if initial_type == "C3":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one')
        residues["lambdas"] = residues["CALVADOS3"]
        residues.to_csv(f'{flib}/{dataset}/residues_pub.csv')
    if initial_type == "0.5":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_-1.csv').set_index('one')
    if initial_type == "Ran":
        residues = pd.read_csv(f'{flib}/{dataset}/residues_-1.csv').set_index('one')
        residues["Ran"] = np.random.rand(20)
        residues.lambdas = residues["Ran"]
        residues.to_csv(f'{flib}/{dataset}/residues_-1.csv')
    print("Properties used:\n", residues)


def load_parameters(flib, dataset, cycle, initial_type):
    cycle = int(cycle)
    if cycle == 0:
        # if calvados_version in [1,2]:
        if initial_type in ["C1","C2","C3"]:
            residues = pd.read_csv(f'{flib}/{dataset}/residues_pub.csv').set_index('one', drop=False)
        if initial_type == "0.5":
            residues = pd.read_csv(f'{flib}/{dataset}/residues_-1.csv').set_index('one', drop=False)
        if initial_type == "Ran":
            residues = pd.read_csv(f'{flib}/{dataset}/residues_-1.csv').set_index('one', drop=False)
    else:
        residues = pd.read_csv(f'{flib}/{dataset}/residues_{cycle - 1}.csv').set_index('one', drop=False)
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

def visualize_traj(cwd, dataset, record, cycle):
    traj_dcd = fix_topology(f'{cwd}/{dataset}/{record}/{cycle}/{record}.dcd', f'{cwd}/{dataset}/{record}/{cycle}/{record}.pdb')
    traj_dcd = traj_dcd.superpose(traj_dcd[0])
    # L = np.squeeze(traj_dcd[0].unitcell_lengths)[0]  # nm
    # the origin is at the point of box
    # traj_dcd.xyz = traj_dcd.xyz + L/2  # nm  <centering>
    traj_dcd[0].save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/{record}_first.pdb')
    traj_dcd[-1].save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/{record}_last.pdb')
    # traj_dcd.save_trr(f"{cwd}/{dataset}/{record}/{cycle}/{record}.trr")
    # traj_dcd.save_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}_CA.dcd")

    first_pdb = md.load_pdb(f'{cwd}/{dataset}/{record}/{cycle}/{record}_first.pdb')
    ini_beads_CA = md.load_pdb(f"{cwd}/{dataset}/{record}/{cycle}/ini_beads.pdb", top=first_pdb.top)
    ini_beads_CA[0].save_pdb(f"{cwd}/{dataset}/{record}/{cycle}/ini_beads_CA.pdb")


@ray.remote(num_cpus=1)
def evaluatePRE(cwd, dataset, label, record, cycle, prot, log_path):
    if not os.path.isdir(f"{cwd}/{dataset}/{log_path}/{record}"):
        os.system(f"mkdir -p {cwd}/{dataset}/{log_path}/{record}")

    prefix = f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res'
    filename = prefix+'-{:d}.pkl'.format(label)
    if isinstance(prot.weights, np.ndarray):
        u = MDAnalysis.Universe(f'{cwd}/{dataset}/{record}/{cycle}/allatom.pdb')
        load_file = filename
    elif isinstance(prot.weights, bool) or isinstance(prot.weights, np.bool_):  # different from original code
        u = MDAnalysis.Universe(f'{cwd}/{dataset}/{record}/{cycle}/allatom.pdb',f'{cwd}/{dataset}/{record}/{cycle}/allatom.dcd')  # Å
        load_file = False
    else:
        raise ValueError('Weights argument is a '+str(type(prot.weights)))
    PRE = PREpredict(u, label, log_file=f'{cwd}/{dataset}/{log_path}/{record}/log', temperature=prot.temp, atom_selection='N', sigma_scaling=1.0)
    PRE.run(output_prefix=prefix, weights=prot.weights, load_file=load_file, tau_t=1e-10, tau_c=prot.tau_c*1e-09, r_2=10, wh=prot.wh)

@ray.remote(num_cpus=1)
def calcDistSums(cwd,dataset, df,name,record,cycle,prot, multidomain_names, rc, fdomains):
    if not os.path.isfile(f'{cwd}/{dataset}/{record}/{cycle}/energy_sums_2.npy'):
        traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd",f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")
        fasta = [res.name for res in traj.top.atoms]
        pairs = traj.top.select_pairs('all','all')  # index starts with 0
        mask = determineMask(name, record, multidomain_names, fasta, fdomains, pairs)
        pairs = pairs[mask]
        d = md.compute_distances(traj,pairs)
        d[d>rc] = np.inf  # cutoff
        r = np.copy(d)
        n1 = np.zeros(r.shape,dtype=np.int8)
        n2 = np.zeros(r.shape,dtype=np.int8)
        pairs = np.array(list(itertools.combinations(fasta,2)))
        pairs = pairs[mask]
        sigmas = 0.5*(df.loc[pairs[:,0]].sigmas.values+df.loc[pairs[:,1]].sigmas.values)
        for i,sigma in enumerate(sigmas):
            mask = r[:,i]>np.power(2.,1./6)*sigma
            r[:,i][mask] = np.inf  # cutoff
            n1[:,i][~mask] = 1
            n2[:,i][np.isfinite(d[:,i])] = 1

        unique_pairs = np.unique(pairs,axis=0)
        pairs = np.core.defchararray.add(pairs[:,0],pairs[:,1])
        d12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),
                              axis=1, arr=np.power(d,-12.))
        d6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),
                              axis=1, arr=-np.power(d,-6.))
        r12 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=np.power(r,-12.))
        r6 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=-np.power(r,-6.))
        # ncut1 is to calculate the constant terms, within 2^(1/6)*sigma
        ncut1 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n1)
        # ncut2 is to calculat all shifted energy, within cutoff
        ncut2 = np.apply_along_axis(lambda x: pd.Series(index=pairs,data=x).groupby(level=0).sum(),axis=1, arr=n2)
        sigmas = 0.5*(df.loc[unique_pairs[:,0]].sigmas.values+df.loc[unique_pairs[:,1]].sigmas.values)
        sigmas6 = np.power(sigmas,6)
        sigmas12 = np.power(sigmas6,2)
        eps = 0.2*4.184
        term_1 = eps*(ncut1+4*(sigmas6*r6 + sigmas12*r12))
        term_2 = sigmas6*(d6+ncut2/np.power(rc,6)) + sigmas12*(d12-ncut2/np.power(rc,12))
        term_2 = 4*eps*term_2 - term_1
        np.save(f'{cwd}/{dataset}/{record}/{cycle}/energy_sums_1.npy',term_1.sum(axis=1))
        np.save(f'{cwd}/{dataset}/{record}/{cycle}/energy_sums_2.npy',term_2)
        np.save(f'{cwd}/{dataset}/{record}/{cycle}/unique_pairs.npy',unique_pairs)

def calcAHenergy(cwd,dataset, df,record, cycle):
    term_1 = np.load(f'{cwd}/{dataset}/{record}/{cycle}/energy_sums_1.npy')
    term_2 = np.load(f'{cwd}/{dataset}/{record}/{cycle}/energy_sums_2.npy')
    unique_pairs = np.load(f'{cwd}/{dataset}/{record}/{cycle}/unique_pairs.npy')
    lambdas = (df.loc[unique_pairs[:,0]].lambdas.values+df.loc[unique_pairs[:,1]].lambdas.values)/2
    return term_1+np.nansum(lambdas*term_2,axis=1)

@ray.remote(num_cpus=1)
def calcWeights(cwd,dataset, df,record,cycle,prot):
    new_ah_energy = calcAHenergy(cwd,dataset, df,record, cycle)
    ah_energy = np.load(f'{cwd}/{dataset}/{record}/{cycle}/{record}_AHenergy.npy')
    kT = 8.3145*prot.temp*1e-3
    weights = np.exp((ah_energy-new_ah_energy)/kT)
    weights /= weights.sum()
    eff = np.exp(-np.sum(weights*np.log(weights*weights.size)))
    return record,weights,eff

def reweight(cwd, dataset, cycle,dp,df,proteinsPRE,proteinsRgs, multidomain_names, proc_PRE, log_path, lambda_oneMax):
    IDP_names = np.setdiff1d(list(proteinsRgs.index), multidomain_names).tolist()
    Rgloss4multidomain = np.ones(shape=len(multidomain_names))
    Rgloss4IDPs = np.ones(shape=len(IDP_names))
    trial_proteinsPRE = proteinsPRE.copy()
    trial_proteinsRgs = proteinsRgs.copy()
    trial_df = df.copy()
    res_sel = np.random.choice(trial_df.index, 5, replace=False)
    trial_df.loc[res_sel,'lambdas'] += np.random.normal(0,dp,res_sel.size)
    if lambda_oneMax:
        f_out_of_01 = lambda df : df.loc[(df.lambdas<=0)|(df.lambdas>1),'lambdas'].index
    else:
        f_out_of_01 = lambda df: df.loc[df.lambdas <= 0, 'lambdas'].index  # remove 1-maximum limit
    out_of_01 = f_out_of_01(trial_df)
    trial_df.loc[out_of_01,'lambdas'] = df.loc[out_of_01,'lambdas']

    # calculate AH energies, weights and fraction of effective frames
    weights = ray.get([calcWeights.remote(cwd,dataset, trial_df,record,cycle,prot) for record,prot in pd.concat((trial_proteinsPRE,trial_proteinsRgs),sort=True).iterrows()])
    for record,w,eff in weights:
        if eff < 0.6:
            return False, df, proteinsPRE, proteinsRgs, Rgloss4multidomain, Rgloss4IDPs
        if record in trial_proteinsPRE.index:
            trial_proteinsPRE.at[record,'weights'] = w
            trial_proteinsPRE.at[record,'eff'] = eff
        else:
            trial_proteinsRgs.at[record,'weights'] = w
            trial_proteinsRgs.at[record,'eff'] = eff
    # calculate PREs and cost function
    ray.get([evaluatePRE.remote(cwd,dataset,label,record,cycle,trial_proteinsPRE.loc[record], log_path) for n,(label,record) in enumerate(proc_PRE)])
    for record in trial_proteinsPRE.index:
        trial_proteinsPRE.at[record,'chi2_pre'] = calcChi2(cwd, dataset, record, cycle, trial_proteinsPRE.loc[record])
    for record in trial_proteinsRgs.index:
        rg, chi2_rg = reweightRg(df,record,trial_proteinsRgs.loc[record])
        if record in multidomain_names:
            Rgloss4multidomain[np.where(np.array(multidomain_names)==record)[0][0]] = chi2_rg
        else:
            Rgloss4IDPs[np.where(np.array(IDP_names)==record)[0][0]] = chi2_rg
        trial_proteinsRgs.at[record,'Rg'] = rg
        trial_proteinsRgs.at[record,'chi2_rg'] = chi2_rg

    return True, trial_df, trial_proteinsPRE, trial_proteinsRgs, Rgloss4multidomain, Rgloss4IDPs

def determineThreadsnode(batch_sys, cost):
    if cost <= 1.5E5:
        Threads = 2
        node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'  # sbinlab_ib
    elif cost <= 2.5E5:
        Threads = 4
        node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'  # sbinlab_ib
    elif cost <= 1E7:
        Threads = 6
        node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'  # sbinlab_ib2
    else:
        Threads = 10
        node = "sbinlab_ib2" if batch_sys=='Deic' else 'thinnode'  # sbinlab_ib2

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

def determineMask(name, record, multidomain_names, fasta, fdomains, pairs):
    # exclude all ssdomain and bonds
    # this method is more reasonable, but much slower than simple loops and requests too much memory;
    """mask = np.ones(int(((N - 1) * N) / 2), dtype=bool)
    for domain in ssdomains:
        folded = np.array(list(itertools.combinations(range(domain[0] - 1, domain[-1]), 2)))
        mask &= ~(pairs[:, None] == folded).all(-1).any(-1)
    mask &= np.squeeze(np.abs(pairs[:, 0] - pairs[:, 1]) > 1)"""
    if record in multidomain_names:
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

def place_chains(pos, chains, L):
    # the origin is at the point of box in openMM
    x_interval = np.ceil(np.abs(np.max(pos[:, 0]) - np.min(pos[:, 0])))+1
    x_allows = int((L // x_interval - 2))
    y_interval = np.ceil(np.abs(np.max(pos[:, 1]) - np.min(pos[:, 1])))+1
    y_allows = int((L // y_interval - 2))
    n_Zplane = int(x_allows*y_allows)  # num of molecules allowed for each Z plane
    z_interval = np.ceil(np.abs(np.max(pos[:, 2]) - np.min(pos[:, 2])))+1
    pos_chains = []
    # print(x_allows, y_allows, n_Zplane, chains//n_Zplane)
    added_num = 0
    for zplane in range((chains//n_Zplane)+1):
        for yplane in range(y_allows):
            for xplane in range(x_allows):
                if added_num<chains:
                    interval_vec = np.array([x_interval*xplane, y_interval*yplane, z_interval*zplane])
                    pos_chains+=(pos+interval_vec).tolist()
                    added_num += 1
    pos_chains = np.array(pos_chains)
    Lz = np.abs(np.max(pos_chains[:,2])-np.min(pos_chains[:,2]))+z_interval*4
    pos_chains_COG = np.mean(pos_chains, axis=0)  # center of geometry
    pos_chains -= pos_chains_COG
    return pos_chains, Lz, pos_chains_COG  # nm

def force2center(box, k):
    rcent_expr = 'k*abs(periodicdistance(x,y,z,x,y,z0))'
    rcent = openmm.openmm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k', k * unit.kilojoules_per_mole / unit.nanometer)
    rcent.addGlobalParameter('z0', box[2] / 2. * unit.nanometer)  # center of box in z
    return rcent

def takeports(cwd):
    if not os.path.exists(f"{cwd}/ports_pool.npy"):
        ports_pool = set(range(10001,26662))
    else:
        ports_pool = np.load(f"{cwd}/ports_pool.npy", allow_pickle=True).item()
    try:
        picked_ports = np.random.choice(list(ports_pool), size=9, replace=False)
    except Exception:
        raise Exception("please manually delete ports_pool.npy")
    else:
        for port in picked_ports:
            ports_pool.remove(port)
        np.save(f"{cwd}/ports_pool.npy", ports_pool)
        return picked_ports

def takeWorkerports(cwd, num_of_jobs):
    num_of_ports = num_of_jobs+1
    if not os.path.exists(f"{cwd}/workerports.npy"):
        workerports = set(range(26662,65536))
    else:
        workerports = np.load(f"{cwd}/workerports.npy", allow_pickle=True).item()
    try:
        picked_workerports = list(workerports)[:num_of_ports]  # at least 2
    except Exception:
        raise Exception("please manually delete workerports.npy")
    else:
        for port in picked_workerports:
            workerports.remove(port)
        np.save(f"{cwd}/workerports.npy", workerports)
        return picked_workerports

if __name__ == '__main__':
    pass