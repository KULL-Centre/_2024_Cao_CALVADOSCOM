import numpy as np
import pandas as pd
import mdtraj as md
import time
import ray
import os
import subprocess
from argparse import ArgumentParser

def fix_topology(dcd,pdb):
    """
    Changes atom names to CA; even if CG-strategy is COM, it is still CA for convenience
    """
    t = md.load_dcd(dcd,pdb)
    cgtop = md.Topology()
    cgchain = cgtop.add_chain()
    for atom in t.top.atoms:
        cgres = cgtop.add_residue(atom.name, cgchain)
        cgtop.add_atom('CA', element=md.element.carbon, residue=cgres)
    traj = md.Trajectory(t.xyz, cgtop, t.time, t.unitcell_lengths, t.unitcell_angles)
    traj = traj.superpose(traj, frame=0)
    return traj

@ray.remote(num_cpus=1)
def run_pulchra(cwd, dataset,name,cycle,pulchra_path,i,frame):
    """
    This function runs Pulchra on a single frame.
    """
    rec_name = f'{cwd}/{dataset}/{name}/{cycle}/{i}.pdb'
    frame.save(rec_name)
    FNULL = open(os.devnull, 'w')
    subprocess.run([pulchra_path,rec_name],stdout=FNULL,stderr=FNULL)
    outname = f'{cwd}/{dataset}/{name}/{cycle}/{i}.rebuilt.pdb'
    trajtemp = md.load(outname)
    os.remove(rec_name)  # delete pdbfiles
    os.remove(outname)  # delete pdbfiles
    return trajtemp.xyz

def reconstruct_pulchra(cwd, dataset, name, cycle, pulchra_path,num_cpus):
    """
    This function reconstructs an all-atom trajectory from a Calpha trajectory.
    Input: trajectory nvt.xtc and topology nvt.gro file.
    n_procs: number of processors to use in parallel.
    Return: A reconstructed mdtraj trajectory.
    """
    print("reconstruct_pulchra....", name)
    cycle = int(cycle)
    # prot = pd.read_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl').loc[name]
    t = fix_topology(f'{cwd}/{dataset}/{name}/{cycle}/{name}.dcd',f'{cwd}/{dataset}/{name}/{cycle}/{name}.pdb')
    first_name = f'{cwd}/{dataset}/{name}/{cycle}/0.pdb'
    t[0].save(first_name)
    subprocess.run([pulchra_path,first_name])
    s = md.load_pdb(f'{cwd}/{dataset}/{name}/{cycle}/0.rebuilt.pdb')
    # n_blocks = t.n_frames // num_cpus
    xyz = np.empty((0,s.n_atoms,3))
    xyz = np.append( xyz, s.xyz )
    # it seems that num_cpus could be increased to speed up, but increasing num_cpus will impair performance
    num_cpus = num_cpus - 1
    for j in range(1, t.n_frames, num_cpus):
        n = j+num_cpus if j+num_cpus<t.n_frames else t.n_frames
        # extract every frame and assign their positions to modified topology
        xyz = np.append( xyz, np.vstack(ray.get([run_pulchra.remote(cwd, dataset,name,cycle,pulchra_path,i,t[i]) for i in range(j,n)])))
    allatom0 = md.Trajectory(xyz.reshape(t.n_frames,s.n_atoms,3), s.top, t.time, t.unitcell_lengths, t.unitcell_angles)
    top = md.Topology()
    chain = top.add_chain()
    for residue in allatom0.top.residues:
        res = top.add_residue(residue.name, chain, resSeq=residue.index+1)
        for atom in residue.atoms:
            top.add_atom(atom.name, element=atom.element, residue=res)
    allatom1 = md.Trajectory(allatom0.xyz, top, t.time, t.unitcell_lengths, t.unitcell_angles)
    allatom1.save_dcd(f'{cwd}/{dataset}/{name}/{cycle}/allatom.dcd')
    allatom1[0].save_pdb(f'{cwd}/{dataset}/{name}/{cycle}/allatom.pdb')
    print(name,'has',allatom1.n_frames,'frames', "done")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cwd', dest='cwd', type=str)
    parser.add_argument('--name', dest='name', type=str)
    parser.add_argument('--dataset', dest='dataset', type=str)
    parser.add_argument('--cycle', dest='cycle', type=int)
    parser.add_argument('--pulchra', dest='pulchra_path', type=str)
    parser.add_argument('--num_cpus', dest='num_cpus', type=int)
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus)

    starttime = time.time()  # begin timer
    reconstruct_pulchra(args.cwd, args.dataset, args.name, args.cycle, args.pulchra_path, args.num_cpus)
    endtime = time.time()  # end timer
    target_seconds = endtime - starttime  # total used time
    print(f"{args.name} total pulchra used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")
