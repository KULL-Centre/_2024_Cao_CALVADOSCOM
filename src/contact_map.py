from MDAnalysis import transformations
import yaml
import ray
import os
import pandas as pd
import numpy as np
import mdtraj as md
import itertools
from argparse import ArgumentParser
from scipy.optimize import least_squares
import time

HALR = lambda r,s,l : 4*0.8368*l*((s/r)**12-(s/r)**6)
HASR = lambda r,s,l : 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)
HA = lambda r,s,l : np.where(r<2**(1/6)*s, HASR(r,s,l), HALR(r,s,l))
HASP = lambda r,s,l,rc : np.where(r<rc, HA(r,s,l)-HA(rc,s,l), 0)

DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)

def calc_zpatch(z,h):
    cutoff = np.min(h)  # original value is 0, but take the minimum should be okay to avoid ZeroDivisionError;
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

def center_slab(cwd, dataset, record, cycle):
    u = MDAnalysis.Universe(f'{cwd}/{dataset}/{record}/{cycle}/top_0.pdb',  # Å
        f'{cwd}/{dataset}/{record}/{cycle}/traj_pool.dcd',in_memory=True)
    n_frames = len(u.trajectory)
    ag = u.atoms
    n_atoms = ag.n_atoms
    lz = u.dimensions[2]  # Å
    print("lz: ", lz, "Å")
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    with MDAnalysis.Writer(f'{cwd}/{dataset}/{record}/{cycle}/wrapped.dcd',n_atoms) as W:
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
    np.save(f'{cwd}/{dataset}/{record}/{cycle}/{record}.npy',hs,allow_pickle=False)
    return hs, z

def calc_energies(t, chain_index_1, chain_index_2, sigmas, lambdas, yukawa_eps, lD, Naa):
    sel1 = t.top.select('chainid {:d}'.format(chain_index_1))  # atom index, starts from 0
    sel2 = t.top.select('chainid {:d}'.format(chain_index_2))
    pairs_indices = t.top.select_pairs(sel1,sel2)
    d = md.compute_distances(t,pairs_indices).reshape(t.n_frames,Naa,Naa)  # (n_frames, chain1(row), otherchain(columns))
    ah_ene = HASP(d,sigmas,lambdas,2.0)  # checked
    dh_ene = DHSP(d,yukawa_eps,lD,4.0)
    switch_2 = (.5-.5*np.tanh((d-sigmas)/.2))
    switch_3 = (.5-.5*np.tanh((d-sigmas)/.3))
    return ah_ene, dh_ene, switch_2, switch_3

def calc_cm_rg(t,masses):
    chain_cm = (np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()).astype(np.float16)
    si = np.linalg.norm(t.xyz - chain_cm[:,np.newaxis,:],axis=2).astype(np.float16)
    chain_rg = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum()).astype(np.float16)
    chain_ete = md.compute_distances(t, [[0,t.n_atoms-1]]).flatten()
    return chain_cm, chain_rg, chain_ete

def calcWidth(cwd, dataset, record):
    # this function finds the z-positions that delimit the slab and the dilute phase
    h = np.load(f'{cwd}/{dataset}/{record}/{record}.npy',allow_pickle=False)
    lz = (h.shape[1]+1)
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h[100:],axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[hm.min(),hm.max(),3,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[1e3]*4))
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[1e3]*4))
    cutoff1 = .5*(np.abs(res1.x[2]-.5*res1.x[3])+np.abs(-res2.x[2]+.5*res2.x[3]))
    cutoff2 = .5*(np.abs(res1.x[2]+6*res1.x[3])+np.abs(-res2.x[2]-6*res2.x[3]))
    return cutoff1, cutoff2, z, edges

def cal_E(cwd, dataset, record, cycle, prot, t, begin, end, df, cutoff1, cutoff2, z, edges, temp, chunk):
    Naa = len(prot.fasta)
    masses = df.loc[prot.fasta, 'MW'].values
    masses[0] += 2
    masses[-1] += 16
    radii = df.loc[prot.fasta, 'sigmas'].values / 2
    n_chains = int(t.n_atoms / Naa)
    t = t[begin:end]
    print(t.n_frames)

    h_res = np.zeros((Naa + 1, edges.size - 1))
    cm_z = np.empty(0)
    indices = np.zeros((n_chains, t.n_frames))
    middle_dist = np.zeros((n_chains, t.n_frames))

    for i in range(n_chains):
        t_chain = t.atom_slice(t.top.select('chainid {:d}'.format(i)))
        chain_cm, chain_rg, chain_ete = calc_cm_rg(t_chain, masses)
        mask_in = np.abs(chain_cm[:, 2]) < cutoff1
        mask_out = np.abs(chain_cm[:, 2]) > cutoff2
        cm_z = np.append(cm_z, chain_cm[:, 2])
        indices[i] = mask_in
        middle_dist[i] = np.abs(chain_cm[:, 2])

    middle_chain = np.argmin(middle_dist, axis=0)  # indices of chains at the center of the slab
    del middle_dist

    fasta = prot.fasta
    df.loc['H', 'q'] = 1. / (1 + 10 ** (prot.pH - 6))
    df.loc['X'] = df.loc[fasta[0]]
    df.loc['Z'] = df.loc[fasta[-1]]
    df.loc['X', 'q'] = df.loc[prot.fasta[0], 'q'] + 1.
    df.loc['Z', 'q'] = df.loc[prot.fasta[-1], 'q'] - 1.
    fasta[0] = 'X'
    fasta[-1] = 'Z'

    pairs = np.asarray(list(itertools.product(fasta, fasta)))
    sigmas = 0.5 * (df.loc[pairs[:, 0]].sigmas.values + df.loc[pairs[:, 1]].sigmas.values).reshape(Naa, Naa)
    lambdas = 0.5 * (df.loc[pairs[:, 0]].lambdas.values + df.loc[pairs[:, 1]].lambdas.values).reshape(Naa, Naa)

    RT = 8.3145 * temp * 1e-3
    fepsw = lambda T: 5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T * T - 0.8292 * 1e-6 * T ** 3
    epsw = fepsw(temp)
    lB = 1.6021766 ** 2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / RT
    qq = (df.loc[pairs[:, 0]].q.values * df.loc[pairs[:, 1]].q.values).reshape(Naa, Naa)
    yukawa_eps = qq * lB * RT
    lD = 1. / np.sqrt(8 * np.pi * lB * prot.ionic * 6.022 / 10)

    ah_mat = np.zeros((t.n_frames, Naa, Naa))
    dh_mat = np.zeros((t.n_frames, Naa, Naa))
    s2_mat = np.zeros((t.n_frames, Naa, Naa))
    s3_mat = np.zeros((t.n_frames, Naa, Naa))
    print(np.unique(middle_chain))
    for chain_1 in np.unique(middle_chain):
        print(chain_1)
        for chain_2 in np.setdiff1d(np.arange(n_chains), [chain_1]):
            ndx = ((middle_chain == chain_1) * indices[chain_2]).astype(bool)  # 'and' operation
            if np.any(ndx):
                print(chain_2)
                # t[ndx] means frames set whose has chain_1 as the middle chain and {chain_2} within cutoff1
                ah_ene, dh_ene, s_2, s_3 = calc_energies(t[ndx], chain_1, chain_2, sigmas, lambdas, yukawa_eps, lD, Naa)
                ah_mat[ndx, :] += ah_ene
                dh_mat[ndx, :] += dh_ene
                s2_mat[ndx, :] += s_2
                s3_mat[ndx, :] += s_3

    # save energy and contact maps
    if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}/s_mat"):
        os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}/s_mat")
    np.save(f'{cwd}/{dataset}/{record}/{cycle}/s_mat/{record}_chunk{chunk}_ah_mat.npy', ah_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/{cycle}/s_mat/{record}_chunk{chunk}_dh_mat.npy', dh_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/{cycle}/s_mat/{record}_chunk{chunk}_s2_mat.npy', s2_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/{cycle}/s_mat/{record}_chunk{chunk}_s3_mat.npy', s3_mat.mean(axis=0))

@ray.remote(num_cpus=1)
def cal_E_MPI(cwd, dataset, record, cycle, prot, t, df, cutoff1, cutoff2, z, edges, temp, chunk):
    Naa = len(prot.fasta)
    masses = df.loc[prot.fasta, 'MW'].values
    masses[0] += 2
    masses[-1] += 16
    radii = df.loc[prot.fasta, 'sigmas'].values / 2
    n_chains = int(t.n_atoms / Naa)
    # print(t.n_frames)
    h_res = np.zeros((Naa + 1, edges.size - 1))
    cm_z = np.empty(0)
    indices = np.zeros((n_chains, t.n_frames))
    middle_dist = np.zeros((n_chains, t.n_frames))
    # print("n_chains:", n_chains)
    for i in range(n_chains):
        t_chain = t.atom_slice(t.top.select('chainid {:d}'.format(i)))
        chain_cm, chain_rg, chain_ete = calc_cm_rg(t_chain, masses)
        mask_in = np.abs(chain_cm[:, 2]) < cutoff1
        mask_out = np.abs(chain_cm[:, 2]) > cutoff2
        cm_z = np.append(cm_z, chain_cm[:, 2])
        indices[i] = mask_in
        middle_dist[i] = np.abs(chain_cm[:, 2])

    middle_chain = np.argmin(middle_dist, axis=0)  # indices of chains at the center of the slab
    del middle_dist

    fasta = prot.fasta
    df.loc['H', 'q'] = 1. / (1 + 10 ** (prot.pH - 6))
    df.loc['X'] = df.loc[fasta[0]]
    df.loc['Z'] = df.loc[fasta[-1]]
    df.loc['X', 'q'] = df.loc[prot.fasta[0], 'q'] + 1.
    df.loc['Z', 'q'] = df.loc[prot.fasta[-1], 'q'] - 1.
    fasta[0] = 'X'
    fasta[-1] = 'Z'

    pairs = np.asarray(list(itertools.product(fasta, fasta)))
    sigmas = 0.5 * (df.loc[pairs[:, 0]].sigmas.values + df.loc[pairs[:, 1]].sigmas.values).reshape(Naa, Naa)
    lambdas = 0.5 * (df.loc[pairs[:, 0]].lambdas.values + df.loc[pairs[:, 1]].lambdas.values).reshape(Naa, Naa)

    RT = 8.3145 * temp * 1e-3
    fepsw = lambda T: 5321 / T + 233.76 - 0.9297 * T + 0.1417 * 1e-2 * T * T - 0.8292 * 1e-6 * T ** 3
    epsw = fepsw(temp)
    lB = 1.6021766 ** 2 / (4 * np.pi * 8.854188 * epsw) * 6.022 * 1000 / RT
    qq = (df.loc[pairs[:, 0]].q.values * df.loc[pairs[:, 1]].q.values).reshape(Naa, Naa)
    yukawa_eps = qq * lB * RT
    lD = 1. / np.sqrt(8 * np.pi * lB * prot.ionic * 6.022 / 10)

    ah_mat = np.zeros((t.n_frames, Naa, Naa))
    dh_mat = np.zeros((t.n_frames, Naa, Naa))
    s2_mat = np.zeros((t.n_frames, Naa, Naa))
    s3_mat = np.zeros((t.n_frames, Naa, Naa))
    # print(np.unique(middle_chain))  # chains that could be used as central chain
    for chain_1 in np.unique(middle_chain):
        # print(chain_1)
        for chain_2 in np.setdiff1d(np.arange(n_chains), [chain_1]):
            ndx = ((middle_chain == chain_1) * indices[chain_2]).astype(bool)
            # print(middle_chain == chain_1, indices[chain_2], np.any(ndx))
            if np.any(ndx):
                ah_ene, dh_ene, s_2, s_3 = calc_energies(t[ndx], chain_1, chain_2, sigmas, lambdas, yukawa_eps, lD, Naa)
                ah_mat[ndx, :] += ah_ene
                dh_mat[ndx, :] += dh_ene
                s2_mat[ndx, :] += s_2
                s3_mat[ndx, :] += s_3

    # save energy and contact maps
    if not os.path.isdir(f"{cwd}/{dataset}/{record}/s_mat"):
        os.system(f"mkdir -p {cwd}/{dataset}/{record}/s_mat")
    np.save(f'{cwd}/{dataset}/{record}/s_mat/{record}_chunk{chunk}_ah_mat.npy', ah_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/s_mat/{record}_chunk{chunk}_dh_mat.npy', dh_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/s_mat/{record}_chunk{chunk}_s2_mat.npy', s2_mat.mean(axis=0))
    np.save(f'{cwd}/{dataset}/{record}/s_mat/{record}_chunk{chunk}_s3_mat.npy', s3_mat.mean(axis=0))

def analyse_traj(cwd, dataset, record, cycle, temp):
    skip = 1200
    num_cpus = 20
    config_sim = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml", 'r'))
    n_chains = config_sim["chains"]
    df = pd.read_csv(f'{cwd}/{dataset}/residues_pub.csv').set_index('three', drop=False).set_index('one')
    allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")
    prot = allproteins.loc[record]
    # center_slab(cwd, dataset, record, cycle)
    # this function finds the index of the chain at the center of the slab for each frame
    cutoff1, cutoff2, z, edges = calcWidth(cwd, dataset, record)
    print(record,cutoff1,cutoff2)
    if not os.path.isfile(f'{cwd}/{dataset}/{record}/top_3letter.pdb'):
        top = md.Topology()
        for _ in range(n_chains):
            chain = top.add_chain()
            for resname in prot.fasta:
                residue = top.add_residue(df.loc[resname,'three'], chain)
                top.add_atom(df.loc[resname,'three'],
                             element=md.element.carbon, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
    else:
        top = md.load_pdb(f'{cwd}/{dataset}/{record}/top_3letter.pdb')
    # the original trajectory (production_0.dcd) contains chains completely out of box
    # wrapped makes sure all atoms are inside of box, so molecules might be broken;
    # make_molecules_whole will make broken molecules in wrapped.dcd intact;
    # all of these are for visualization; because md.compute_distances takes care of PBC;
    if not os.path.isfile(f'{cwd}/{dataset}/{record}/wrapped_make_molecules_whole.dcd'):
        t = md.load_dcd(f'{cwd}/{dataset}/{record}/wrapped.dcd',top)
        t.make_molecules_whole(inplace=True)  # remove translation, very necessary
        t.save_dcd(f'{cwd}/{dataset}/{record}/wrapped_make_molecules_whole.dcd')
    else:
        t = md.load_dcd(f'{cwd}/{dataset}/{record}/wrapped_make_molecules_whole.dcd', top)

    if not os.path.isfile(f'{cwd}/{dataset}/{record}/top_3letter.pdb'):
        t[0].save_pdb(f'{cwd}/{dataset}/{record}/top_3letter.pdb')
    t.xyz -= t.unitcell_lengths[0, :] / 2
    t.save_dcd(f'{cwd}/{dataset}/{record}/wrapped_make_molecules_whole_unitcell_lengths.dcd')
    t = t[skip:]  # keep last part
    # begin = 0
    # end = 3
    # cal_E_MPI(cwd, dataset, record, cycle, prot, t, begin, end, df, cutoff1, cutoff2, z, edges, temp, 0)
    if len(t)%num_cpus != 0:
        raise Exception("num_cpus are not good!")
    else:
        frames_perCPU = int(len(t) / num_cpus)
    ray.init()
    ray.get([cal_E_MPI.remote(cwd, dataset, record, cycle, prot, t[chunk*frames_perCPU:(chunk+1)*frames_perCPU], df, cutoff1, cutoff2, z, edges, temp, chunk) for chunk in range(num_cpus)])
    cmap = []
    for i in range(num_cpus):
        cmap.append(np.load(f"{cwd}/{dataset}/{record}/s_mat/{record}_chunk{i}_ah_mat.npy"))
    cmap = np.array(cmap).mean(axis=0)
    np.save(f"{cwd}/{dataset}/{record}/cmap_slabdense.npy", cmap)



parser = ArgumentParser()
parser.add_argument('--path2config',nargs='?',const='', type=str)
args = parser.parse_args()
config = yaml.safe_load(open(f'{args.path2config}', 'r'))
starttime = time.time()  # begin timer
analyse_traj(config["cwd"],config["dataset"],config["record"],int(config["cycle"]),float(config["temp"]))
endtime = time.time()  # end timer
target_seconds = endtime - starttime  # total used time
print(f"total used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")