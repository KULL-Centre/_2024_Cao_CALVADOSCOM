import numpy as np
import Bio.SeqUtils
from scipy import optimize

import MDAnalysis as mda
from MDAnalysis.analysis import distances

import yaml

def aa_three_to_one(three_list):
    """ Convert list of e.g. ['Asp','Gly'] --> ['D','G'] """
    three_str = "".join(three_list)
    one_str = Bio.SeqUtils.seq1(three_str)
    one_list = list(one_str)
    return one_list

def calc_dmap(domain0,domain1):
    """ Distance map (nm) for single configuration
    
    Input: Atom groups
    Output: Distance map"""
    dmap = distances.distance_array(domain0.positions, # reference
                                    domain1.positions, # configuration
                                    box=domain0.dimensions) / 10.
    return dmap

def self_distances(N,pos):
    """ Self distance map for matrix of positions 
    
    Input: Matrix of positions
    Output: Self distance map
    """
    dmap = np.zeros((N,N))
    d = distances.self_distance_array(pos)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            dmap[i, j] = d[k]
            dmap[j, i] = d[k]
            k += 1
    return dmap

def calc_cmap(domain0,domain1,cutoff=1.5):
    """ Contact map for single configuration 
    
    Input: Atom groups
    Output: Contact map
    """
    # Cutoff in nm
    dmap = calc_dmap(domain0,domain1)
    cmap = np.where(dmap<cutoff,1,0)
    return(cmap)

def cmap_traj(u,domain0,domain1,cutoff=1.5):
    """ Average number of contacts along trajectory 
    
    Input:
      * Universe
      * Atom groups
    Output:
      * Average contact map
    """
    cmap = np.zeros((len(domain0),len(domain1)))
    for ts in u.trajectory:
        cmap += calc_cmap(domain0,domain1,cutoff)
    cmap /= len(u.trajectory)
    return cmap

def scaling_exp(n,r0,v):
    rh = r0 * n**v
    return rh

def fit_scaling_exp(u,ag,r0=None):
    """ Fit scaling exponent of single chain

    Input:
      * mda Universe
      * atom group
    Output:
      * ij seq distance
      * dij cartesian distance
      * r0
      * v (scaling exponent)
    """
    N = len(ag)
    dmap = np.zeros((N,N))
    for t,ts in enumerate(u.trajectory):
        dmap += calc_dmap(ag,ag)
    dmap /= len(u.trajectory)
    ij = np.arange(N)
    dij = []
    for i in range(N):
        dij.append([])
    for i in ij:
        for j in range(i,N):
            dij[j-i].append(dmap[i,j]) # in nm

    for i in range(N):
        dij[i] = np.mean(dij[i])
    # print(dij)
    dij = np.array(dij)
    # print(ij.shape)
    # print(dij.shape)
    if r0 == None:
        (r0, v), pcov = optimize.curve_fit(scaling_exp,ij,dij)
    else:
        v, pcov = optimize.curve_fit(lambda x, v: scaling_exp(x,r0,v), ij, dij)
        v = v[0]
    return ij, dij, r0, v

def autocorr(x,norm=True):
    y = x.copy()
    if norm:
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / (np.std(y))
    c = np.correlate(x,y,mode='full')
    c = c[len(c)//2:]
    return c

def genParamsLJ(df,record,prot):
    fasta = prot.fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = prot.eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,record,prot,temp,calvados_version):
    kT = 8.3145*temp*1e-3
    fasta = prot.fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    if calvados_version == 4:
        q = 0.75
    else:
        q = 1.0
    
    r.loc['H','q'] = q / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[prot.fasta[0],'q'] + q
    r.loc['Z','q'] = r.loc[prot.fasta[-1],'q'] - q
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa
