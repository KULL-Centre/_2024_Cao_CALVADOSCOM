import itertools
import sys
from PIL import Image
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append('..')
from utils import load_parameters, calcAHenergy, determineMask
from color_setting import *
import mdtraj as md
import matplotlib.pyplot as plt
from protein_repo import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
cwd = "/home/fancao/CALVADOSCOM"
aa = ["GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "TYR", "ASP", "ASN",
      "GLU", "LYS", "GLN", "MET", "SER", "THR", "CYS", "PRO", "HIS", "ARG",
      "HID", "ASH", "HIE", "HIP", "HSD", "HSE", "HSP"]  # irregular aa

element_mass_dict = {"C": 12, "N": 14, "O": 16, "H": 1, "S": 32}
electron_num_dict = {"C": 6, "N": 7, "O": 8, "H": 1, "S": 16}

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

import matplotlib as mpl
text_size = 6
label_size = text_size*1.5
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=text_size)

def pairsE(traj, pairs, AApairs, df, rc, mask):
    loops = 20
    expand_mask = np.tile(mask, (int(len(traj)/loops),1))
    AllpairsEtraj = []
    HALR = lambda r, s, l: 4 * 0.8368 * l * ((s / r) ** 12 - (s / r) ** 6)
    HASR = lambda r, s, l: 4 * 0.8368 * ((s / r) ** 12 - (s / r) ** 6) + 0.8368 * (1 - l)
    HA = lambda r, s, l: np.where(r < 2 ** (1 / 6) * s, HASR(r, s, l), HALR(r, s, l))
    HASP = lambda r, s, l, rc: np.where(r < rc, HA(r, s, l) - HA(rc, s, l), 0)
    d = md.compute_distances(traj, pairs)  # distances between pairs for each frame
    sigmas = 0.5 * (df.loc[AApairs[:, 0]].sigmas.values + df.loc[AApairs[:, 1]].sigmas.values)
    lambdas = 0.5 * (df.loc[AApairs[:, 0]].lambdas.values + df.loc[AApairs[:, 1]].lambdas.values)
    emap = np.zeros(AApairs.shape[0])
    for i, r in enumerate(np.split(d, loops, axis=0)):
        tmp = HASP(r, sigmas[np.newaxis, :], lambdas[np.newaxis, :], rc)  # (*, n*n)
        tmp[~expand_mask] = 0
        AllpairsEtraj += np.nansum(tmp, axis=1).tolist()
        emap += np.nansum(tmp, axis=0)
    emap_ave = emap / d.shape[0]
    return pairs, emap_ave, AllpairsEtraj

def fig2_1(ax1, ax2):
    # single chain, ax1
    cwd = "/home/fancao/CALVADOSCOM"
    dataset = "CALVADOS2CA_2.0_0.05_1_validate"
    initial_type = "C2"  # C2, ...
    record = "GS0"
    name = "GS0"
    cycle = 0  # production runs
    rc = 2.0  # nm
    if not os.path.isfile(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy"):
        multidomain_names = list(pd.read_pickle(f'{cwd}/{dataset}/MultiDomainsRgs.pkl').astype(object).index)
        df = load_parameters(cwd, dataset, cycle, "calvados_version", initial_type).set_index("three")
        traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd",
                           f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")
        fasta = [res.name for res in traj.top.atoms]
        fdomains = f"{cwd}/{dataset}/domains.yaml"
        pairs = traj.top.select_pairs('all', 'all')  # index starts with 0
        mask = determineMask(name, record, multidomain_names, fasta, fdomains, pairs)
        AApairs = np.array(list(itertools.combinations(fasta, 2)))
        pairs, emap_ave, AllpairsEtraj = pairsE(traj, pairs, AApairs, df, rc, mask)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/pairs.npy", pairs)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy", emap_ave)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/AllpairsEtraj.npy", AllpairsEtraj)
    else:
        pairs = np.load(f"{cwd}/{dataset}/{record}/{cycle}/pairs.npy")
        emap_ave = np.load(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy")
        # total AH energy for each frame
        AllpairsEtraj = np.load(f"{cwd}/{dataset}/{record}/{cycle}/AllpairsEtraj.npy")

    E_matrix = np.zeros(shape=(np.max(pairs)+1, np.max(pairs)+1))
    for i, e in enumerate(emap_ave):
        row = pairs[i][0]
        col = pairs[i][1]
        E_matrix[row][col] = e
    max_v = np.max(np.abs(E_matrix))
    print(E_matrix.shape)

    cmp = mpl.colors.ListedColormap(['#0500cf','#0f0fff','#6d6dff','w','#fd6c6d','#fd1010','#de0102'])
    norm = mpl.colors.BoundaryNorm([-0.33, -0.2, -0.1, -0.015, 0.015, 0.1, 0.2, 0.33], cmp.N)
    im1 = ax1.imshow(E_matrix, norm=norm, cmap=cmp, origin='lower', aspect='equal')
    divider = make_axes_locatable(ax1)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(im1, cax=ax_cb)

    # single chain, ax2
    cwd = "/home/fancao/CALVADOSCOM"
    dataset = "CALVADOS2COM_2.0_0.05_1_validate"
    initial_type = "C2"  # C2, ...
    record = "GS0"
    name = "GS0"
    cycle = 0  # production runs
    rc = 2.0  # nm
    if not os.path.isfile(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy"):
        multidomain_names = list(pd.read_pickle(f'{cwd}/{dataset}/MultiDomainsRgs.pkl').astype(object).index)
        df = load_parameters(cwd, dataset, cycle, "calvados_version", initial_type).set_index("three")
        traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd", f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")
        fasta = [res.name for res in traj.top.atoms]
        fdomains = f"{cwd}/{dataset}/domains.yaml"
        pairs = traj.top.select_pairs('all', 'all')  # index starts with 0
        mask = determineMask(name, record, multidomain_names, fasta, fdomains, pairs)
        AApairs = np.array(list(itertools.combinations(fasta, 2)))
        pairs, emap_ave, AllpairsEtraj = pairsE(traj, pairs, AApairs, df, rc, mask)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/pairs.npy", pairs)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy", emap_ave)
        np.save(f"{cwd}/{dataset}/{record}/{cycle}/AllpairsEtraj.npy", AllpairsEtraj)
    else:
        pairs = np.load(f"{cwd}/{dataset}/{record}/{cycle}/pairs.npy")
        emap_ave = np.load(f"{cwd}/{dataset}/{record}/{cycle}/emap_ave.npy")
        # total AH energy for each frame
        AllpairsEtraj = np.load(f"{cwd}/{dataset}/{record}/{cycle}/AllpairsEtraj.npy")

    E_matrix = np.zeros(shape=(np.max(pairs) + 1, np.max(pairs) + 1))
    for i, e in enumerate(emap_ave):
        row = pairs[i][0]
        col = pairs[i][1]
        E_matrix[row][col] = e
    max_v = np.max(np.abs(E_matrix))
    im2 = ax2.imshow(E_matrix, norm=norm, cmap=cmp, origin='lower', aspect='equal')
    divider = make_axes_locatable(ax2)
    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    plt.colorbar(im2, cax=ax_cb)

    ax1.set_xlabel("Residue number")
    ax1.set_ylabel("Residue number")
    ax1.set_xlim(200, 469)
    ax1.set_ylim(0, 269)
    ax2.set_xlabel("Residue number")
    ax2.set_ylabel("Residue number")
    ax2.set_xlim(200, 469)
    ax2.set_ylim(0, 269)
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax1.set_title("CALVADOS 2+$\\rm{C_α}$")
    ax2.set_title("CALVADOS 2+COM")

def fig2_2(ax3):
    cwd = "/home/fancao/CALVADOSCOM"
    sticky_png = Image.open(f'{cwd}/paper_multidomainCALVADOS/fig2_sticky.png')
    ax3.set_xlim(0, sticky_png.width)
    ax3.set_ylim(sticky_png.height, 0)
    ax3.axis('off')
    ax3.imshow(sticky_png)
 
fig = plt.figure(figsize=(6.5,5), dpi=600)
ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2,2), (0,1), rowspan=2, colspan=1)
fig2_2(ax3)
fig2_1(ax1, ax2)

fig.text(0.055, 0.97, 'A', fontsize=label_size)
fig.text(0.055, 0.47, 'B', fontsize=label_size)
fig.text(0.555, 0.97, 'C', fontsize=label_size)
fig.text(0.745, 0.64, '$\\rm{C_α}$', fontsize=label_size)
fig.text(0.745, 0.38, 'COM', fontsize=label_size)

plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/fig2.pdf', bbox_inches='tight')
