import matplotlib.pyplot as plt
from protein_repo import *
import mdtraj as md
cwd = "/home/fancao/CALVADOSCOM"
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
import matplotlib as mpl
label_size = 12
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=5.5)  

def figS7_1(ax1):
    cwd = "/home/fancao/CALVADOSCOM"
    dataset = "CALVADOS2COM_2.0_0.05_1_validate"  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    cycle = 0
    MultiDomainsRgs = pd.read_pickle(f"{cwd}/{dataset}/MultiDomainsRgs_test.pkl")
    allmultidomain_names = list(MultiDomainsRgs.index)
    ax1.set_title("")
    ax1.set_ylabel("Protein domains")
    ax1.set_xlabel("$R_{g,sim}$ [nm]")

    # rgarrays = []
    xticklabels = []
    count = 1
    num_domain = 0
    for name in allmultidomain_names:
        if "@"in name:
            continue
        print(name)
        ssdomains = get_ssdomains(name, f'{cwd}/domains.yaml')
        for ssdomain_idx in range(len(ssdomains)):
            num_domain += 1
            # change hnRNPA1S and hSUMO_hRNPA1S:
            if name in ["hnRNPA1S", "hSUMO_hnRNPA1S"]:
                name_changed = name[:-1] + '*'
            else:
                name_changed = name
            xticklabels.append(f"{name_changed}/D{ssdomain_idx}")
            rgarrays = []
            if not os.path.isfile(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy"):
                df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle - 1}.csv').set_index('three')
                t = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle}/{name}.dcd", f"{cwd}/{dataset}/{name}/{cycle}/{name}.pdb")  # nm
                residues = [res.name for res in t.top.atoms]
                masses = df.loc[residues, 'MW'].values
                masses[0] += 2
                masses[-1] += 16
                print("calculating Rg..........")
                ssdomain = ssdomains[ssdomain_idx]
                mask = np.zeros(len(residues), dtype=bool)  # used to filter residues within domains
                # ssdomains = np.array(sum(ssdomains, []))  # the numbe of residues within domains
                mask[np.array(ssdomain) - 1] = True  # the number of residues, (N,)
                # calculate the center of mass
                # print(t.xyz.shape)  (N_traj, N_res, 3)
                filter_traj = np.array([traj[mask] for traj in t.xyz])  # (N_traj, N_res-N_notdomain, 3)
                filter_masses = masses[mask]
                cm = np.sum(filter_traj * filter_masses[np.newaxis, :, np.newaxis],
                            axis=1) / filter_masses.sum()  # (N_traj, 3)

                # calculate residue-cm distances
                si = np.linalg.norm(filter_traj - cm[:, np.newaxis, :], axis=2)
                # calculate rg
                rgarray = np.sqrt(np.sum(si ** 2 * filter_masses, axis=1) / filter_masses.sum())
                np.save(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy", rgarray)
            else:
                rgarray = np.load(f"{cwd}/{dataset}/{name}/{cycle}/Rg_multidomain_{ssdomain_idx}.npy")
            rgarrays += rgarray.tolist()
            print(len(rgarrays))
            ax1.scatter(np.mean(rgarrays), count, s=s)
            ax1.errorbar(np.mean(rgarrays), count, xerr=np.std(rgarrays), elinewidth=1, capsize=3, capthick=1, color='k')
            count += 1
    plt.setp(ax1, yticks=[i + 1 for i in range(len(xticklabels))], yticklabels=xticklabels)
    # plt.legend(fontsize=20, markerscale=6)
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0.3, num_domain+0.6)

fig = plt.figure(figsize=(4,6), dpi=600)
ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)


figS7_1(ax1)

plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/figS7.pdf', bbox_inches='tight')

