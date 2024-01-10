from color_setting import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
sys.path.append('../BLOCKING')
from main import BlockAnalysis
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from protein_repo import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display 
np.set_printoptions(threshold=np.inf)
cwd = "/home/fancao/CALVADOSCOM"
import matplotlib as mpl
text_size = 6
label_size = text_size*1.5
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=text_size)

def fig5_2(ax_list):
    datasets = ["slabC2_CA_cutoff2.0_1", "slabCALVADOSCOMcutoff2.2_0.08_2_COM_cutoff2.0_1"]
    record = "hnRNPA1S"
    for dataset_idx, dataset in enumerate(datasets):
        cmap = np.load(f"{cwd}/{dataset}/{record}/cmap_slabdense.npy")
        max_v = np.max(np.abs(cmap))
        im1 = ax_list[dataset_idx].imshow(cmap, cmap=plt.cm.get_cmap('bwr'), origin='lower', aspect='equal', vmin=-max_v, vmax=max_v)
        ax_list[dataset_idx].set_xlabel("Chains in condensates")
        ax_list[dataset_idx].set_ylabel("Central chain")
        # ax_list[dataset_idx].set_title(label_dict[dataset], fontsize=label_fontsize)
        ax_list[dataset_idx].tick_params(axis='x')
        ax_list[dataset_idx].tick_params(axis='y')
        divider = make_axes_locatable(ax_list[dataset_idx])
        cax = divider.new_vertical(size="5%", pad=.15)
        fig.add_axes(cax)
        cb = fig.colorbar(im1, cax=cax, orientation="horizontal")
        cb.ax.tick_params()
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_ticks_position('top')

def fig5_1(ax_list):
    cwd = "/home/fancao/CALVADOSCOM"
    datasets_dict = {
        "slabC2_CA_cutoff2.0_1": ("hnRNPA1S", 0),
        "slabCALVADOSCOMcutoff2.2_0.08_2_COM_cutoff2.0_1": ("hnRNPA1S", 0),
    }
    CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"
    label_dict = {"slabC2_CA_cutoff2.0_1": "CALVADOS 2+$\\rm{C_α}$",
        "slabCALVADOSCOMcutoff2.2_0.08_2_COM_cutoff2.0_1": CALVADOS_COM}
    for dataset_idx, dataset in enumerate(list(datasets_dict.keys())):
        record = datasets_dict[dataset][0]
        cycle = datasets_dict[dataset][1]  
        # center_slab(cwd, dataset, record, cycle)
        config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml", 'r'))
        L = config["L"]
        model = record
        value = pd.DataFrame(index=[record], dtype=object)
        error = value.copy()
        nskip = 1200
        df_proteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl").astype(object)
        proteins = [record]
        for i, m in enumerate(proteins):
            if os.path.isfile(f"{cwd}/{dataset}/{record}/{record}.npy"):
                h = np.load(f"{cwd}/{dataset}/{record}/{record}.npy")
                fasta = df_proteins.loc[m].fasta
                N = len(fasta)
                conv = 1 / 6.022 / N / L / L / 0.1 * 1e4
                h = h[nskip:] * conv  # mM/L of every bin
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

                bool1 = np.logical_and(z < cutoffs1[0], z > cutoffs1[1])  # dense region
                bool2 = np.logical_or(z > cutoffs2[0], z < cutoffs2[1])  # dilute region

                dilarray = np.apply_along_axis(lambda a: a[
                    bool2].mean(), 1, h)  # averaged over dilute region, (n_raj-nskip,)
                denarray = np.apply_along_axis(lambda a: a[
                    bool1].mean(), 1, h)  # averaged over dense region, (n_raj-nskip,)
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
                ax_list[dataset_idx].plot(z, hm * 1000)
                for c1, c2 in zip(cutoffs1, cutoffs2):
                    ax_list[dataset_idx].axvline(c1, color='gray')
                    ax_list[dataset_idx].axvline(c2, color='black')
                ax_list[dataset_idx].set_ylabel('Concentration [μM]')
                ax_list[dataset_idx].set_xlabel('z [nm]')
                ax_list[dataset_idx].set_yscale('log')
                Csat_text = "$c_{sat}$"
                ax_list[
                    dataset_idx].set_title(f"{label_dict[dataset]}\n{Csat_text}: {np.round(value.loc[m, model + '_dil'] * 1000, 1)}\u00B1{np.round(error.loc[m, model + '_dil'] * 1000, 1)} μM")
            else:
                print('DATA NOT FOUND FOR', m, model)
        print(value)
        print(error)


fig = plt.figure(figsize=(5,5), dpi=600)
ax1 = plt.subplot2grid((2,2), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2,2), (0,1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=1)
ax4 = plt.subplot2grid((2,2), (1,1), rowspan=1, colspan=1)

fig5_1([ax1, ax2])
fig5_2([ax3, ax4])


fig.text(0.04, 0.95, 'A', fontsize=label_size)
fig.text(0.54, 0.95, 'B', fontsize=label_size)
fig.text(0.04, 0.45, 'C', fontsize=label_size)
fig.text(0.54, 0.45, 'D', fontsize=label_size)


plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/fig5.pdf', bbox_inches='tight')