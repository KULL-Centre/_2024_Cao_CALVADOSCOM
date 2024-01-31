import numpy as np
from color_setting import *
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
import sys
sys.path.append('../BLOCKING')
from main import BlockAnalysis
import matplotlib.pyplot as plt
from protein_repo import *

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
cwd = "/home/fancao/CALVADOSCOM"
import matplotlib as mpl

label_size = 12
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=7)

def calcProfiles_rc(cwd, dataset, record, cycle, ax):
    # center_slab(cwd, dataset, record, cycle)
    config = yaml.safe_load(open(f"{cwd}/{dataset}/{record}/{cycle}/config_sim.yaml", 'r'))
    L = config["L"]
    Lz = config["Lz"]
    model = record
    value = pd.DataFrame(index=[record], dtype=object)
    error = value.copy()
    nskip = 1200  # default 1200
    df_proteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl").astype(object)
    proteins = [record]
    wfreq = config['wfreq']
    for i, m in enumerate(proteins):
        if os.path.isfile(f"{cwd}/{dataset}/{record}/{record}.npy"):
            h = np.load(f"{cwd}/{dataset}/{record}/{record}.npy")
            print(h.shape)
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

            save_interval = wfreq * 1E-8  # μs
            im = ax.imshow(h, extent=[-Lz / 2, Lz / 2, nskip * save_interval, h.shape[0] * save_interval], cmap=plt.cm.Blues, origin='lower', norm=LogNorm(vmin=1e-2, vmax=1e2), aspect='auto')


            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size="5%", pad=.15)
            fig.add_axes(cax)
            if record in ['hnRNPA1S', 'hSUMO_hnRNPA1S']:
                name_changed = record[:-1] + "*"
            else:
                name_changed = record
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", label=name_changed)
            cb.ax.tick_params()
            cax.xaxis.set_label_position('top')
            cax.xaxis.set_ticks_position('top')

            print(value)
            os.system(f"rm {cwd}/{dataset}/{record}/{record}.npy")
            return value.loc[m, model + '_dil']
        else:
            print('DATA NOT FOUND FOR', m, model)
            return None


variants = ['A1', 'M12FP12Y', 'P7FM7Y', 'M9FP6Y', 'M8FP4Y', 'M9FP3Y', 'P23GM23SM12FP12Y', 'M10R', 'M6R', 'P2R', 'P7R',
    'M3RP3K', 'M10GP10S', 'M20GP20S', 'P23GM23SP7FM7Y', 'M6RP6K', 'M10RP10K', 'M4D', 'P4D', 'P8D', 'P12D', 'P12E',
    'P23GM23S', 'M30GP30S', 'P7KP12D', 'P7KP12Db', 'M12FP12YM10R', 'M10FP7RP12D', 'M14NP14Q', 'M23SP23T', ]
# 'LAF1', 'LAF1D2130', 'LAF1shuf', 'A2', 'FUS', 'Ddx4WT', 'A1S150', 'A1S200', 'A1S300', 'A1S500'
IDPs = False
fig = plt.figure(figsize=(7, 7), dpi=600)
model = "slabCALVADOSCOMcutoff2.2_0.08_2_COM_cutoff2.0_1"
# fig.suptitle(model)
if IDPs:
    df_csat = pd.read_csv(f"{cwd}/csat_calvados2_test.csv").set_index("Unnamed: 0")
    rows = 6
    columns = 6
else:
    df_csat = pd.read_csv(f"{cwd}/csat_MDPs_test.csv").set_index("Unnamed: 0")
    rows = 3
    columns = 3
ax_list = []
for row in range(rows):
    for column in range(columns):
        ax_list.append(plt.subplot2grid((rows, columns), (row, column), rowspan=1, colspan=1))

models_dic = {model: {model: list(df_csat.index)}, }
cycle = 0
i = 0

for model in models_dic.keys():
    for dataset in models_dic[model].keys():
        for record in models_dic[model][dataset]:
            if record in ["hSUMO_TIA1PrLD"]:
                continue
            calcProfiles_rc(cwd, dataset, record, cycle, ax_list[i]) * 1000
            i += 1

ax_list[-1].set_axis_off()

for i in [0, 3, 6]:
    ax_list[i].set_ylabel('$t$ [µs]')

for i in [-1, -2, -3]:
    ax_list[i].set_xlabel('$z$ [nm]')

plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/figS6.pdf', bbox_inches='tight')
