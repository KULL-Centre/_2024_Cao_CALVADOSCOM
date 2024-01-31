import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import pearsonr, spearmanr
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

import matplotlib as mpl
label_size = 12
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=7.5)  

def fig4_1(models_dic, ax, df_csat):
    cwd = "/home/fancao/CALVADOSCOM"
    variants = ['A1', 'M12FP12Y', 'P7FM7Y', 'M9FP6Y',
        'M8FP4Y', 'M9FP3Y', 'P23GM23SM12FP12Y',
        'M10R', 'M6R', 'P2R', 'P7R', 'M3RP3K',
        'M10GP10S', 'M20GP20S', 'P23GM23SP7FM7Y',
        'M6RP6K', 'M10RP10K', 'M4D', 'P4D',
        'P8D', 'P12D', 'P12E', 'P23GM23S', 'M30GP30S',
        'P7KP12D', 'P7KP12Db', 'M12FP12YM10R',
        'M10FP7RP12D','M14NP14Q', 'M23SP23T',
    ]
    giulio = ['A1', 'P7FM7Y', 'M12FP12Y', 'M23SP23T',
        'M9FP6Y', 'M14NP14Q', 'M10GP10S', 'M20GP20S',
        'M30GP30S', 'P23GM23S', 'P23GM23SP7FM7Y',
        'P23GM23SM12FP12Y', 'M9FP3Y', 'M8FP4Y',
        'M3RP3K', 'M6R', 'M4D', 'P4D', 'P8D',
        'P2R', 'A1S150', 'A1S200', 'A1S300', 'A1S500',
        'LAF1', 'LAF1shuf', 'LAF1D2130', 'A2', 'FUS', 'Ddx4WT',
        ]
    for model in models_dic.keys():
        Csat_exp = []
        Csat_cal = []
        Csat_exp_A1 = []
        Csat_cal_A1 = []
        Csat_exp_giulio = []
        Csat_cal_giulio = []
        dataset = models_dic[model][0]
        for record in models_dic[model][1]:
            cal_table = np.load(f"{cwd}/{dataset}/{record}/dilarray.npy").mean()*1000
            if record in variants:
                Csat_exp_A1.append(df_csat.loc[record]["csat_exp"] * 1000)  # μM
                Csat_cal_A1.append(cal_table)  # μM
            if record in giulio:
                Csat_exp_giulio.append(df_csat.loc[record]["csat_exp"] * 1000)  # μM
                Csat_cal_giulio.append(cal_table)  # μM
            Csat_exp.append(df_csat.loc[record]["csat_exp"]*1000)  # μM
            Csat_cal.append(cal_table)  # μM
            # print(record, df_csat.loc[record]["csat_exp"]*1000, cal_table, int((np.load(f"{cwd}/{dataset}/{record}/dilarray.npy").shape[0]+1200)/4000))

        df_csat["csat_CALVADOSCOM"] = np.array(Csat_cal)/1000  # mM
        df_csat.to_csv(f"{cwd}/{dataset}/df_csat.csv")
        # coefficient = np.corrcoef(np.log10(Csat_exp), np.log10(Csat_cal))[0][1]
        Csat_exp_A1 = np.array(Csat_exp_A1)
        Csat_cal_A1 = np.array(Csat_cal_A1)
        Csat_exp_giulio = np.array(Csat_exp_giulio)
        Csat_cal_giulio = np.array(Csat_cal_giulio)
        Csat_exp = np.array(Csat_exp)
        Csat_cal = np.array(Csat_cal)
        A1_variant_co = pearsonr(np.log10(Csat_exp_A1), np.log10(Csat_cal_A1+0.01))[0]
        giulio_co = pearsonr(np.log10(Csat_exp_giulio), np.log10(Csat_cal_giulio + 0.01))[0]
        coefficient = pearsonr(np.log10(Csat_exp), np.log10(Csat_cal))[0]
        print("A1_variant_co: ", A1_variant_co)
        print("giulio_co: ", giulio_co)
        print("coefficient: ", coefficient)
        # ax.set_title("$\\rm{CALVADOS_{COM}}$")
        CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"
        ax.scatter(Csat_exp, Csat_cal, label=f"{CALVADOS_COM}\nr={np.round(coefficient,2)}", s=s, linewidths=linewidth, color=blue)
        # ax.scatter(Csat_exp_A1, Csat_cal_A1, label=f"A1_variants, r={np.round(A1_variant_co, 2)}", s=25, linewidths=1.5, color=orange)
        # ax.scatter(Csat_exp_giulio, Csat_cal_giulio, label=f"Giulio's seq, r={np.round(giulio_co, 2)}", s=25, linewidths=1.5, color=orange)
        ax.plot([0.5, 5000], [0.5, 5000], color="black")
        ax.set(xlabel='Experimental $\\rm{c_{sat}}$ [μM]', ylabel='Simulated $\\rm{c_{sat}}$ [μM]')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_aspect('equal')
        ax.legend(edgecolor="white", frameon=False)

def fig4_2(ax2):
    cwd = "/home/fancao/CALVADOSCOM"
    dataset = "slabCALVADOSCOMcutoff2.2_0.08_2_CA_cutoff2.0_1"
    df_csat = pd.read_csv(f"{cwd}/csat_calvados2_test.csv").set_index("Unnamed: 0")
    df_csat_exp = np.array(df_csat["csat_exp"]) * 1000  # μM
    variants = ['A1', 'M12FP12Y', 'P7FM7Y', 'M9FP6Y', 'M8FP4Y', 'M9FP3Y', 'P23GM23SM12FP12Y', 'M10R', 'M6R', 'P2R',
        'P7R', 'M3RP3K', 'M10GP10S', 'M20GP20S', 'P23GM23SP7FM7Y', 'M6RP6K', 'M10RP10K', 'M4D', 'P4D', 'P8D', 'P12D',
        'P12E', 'P23GM23S', 'M30GP30S', 'P7KP12D', 'P7KP12Db', 'M12FP12YM10R', 'M10FP7RP12D', 'M14NP14Q', 'M23SP23T', ]

    df_csat_CALVADOSCOM = []
    Csat_C2_A1 = []
    Csat_cal_A1 = []
    for record in df_csat.index:
        cal_table = np.load(f"{cwd}/{dataset}/{record}/dilarray.npy").mean()
        df_csat_CALVADOSCOM.append(cal_table)  # mM
        if record in variants:
            Csat_C2_A1.append(df_csat.loc[record]["csat_C2"] * 1000)  # μM
            Csat_cal_A1.append(cal_table*1000)  # μM
    df_csat_C2 = np.array(df_csat["csat_C2"]) * 1000  # μM
    df_csat_CALVADOSCOM = np.array(df_csat_CALVADOSCOM) * 1000  # μM
    x = np.array([1, 10000])
    # popt, _ = curve_fit(lambda x, a, b: a * x + b, np.log10(df_csat_C2), np.log10(df_csat_CALVADOSCOM))
    ax2.scatter(df_csat_C2, df_csat_CALVADOSCOM, s=s)
    CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"
    ax2.set(xlabel='CALVADOS 2 [μM]', ylabel=f'{CALVADOS_COM} [μM]')
    ax2.plot([1, 5000], [1, 5000], color="black")
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_aspect('equal')

    fig.text(0.6, 0.895, "$<(c_{sat,sim}-c_{sat,exp})/c_{sat,exp}>$")
    fig.text(0.6, 0.845, f"CALVADOS 2: {np.round(np.mean((df_csat_C2 - df_csat_exp) / df_csat_exp) * 100, 1)}%")
    fig.text(0.6, 0.795, f"{CALVADOS_COM}: {np.round(np.mean((df_csat_CALVADOSCOM - df_csat_exp) / df_csat_exp) * 100,1)}%")

# figsize is independent on fontsize;
# if fontsize is too big, it might cause non-cubic figures
df_csat = pd.read_csv(f"{cwd}/csat_calvados2_test.csv").set_index("Unnamed: 0")
models_dic = {
        "$CALVADOS_{COM}$": ("slabCALVADOSCOMcutoff2.2_0.08_2_CA_cutoff2.0_1", list(df_csat.index)),
    }

fig = plt.figure(figsize=(6,3), dpi=600)
ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)


fig4_1(models_dic, ax1, df_csat)
fig4_2(ax2)


plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/figS4.pdf', bbox_inches='tight')
 