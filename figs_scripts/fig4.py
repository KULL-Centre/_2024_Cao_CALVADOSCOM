from fig1 import get_predictions
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
text_size = 7.5
label_size = text_size*1.7
s = 20
linewidth = 1
times = 100
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=text_size)

def fig4_1(ax1, ax2, ax3):
    cwd = "/home/fancao/CALVADOSCOM"
    bootstrapping = True
    # use current lambda values to simulate next cycle
    dataset = "IDPs_MDPsCOM_2.2_0.08_2_validate"
    cycle = 3
    validate = True
    PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
    predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)
    Rg_exp = np.array(predictions.loc[IDP_names + multidomain_names]["expRg"])
    RgIDPs_exp = np.array(predictions.loc[IDP_names]["expRg"])
    RgMDPs_exp = np.array(predictions.loc[multidomain_names]["expRg"])

    Rg_cal = np.array(predictions.loc[IDP_names + multidomain_names]["cal"])
    RgIDPs_cal = np.array(predictions.loc[IDP_names]["cal"])
    RgMDPs_cal = np.array(predictions.loc[multidomain_names]["cal"])

    RgIDPs_bt = np.array([predictions.loc[name, "bt"] for name in IDP_names])  # (n_seq, times)
    RgMDPs_bt = np.array([predictions.loc[name, "bt"] for name in multidomain_names])  # (n_seq, times)
    # Error = np.array(predictions.loc[IDP_names+multidomain_names]["err"])
    RgIDPs_err = np.array(predictions.loc[IDP_names]["expRgErr"])
    RgMDPs_err = np.array(predictions.loc[multidomain_names]["expRgErr"])
    # Rg_exp_vs_Rg_calc
    chi2_Rg_IDPs = np.power((RgIDPs_exp - RgIDPs_cal) / RgIDPs_err, 2)

    chi2_Rg_MDPs = np.power((RgMDPs_exp - RgMDPs_cal) / RgMDPs_err, 2)

    chi2_Rg_IDPs_res = np.power((RgIDPs_exp - RgIDPs_bt.T) / RgIDPs_err, 2).mean(axis=1)
    chi2_Rg_MDPs_res = np.power((RgMDPs_exp - RgMDPs_bt.T) / RgMDPs_err, 2).mean(axis=1)


    corrcoef_RgIDPs = np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_cal.tolist())[0][1]  # r
    corrcoef_RgMDPs = np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_cal.tolist())[0][1]

    corrcoef_RgIDPs_res = []
    corrcoef_RgMDPs_res = []
    for time in range(RgIDPs_bt.shape[-1]):
        corrcoef_RgIDPs_res.append(np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_bt.T[time].tolist())[0][1])
        corrcoef_RgMDPs_res.append(np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_bt.T[time].tolist())[0][1])

    ax1.scatter(RgIDPs_exp, RgIDPs_cal, label=f"IDPs, r={np.round(corrcoef_RgIDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgIDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgIDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_IDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_IDPs_res.std(), 1)}", color=orange, s=s, marker="o")
    ax1.scatter(RgMDPs_exp, RgMDPs_cal, label=f"MDPs, r={np.round(corrcoef_RgMDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgMDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgMDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_MDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_MDPs_res.std(), 1)}", color=blue, s=s, marker="s")
    # coefficient = np.corrcoef(Rg_exp, Rg_cal)[0][1]
    ax1.set_title(CALVADOS_COM)
    ax1.plot([1, 7], [1, 7], color="black", linewidth=linewidth)
    ax1.legend(edgecolor="white", frameon=False)
    ax1.set_ylabel("$R_{g,sim}$ [nm]")
    ax1.set_xlabel("$R_{g,exp}$ [nm]")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')
    ax1.set_aspect('equal')

    # use current lambda values to simulate next cycle
    dataset = "CALVADOS2CA_2.0_0.05_1_validate"
    cycle = -1
    validate = True
    PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
    predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)

    Rg_exp = np.array(predictions.loc[IDP_names + multidomain_names]["expRg"])
    RgIDPs_exp = np.array(predictions.loc[IDP_names]["expRg"])
    RgMDPs_exp = np.array(predictions.loc[multidomain_names]["expRg"])

    Rg_cal = np.array(predictions.loc[IDP_names + multidomain_names]["cal"])
    RgIDPs_cal = np.array(predictions.loc[IDP_names]["cal"])
    RgMDPs_cal = np.array(predictions.loc[multidomain_names]["cal"])

    RgIDPs_bt = np.array([predictions.loc[name, "bt"] for name in IDP_names])  # (n_seq, times)
    RgMDPs_bt = np.array([predictions.loc[name, "bt"] for name in multidomain_names])  # (n_seq, times)
    # Error = np.array(predictions.loc[IDP_names+multidomain_names]["err"])
    RgIDPs_err = np.array(predictions.loc[IDP_names]["expRgErr"])
    RgMDPs_err = np.array(predictions.loc[multidomain_names]["expRgErr"])
    # Rg_exp_vs_Rg_calc
    chi2_Rg_IDPs = np.power((RgIDPs_exp - RgIDPs_cal) / RgIDPs_err, 2)
    chi2_Rg_MDPs = np.power((RgMDPs_exp - RgMDPs_cal) / RgMDPs_err, 2)

    chi2_Rg_IDPs_res = np.power((RgIDPs_exp - RgIDPs_bt.T) / RgIDPs_err, 2).mean(axis=1)
    chi2_Rg_MDPs_res = np.power((RgMDPs_exp - RgMDPs_bt.T) / RgMDPs_err, 2).mean(axis=1)

    corrcoef_RgIDPs = np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_cal.tolist())[0][1]  # r
    corrcoef_RgMDPs = np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_cal.tolist())[0][1]

    corrcoef_RgIDPs_res = []
    corrcoef_RgMDPs_res = []
    for time in range(RgIDPs_bt.shape[-1]):
        corrcoef_RgIDPs_res.append(np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_bt.T[time].tolist())[0][1])
        corrcoef_RgMDPs_res.append(np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_bt.T[time].tolist())[0][1])

    ax2.scatter(RgIDPs_exp, RgIDPs_cal, label=f"IDPs, r={np.round(corrcoef_RgIDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgIDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgIDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_IDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_IDPs_res.std(), 1)}", color=orange, s=s, marker="o")
    ax2.scatter(RgMDPs_exp, RgMDPs_cal, label=f"MDPs, r={np.round(corrcoef_RgMDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgMDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgMDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_MDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_MDPs_res.std(), 1)}", color=blue, s=s, marker="s")
    # coefficient = np.corrcoef(Rg_exp, Rg_cal)[0][1]


    # assert len(allproteins.index) == len(Rg_cal)
    ax2.set_title("CALVADOS 2+$\\rm{C_α}$")
    ax2.plot([1, 7], [1, 7], color="black", linewidth=linewidth)
    ax2.legend(edgecolor="white", frameon=False)
    ax2.set_ylabel("$R_{g,sim}$ [nm]")
    ax2.set_xlabel("$R_{g,exp}$ [nm]")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.set_aspect('equal')

    # use current lambda values to simulate next cycle
    dataset = "CALVADOS2COM_2.0_0.05_1_validate"

    cycle = -1
    validate = True
    PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
    predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)

    Rg_exp = np.array(predictions.loc[IDP_names + multidomain_names]["expRg"])
    RgIDPs_exp = np.array(predictions.loc[IDP_names]["expRg"])
    RgMDPs_exp = np.array(predictions.loc[multidomain_names]["expRg"])

    Rg_cal = np.array(predictions.loc[IDP_names + multidomain_names]["cal"])
    RgIDPs_cal = np.array(predictions.loc[IDP_names]["cal"])
    RgMDPs_cal = np.array(predictions.loc[multidomain_names]["cal"])

    RgIDPs_bt = np.array([predictions.loc[name, "bt"] for name in IDP_names])  # (n_seq, times)
    RgMDPs_bt = np.array([predictions.loc[name, "bt"] for name in multidomain_names])  # (n_seq, times)
    # Error = np.array(predictions.loc[IDP_names+multidomain_names]["err"])
    RgIDPs_err = np.array(predictions.loc[IDP_names]["expRgErr"])
    RgMDPs_err = np.array(predictions.loc[multidomain_names]["expRgErr"])
    # Rg_exp_vs_Rg_calc
    chi2_Rg_IDPs = np.power((RgIDPs_exp - RgIDPs_cal) / RgIDPs_err, 2)
    chi2_Rg_MDPs = np.power((RgMDPs_exp - RgMDPs_cal) / RgMDPs_err, 2)

    chi2_Rg_IDPs_res = np.power((RgIDPs_exp - RgIDPs_bt.T) / RgIDPs_err, 2).mean(axis=1)
    chi2_Rg_MDPs_res = np.power((RgMDPs_exp - RgMDPs_bt.T) / RgMDPs_err, 2).mean(axis=1)

    corrcoef_RgIDPs = np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_cal.tolist())[0][1]  # r
    corrcoef_RgMDPs = np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_cal.tolist())[0][1]

    corrcoef_RgIDPs_res = []
    corrcoef_RgMDPs_res = []
    for time in range(RgIDPs_bt.shape[-1]):
        corrcoef_RgIDPs_res.append(np.corrcoef(RgIDPs_exp.tolist(), RgIDPs_bt.T[time].tolist())[0][1])
        corrcoef_RgMDPs_res.append(np.corrcoef(RgMDPs_exp.tolist(), RgMDPs_bt.T[time].tolist())[0][1])
    # "A Brief Note on the Standard Error of the Pearson Correlation"
    ax3.scatter(RgIDPs_exp, RgIDPs_cal, label=f"IDPs, r={np.round(corrcoef_RgIDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgIDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgIDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_IDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_IDPs_res.std(), 1)}", color=orange, s=s, marker="o")
    ax3.scatter(RgMDPs_exp, RgMDPs_cal, label=f"MDPs, r={np.round(corrcoef_RgMDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgMDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgMDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_MDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_MDPs_res.std(), 1)}", color=blue, s=s, marker="s")
    # coefficient = np.corrcoef(Rg_exp, Rg_cal)[0][1]

    # assert len(allproteins.index) == len(Rg_cal)
    ax3.set_title("CALVADOS 2+COM")
    ax3.plot([1, 7], [1, 7], color="black", linewidth=linewidth)
    ax3.legend(edgecolor="white", frameon=False)
    ax3.set_ylabel("$R_{g,sim}$ [nm]")
    ax3.set_xlabel("$R_{g,exp}$ [nm]")
    ax3.tick_params(axis='x')
    ax3.tick_params(axis='y')
    ax3.set_aspect('equal')

def fig4_2(ax4):
    bootstrapping = True
    cwd = "/home/fancao/CALVADOSCOM"
    datasets_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": 3,
    "CALVADOS2CA_2.0_0.05_1_validate": -1,
    "CALVADOS2COM_2.0_0.05_1_validate": -1}
    # color_dict = {"IDPs_MDPsCOM_0.05_6": "tab:blue", "CALVADOS2CA_0.05_1": "tab:red", "CALVADOS2COM_0.05_1": "tab:blue"}
    label_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": CALVADOS_COM, "CALVADOS2CA_2.0_0.05_1_validate": "CALVADOS 2+$\\rm{C_α}$", "CALVADOS2COM_2.0_0.05_1_validate": "CALVADOS 2+COM"}
    edgecolor_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": blue, "CALVADOS2CA_2.0_0.05_1_validate": orange, "CALVADOS2COM_2.0_0.05_1_validate": green}
    # {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    # hatch_dict = {"IDPs_MDPsCOM_0.05_6": '..', "CALVADOS2CA_0.05_1": "//", "CALVADOS2COM_0.05_1": "\\\\"}
    height_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": .6, "CALVADOS2CA_2.0_0.05_1_validate": .4, "CALVADOS2COM_2.0_0.05_1_validate": .2}
    # ax[0].set_title("λ among different datasets", fontsize=20)
    # use current lambda values to simulate next cycle
    # ax.set_xlabel("Rg_exp [nm]")  
    for dataset_idx, dataset in enumerate(list(datasets_dict.keys())):
        # validate = True  
        validate = True
        cycle = datasets_dict[dataset]
        PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
        predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)
        names = list(predictions.index)
        for name in np.setdiff1d(list(predictions.index), IDP_names+multidomain_names):
            names.remove(name)
        Rg_exp = np.array(predictions.loc[names]["expRg"])
        Rg_cal = np.array(predictions.loc[names]["cal"])
        Error = np.array(predictions.loc[names]["expRgErr"])
        chi2_Rg = np.power((Rg_exp - Rg_cal) / Error, 2)
        Rg_bt = np.array([predictions.loc[name, "bt"] for name in names])  # (n_seq, times)
        chi2_Rg_bt = np.power((Rg_exp - Rg_bt.T) / Error, 2).mean(axis=1)
        ax4.bar(names, (Rg_cal - Rg_exp) / Rg_exp * 100, width=height_dict[dataset], label=f"{label_dict[dataset]}, <{string_chi2_rg}>={np.round(chi2_Rg.mean(), 1)}\u00B1{np.round(np.std(chi2_Rg_bt), 1)}", color=edgecolor_dict[dataset])
        ax4.errorbar(names, [0] * len(names), yerr=Error / Rg_exp * 100, lw=0, ms=0, elinewidth=.5, capsize=2, capthick=.5, color='k')

        ax4.set_xticks(np.arange(len(names)))
        ax4.set_xticklabels(labels=names, rotation=90)
        ax4.set_ylim(-40, 15)
        ax4.set_xlim(-1, len(names))

    ax4.tick_params(axis='y')
    ax4.tick_params(axis='x')
    ax4.legend(edgecolor="white", frameon=False)
    ax4.set_ylabel('$\Delta R_g$  /  $R_{g,exp}$ %')
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')

# figsize is independent on fontsize;
# if fontsize is too big, it might cause non-cubic figures

fig = plt.figure(figsize=(8,7), dpi=600)
ax1 = plt.subplot2grid((2,3), (0,0), rowspan=1, colspan=1)
ax2 = plt.subplot2grid((2,3), (0,1), rowspan=1, colspan=1)
ax3 = plt.subplot2grid((2,3), (0,2), rowspan=1, colspan=1)
ax5 = plt.subplot2grid((2,3), (1,0), rowspan=1, colspan=3)

fig4_1(ax3, ax1, ax2)
fig4_2(ax5)

fig.text(0.04, 0.96, 'A', fontsize=label_size)
fig.text(0.35, 0.96, 'B', fontsize=label_size)
fig.text(0.68, 0.96, 'C', fontsize=label_size)
fig.text(0.04, 0.55, 'D', fontsize=label_size)

plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/fig4.pdf', bbox_inches='tight')