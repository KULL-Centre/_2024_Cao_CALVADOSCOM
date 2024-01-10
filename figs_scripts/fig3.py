from fig1 import get_predictions
import numpy as np
from color_setting import *
import statsmodels
import scipy
import mdtraj as md
import matplotlib.pyplot as plt
from protein_repo import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
cwd = "/home/fancao/CALVADOSCOM"
string_chi2_rg = "$χ^2_{R_g}$"
  
import matplotlib as mpl
text_size = 6
label_size = text_size*1.8
s = 20
linewidth = 1
times = 100
mpl.rc('font', size=text_size)
CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"
def fig3_1(ax1):
    # fig2_1
    cwd = "/home/fancao/CALVADOSCOM"
    datasets_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": [3]}
    marker_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": "s"}
    # ax[0].set_title("λ among different datasets", fontsize=20)
    residues_publication = pd.read_csv(f"{cwd}/residues_pub.csv")
    # ax1.bar(list(residues_publication.one), list(residues_publication.lambdas), color=orange, label="CALVADOS2", width=0.8)
    ax1.scatter(list(residues_publication.one), list(residues_publication.lambdas), color=orange, label="CALVADOS 2", marker="*", s=s, zorder=99)
    for dataset in datasets_dict.keys():
        for cycle in datasets_dict[dataset]:
            lambda_my = pd.read_csv(f"{cwd}/{dataset}/residues_{cycle}.csv")
            print(list(residues_publication.one))
            # it is for different datasets
            ax1.bar(list(lambda_my.one), list(lambda_my["lambdas"]), label=CALVADOS_COM, zorder=98)

    ax1.set_ylabel("λ values")
    ax1.set_xlabel("Residues")
    ax1.legend(edgecolor="white")
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')

def fig3_2(ax2):
    cwd = "/home/fancao/CALVADOSCOM"
    # fig2_2
    # use current lambda values to simulate next cycle
    dataset = "IDPs_MDPsCOM_2.2_0.08_2_validate"
    bootstrapping = True
    cycle = 3
    validate = False
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

    corrcoef_RgIDPs = np.corrcoef(RgIDPs_exp, RgIDPs_cal)[0][1]  # r
    corrcoef_RgMDPs = np.corrcoef(RgMDPs_exp, RgMDPs_cal)[0][1]

    corrcoef_RgIDPs_res = []
    corrcoef_RgMDPs_res = []
    for time in range(RgIDPs_bt.shape[-1]):
        corrcoef_RgIDPs_res.append(np.corrcoef(RgIDPs_exp, RgIDPs_bt.T[time])[0][1])
        corrcoef_RgMDPs_res.append(np.corrcoef(RgMDPs_exp, RgMDPs_bt.T[time])[0][1])
    string_chi2_rg = "$χ^2_{R_g}$"

    ax2.scatter(RgIDPs_exp, RgIDPs_cal, label=f"IDPs, r={np.round(corrcoef_RgIDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgIDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgIDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_IDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_IDPs_res.std(), 1)}", color=orange, s=s, marker="o")
    ax2.scatter(RgMDPs_exp, RgMDPs_cal, label=f"MDPs, r={np.round(corrcoef_RgMDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgMDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgMDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_MDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_MDPs_res.std(), 1)}", color=blue, s=s, marker="s")
    coefficient = np.corrcoef(Rg_exp, Rg_cal)[0][1]
    print(coefficient)

    # assert len(allproteins.index) == len(Rg_cal)
    ax2.set_title(CALVADOS_COM)
    ax2.plot([1, 7], [1, 7], color="black")
    ax2.legend()
    ax2.legend(edgecolor="white", frameon=False)
    ax2.set_ylabel("$R_{g,sim}$ [nm]")
    ax2.set_xlabel("$R_{g,exp}$ [nm]")
    ax2.tick_params(axis='x')
    ax2.tick_params(axis='y')
    ax2.set_aspect('equal')

def fig3_3(ax3):
    cwd = "/home/fancao/CALVADOSCOM"
    bootstrapping = True
    datasets_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": 3,
    "CALVADOS2CA_2.0_0.05_1_validate": -1,
    "CALVADOS2COM_2.0_0.05_1_validate": -1
    }
    label_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": CALVADOS_COM, "CALVADOS2CA_2.0_0.05_1_validate": "CALVADOS 2+$\\rm{C_α}$", "CALVADOS2COM_2.0_0.05_1_validate": "CALVADOS 2+COM"}
    edgecolor_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": blue, "CALVADOS2CA_2.0_0.05_1_validate": orange, "CALVADOS2COM_2.0_0.05_1_validate": green}
    height_dict = {"IDPs_MDPsCOM_2.2_0.08_2_validate": .8, "CALVADOS2CA_2.0_0.05_1_validate": .5, "CALVADOS2COM_2.0_0.05_1_validate": .3}
    # use current lambda values to simulate next cycle
    for dataset_idx, dataset in enumerate(list(datasets_dict.keys())):
        # validate = False
        validate = False
        cycle = datasets_dict[dataset]
        PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
        predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)
        names = list(predictions.index) 

        Rg_exp = np.array(predictions.loc[names]["expRg"])
        Rg_cal = np.array(predictions.loc[names]["cal"])
        Error = np.array(predictions.loc[names]["expRgErr"])
        chi2_Rg = np.power((Rg_exp - Rg_cal) / Error, 2)
        Rg_bt = np.array([predictions.loc[name, "bt"] for name in names])  # (n_seq, times)
        chi2_Rg_bt = np.power((Rg_exp - Rg_bt.T) / Error, 2).mean(axis=1)
        ax3.barh(names, (Rg_cal - Rg_exp) / Rg_exp * 100, height=height_dict[dataset], label=f"{label_dict[dataset]}, <{string_chi2_rg}>={np.round(chi2_Rg.mean(), 1)}\u00B1{np.round(np.std(chi2_Rg_bt), 1)}", color=edgecolor_dict[dataset])
        ax3.errorbar([0] * len(names), names, xerr=Error / Rg_exp * 100, lw=0, ms=0, elinewidth=.5, capsize=2, capthick=.5, color='k')
        ax3.set_yticks(np.arange(len(names)))
        # change hnRNPA1S and hSUMO_hRNPA1S:
        names_changed = []
        for name in names:
            if name in ["hnRNPA1S", "hSUMO_hnRNPA1S"]:
                name = name[:-1] + '*'
            names_changed.append(name)
        ax3.set_yticklabels(labels=names_changed)
        ax3.set_xlim(-40, 15)
        ax3.set_ylim(-1, len(names)+6)
    ax3.tick_params(axis='y')
    ax3.tick_params(axis='x')
    ax3.set_xlabel('$\Delta R_g$  /  $R_{g,exp}$ %')
    ax3.legend(edgecolor="white", frameon=False)


fig = plt.figure(figsize=(5.5,6.5), dpi=600)
ax1 = plt.subplot2grid((4,4), (0,0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((4,4), (2,0), rowspan=2, colspan=2)
ax3 = plt.subplot2grid((4,4), (0,2), rowspan=4, colspan=2)


fig3_1(ax1)
fig3_2(ax2)
fig3_3(ax3)
plt.tight_layout()

fig.text(0.005, 0.97, 'A', fontsize=label_size)
fig.text(0.005, 0.42, 'B', fontsize=label_size)
fig.text(0.51, 0.97, 'C', fontsize=label_size)

plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/fig3.pdf', bbox_inches='tight')
