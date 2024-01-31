from fig1 import get_predictions
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
mpl.rc('font', size=7)

def figS3_1(ax_list):
    cwd = "/home/fancao/CALVADOSCOM"
    bootstrapping = True
    validate = True
    label_dict = {"CALVADOS2SCCOM_2.0_0.05_1_validate": "CALVADOS 2+SCCOM", "IDPs_MDPsSCCOM_2.2_0.08_1_validate": "$\\rm{CALVADOS_{SCCOM}}$"}
    # use current lambda values to simulate next cycle
    dataset_tuples = {"CALVADOS2SCCOM_2.0_0.05_1_validate": -1, "IDPs_MDPsSCCOM_2.2_0.08_1_validate": 3, }
    for dataset_tuple_idx, dataset in enumerate(list(dataset_tuples.keys())):
        cycle = dataset_tuples[dataset]
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
        print(predictions.loc[IDP_names][["expRg", "cal", "expRgErr"]])
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

        ax_list[dataset_tuple_idx].scatter(RgIDPs_exp, RgIDPs_cal, label=f"IDPs, r={np.round(corrcoef_RgIDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgIDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgIDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_IDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_IDPs_res.std(), 1)}", color=orange, s=s, marker="o")
        ax_list[dataset_tuple_idx].scatter(RgMDPs_exp, RgMDPs_cal, label=f"MDPs, r={np.round(corrcoef_RgMDPs, 2)}\u00B1{0.01 if np.round(np.std(corrcoef_RgMDPs_res), 2) < 0.001 else np.round(np.std(corrcoef_RgMDPs_res), 2)},\n <{string_chi2_rg}>={np.round(chi2_Rg_MDPs.mean(), 1)}\u00B1{np.round(chi2_Rg_MDPs_res.std(), 1)}", color=blue, s=s, marker="s")
        coefficient = np.corrcoef(Rg_exp, Rg_cal)[0][1]
        # assert len(allproteins.index) == len(Rg_cal)
        ax_list[dataset_tuple_idx].plot([1, 7], [1, 7], color="black")
        ax_list[dataset_tuple_idx].set_ylabel("$R_{g,sim}$ [nm]")
        ax_list[dataset_tuple_idx].set_xlabel("$R_{g,exp}$ [nm]")
        ax_list[dataset_tuple_idx].set_aspect('equal')
        ax_list[dataset_tuple_idx].tick_params(axis='x')
        ax_list[dataset_tuple_idx].tick_params(axis='y')
        ax_list[dataset_tuple_idx].set_title(label_dict[dataset])
        ax_list[dataset_tuple_idx].legend(edgecolor="white", frameon=False)


def figS3_2(ax4):
    cwd = "/home/fancao/CALVADOSCOM"
    bootstrapping = True
    # use current lambda values to simulate next cycle
    datasets_dict = {"IDPs_MDPsSCCOM_2.2_0.08_1_validate": 3, "CALVADOS2SCCOM_2.0_0.05_1_validate": -1, }
    label_dict = {"IDPs_MDPsSCCOM_2.2_0.08_1_validate": "$\\rm{CALVADOS_{SCCOM}}$", "CALVADOS2SCCOM_2.0_0.05_1_validate": "CALVADOS 2+SCCOM", "CALVADOS2COM_0.05_1": "CALVADOS2+COM"}
    edgecolor_dict = {"IDPs_MDPsSCCOM_2.2_0.08_1_validate": blue, "CALVADOS2SCCOM_2.0_0.05_1_validate": orange, "CALVADOS2COM_0.05_1": "#3a52d9ff"}
    height_dict = {"IDPs_MDPsSCCOM_2.2_0.08_1_validate": .8, "CALVADOS2SCCOM_2.0_0.05_1_validate": .5, "CALVADOS2COM_0.05_1": .3}
    for dataset_idx, dataset in enumerate(list(datasets_dict.keys())):
        cycle = datasets_dict[dataset]
        # validate = True
        validate = True
        PRE_seq = list(pd.read_pickle(f"{cwd}/{dataset}/proteinsPRE.pkl").index)
        predictions, IDP_names, multidomain_names = get_predictions(cwd, dataset, cycle, validate, PRE_seq, bootstrapping=bootstrapping)
        names = list(predictions.index)
        for name in np.setdiff1d(list(predictions.index), IDP_names + multidomain_names):
            names.remove(name)
        Rg_exp = np.array(predictions.loc[names]["expRg"])
        Rg_cal = np.array(predictions.loc[names]["cal"])
        Error = np.array(predictions.loc[names]["expRgErr"])
        chi2_Rg = np.power((Rg_exp - Rg_cal) / Error, 2)
        Rg_bt = np.array([predictions.loc[name, "bt"] for name in names])  # (n_seq, times)
        chi2_Rg_bt = np.power((Rg_exp - Rg_bt.T) / Error, 2).mean(axis=1)
        ax4.barh(names, (Rg_cal - Rg_exp) / Rg_exp * 100, height=height_dict[
            dataset], label=f"{label_dict[dataset]}, <{string_chi2_rg}>={np.round(chi2_Rg.mean(), 1)}\u00B1{np.round(np.std(chi2_Rg_bt), 1)}", color=
        edgecolor_dict[dataset])
        ax4.errorbar([0] * len(names), names, xerr=Error / Rg_exp * 100, lw=0, ms=0, elinewidth=.5, capsize=2, capthick=.5, color='k')
        ax4.set_yticks(np.arange(len(names)))
        ax4.set_yticklabels(labels=names)
        ax4.set_xlim(-17.5, 17)
        ax4.set_ylim(-1, len(names)+3)

    ax4.tick_params(axis='y')
    ax4.tick_params(axis='x')
    ax4.legend(edgecolor="white", frameon=False)
    ax4.set_xlabel('$\Delta R_g$  /  $R_{g,exp}$ %')
    ax4.tick_params(axis='x')
    ax4.tick_params(axis='y')

fig = plt.figure(figsize=(6.5,6), dpi=600)
ax1 = plt.subplot2grid((4,4), (0,0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((4,4), (2,0), rowspan=2, colspan=2)
ax3 = plt.subplot2grid((4,4), (0,2), rowspan=4, colspan=2)


figS3_1([ax1, ax2])
figS3_2(ax3)

fig.text(0.005, 0.97, 'A', fontsize=label_size)
fig.text(0.005, 0.45, 'B', fontsize=label_size)
fig.text(0.545, 0.97, 'C', fontsize=label_size)

plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/figS3.pdf', bbox_inches='tight')