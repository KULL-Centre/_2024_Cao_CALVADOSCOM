import matplotlib.patches as mpatches
from collections import OrderedDict
from color_setting import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from protein_repo import *
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)   
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
import matplotlib as mpl
text_size = 6
label_size = text_size*1.5
s = 20
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=text_size)

def figS6_1(cwd, models_dic, df_csat, axs):
    markers = {"hnRNPA1S":"o",
        "hSUMO_hnRNPA1S":"v",
        "FL_FUS":"^",
        "GFP_FUS":"s",
        "SNAP_FUS":"p",
        "SNAP_FUS_PLDY2F_RBDR2K":"*",
        "SNAP_FUS_PLDY2F":"X",
        "FUS_PLDY2F_RBDR2K":"D"}
    # markers = ["o","v","^","s","p","*","X","D"]
    colors = [orange, blue]
    for model_idx, model in enumerate(models_dic.keys()):
        print(model)
        dataset = models_dic[model][0]
        MDPs_names = models_dic[model][1]
        """real_MDPs_names = []
        Csat_exp = []
        Csat_cal = []
        for record in MDPs_names:
            cal_table = pd.read_pickle(f"{cwd}/{dataset}/{record}/value_{record}.pkl").loc[record]
            Csat_exp.append(df_csat.loc[record]["csat_exp"])  # μM
            Csat_cal.append(cal_table[f"{record}_dil"]*1000)  # μM
            print(record, cal_table[f"{record}_dil"]*1000)

        df_csat[f"{model}_Csat"] = Csat_cal
        assert len(Csat_exp)==len(Csat_cal)
        Csat_exp_tmp = []
        Csat_cal_tmp = []
        for i in range(len(MDPs_names)):
            if str(Csat_cal[i])!="nan":
                Csat_exp_tmp.append(Csat_exp[i])
                Csat_cal_tmp.append(Csat_cal[i])
                real_MDPs_names.append(MDPs_names[i])
        coefficient = np.corrcoef(Csat_exp_tmp, Csat_cal_tmp)[0][1]
        assert len(markers)==len(Csat_exp_tmp)
        print(Csat_cal_tmp)
        for idx, marker in enumerate(markers):
            axs.scatter(Csat_exp_tmp[idx], Csat_cal_tmp[idx], label=real_MDPs_names[idx], s=s,
                linewidths=linewidth, marker=markers[idx], color=colors[model_idx])"""
        for record in MDPs_names:
            if record in ["hSUMO_TIA1PrLD"]:
                continue
            cal_table = pd.read_pickle(f"{cwd}/{dataset}/{record}/value_{record}.pkl").loc[record]
            Csat_exp = df_csat.loc[record]["csat_exp"]  # μM
            Csat_cal = cal_table[f"{record}_dil"]*1000  # μM
            axs.scatter(Csat_exp, Csat_cal, s=s, linewidths=linewidth, marker=markers[record], color=colors[model_idx])

    axs.plot([0.0001, 800], [0.0001, 800], color="black")
    axs.set_yscale('log')
    axs.set_xscale('log')
    axs.set_aspect('equal')
    axs.set(xlabel='Experimental $\\rm{c_{sat}}$ [μM]', ylabel='Simulated $\\rm{c_{sat}}$ [μM]')
    CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"
    # axs.set_title(CALVADOS_COM)
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), edgecolor="white", frameon=False)

    blue_patch = mpatches.Patch(color=blue, label=CALVADOS_COM)
    orange_patch = mpatches.Patch(color=orange, label="CALVADOS 2+$\\rm{C_α}$")
    axs.legend(handles=[blue_patch,orange_patch], edgecolor="white", frameon=False)
    return df_csat

cwd = "/home/fancao/CALVADOSCOM"
IDPs = False
cycle = 0
fig = plt.figure(figsize=(3,3), dpi=600)
ax1 = plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
# ax2 = plt.subplot2grid((1,2), (0,1), rowspan=1, colspan=1)
# ax3 = plt.subplot2grid((2,2), (1,0), rowspan=1, colspan=2)
CALVADOS_COM = "$\\rm{CALVADOS_{COM}}$"

if IDPs:
    df_csat = pd.read_csv(f"{cwd}/csat_calvados2_test.csv").set_index("Unnamed: 0")
else:
    df_csat = pd.read_csv(f"{cwd}/csat_MDPs_test.csv").set_index("Unnamed: 0")
models_dic = {
        "CALVADOS 2+$\\rm{C_α}$": ("slabC2_CA_cutoff2.0_1", list(df_csat.index)),
        CALVADOS_COM: ("slabCALVADOSCOMcutoff2.2_0.08_2_COM_cutoff2.0_1", list(df_csat.index)),
    }
df_csat = figS6_1(cwd, models_dic, df_csat, ax1)
"""df_csat = df_csat[["csat_exp"]+[f"{model}_Csat" for model in list(models_dic.keys())]]
df_csat[f"{CALVADOS_COM}_diff"] = (df_csat[f"{CALVADOS_COM}_Csat"]-df_csat["csat_exp"])/df_csat["csat_exp"] *100
df_csat["CALVADOS 2+COM_diff"] = (df_csat["CALVADOS 2+COM_Csat"]-df_csat["csat_exp"])/df_csat["csat_exp"] *100
print(df_csat[list(df_csat.columns)[:3]])
ax3.set_title("$(C_{sat,cal}-C_{sat,exp})$ / $C_{sat,exp}$ [%]")
ax3.bar(np.arange(len(df_csat.index))+1-0.1, df_csat["opt2.2nm_0.08_diff"], width=0.2, label="opt2.2nm_0.08")
ax3.bar(np.arange(len(df_csat.index))+1+0.1, df_csat["CALVADOS 2+COM_diff"], width=0.2, label="CALVADOS 2+COM")
ax3.set_xticks(np.arange(len(df_csat.index))+1)
ax3.set_xticklabels(labels=list(df_csat.index), rotation=90)
ax3.set_xlim(0, len(df_csat.index)+1)"""

# plt.legend(edgecolor="white", frameon=False)
plt.tight_layout()
plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/fig6.pdf', bbox_inches='tight')

