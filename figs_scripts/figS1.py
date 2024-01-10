from color_setting import *
import matplotlib.colors as mcolors
from collections import OrderedDict
import mdtraj as md
import os
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
s = 5
linewidth = 1
times = 10
string_chi2_rg = "$χ^2_{R_g}$"
mpl.rc('font', size=text_size)

def figS1_1(ax1, dataset, total_cycles):
    eta = .1  # weight for pre
    datasets_dict = {dataset: total_cycles}
    loss_types = ['cost', 'chi2_rg', 'chi2_pre', 'theta_prior']  # theta_prior, chi2_rg, chi2_pre, cost
    cwd = "/home/fancao/CALVADOSCOM"
    labels_dict = {'cost': "total cost", "chi2_rg": "$<\chi^2_{R_g}$>", "chi2_pre": "η*<$\chi^2_{PRE}$>", "theta_prior": "θ*prior_loss"}
    # ax1.set_title("IDPs_MDPsCOM_0.05_6", fontsize=title_fontsize)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Loss")
    dataframe = pd.DataFrame(columns=['cost', 'Rgloss_IDP', 'Rgloss_multidomain'])
    color_loss = {'cost':blue, 'chi2_rg':orange, 'chi2_pre':yellow, 'theta_prior':red}
    for loss_type in loss_types:
        for dataset in datasets_dict.keys():
            print(dataset)
            total_cycles = datasets_dict[dataset]
            start_x = 0
            points_x = []
            loss = []
            cur_minimum = 0
            for cycle in range(total_cycles + 1):
                # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
                res = pd.read_pickle(f"{cwd}/{dataset}/{cycle}_chi2.pkl")
                points_x += (np.array(res.index)[1:] + start_x).tolist()
                loss_tmp = np.array(res[loss_type])[1:]
                if loss_type == 'theta_prior':
                    loss_tmp = list(-loss_tmp)
                if loss_type == "chi2_pre":
                    loss_tmp = list(loss_tmp * eta)
                if cycle != total_cycles:
                    iterations_min = res.index[np.argmin(res["cost"].to_numpy())]
                    cur_minimum = res.loc[iterations_min][loss_type]
                    if loss_type == 'theta_prior':
                        cur_minimum = -cur_minimum
                    if loss_type == "chi2_pre":
                        cur_minimum = eta * cur_minimum
                    if loss_type == "cost":
                        Rgloss_IDP_minimum = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl").loc[
                            iterations_min].to_numpy().mean()
                        Rgloss_multidomain_minimum = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl").loc[
                            iterations_min].to_numpy().mean()
                        print(loss_type, cycle, cur_minimum, Rgloss_IDP_minimum, Rgloss_multidomain_minimum)
                        dataframe.loc[dataset] = dict(cost=np.round(cur_minimum, 2), Rgloss_IDP=np.round(Rgloss_IDP_minimum, 2), Rgloss_multidomain=np.round(Rgloss_multidomain_minimum, 2))
                ax1.vlines(res.index[0]+start_x, -1, np.max(loss_tmp)*1.4, linestyle=':', color="black")
                loss += list(loss_tmp)
                start_x = points_x[-1]
            ax1.scatter(points_x, loss, s=s, label=f"{labels_dict[loss_type]}, {'min' if loss_type == 'cost' else 'picked'} value: {np.round(cur_minimum, 2)}", marker="o", color=color_loss[loss_type])
        if loss_type == "cost":
            dataframe.to_csv(f"{cwd}/dataframe.csv")
    ax1.legend(edgecolor="white", frameon=False)
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x')
    ax1.set_ylim(-10,500)

def figS1_2(ax2, dataset, total_cycles):
    eta = .1  # weight for pre
    datasets_dict = {dataset: total_cycles}
    loss_types = ['theta_prior']  # theta_prior, chi2_rg, chi2_pre, cost
    cwd = "/home/fancao/CALVADOSCOM"
    labels_dict = {'cost': "total cost", "chi2_rg": "$<\chi^2_{R_g}$>", "chi2_pre": "η*<$\chi^2_{PRE}$>", "theta_prior": "θ*prior_loss"}
    # ax2.set_title("IDPs_MDPsCOM_0.05_6", fontsize=title_fontsize)
    ax2.set_xlabel("Iterations")
    dataframe = pd.DataFrame(columns=['cost', 'Rgloss_IDP', 'Rgloss_multidomain'])
    for loss_type in loss_types:
        for dataset in datasets_dict.keys():
            print(dataset)
            total_cycles = datasets_dict[dataset]
            start_x = 0
            points_x = []
            loss = []
            cur_minimum = 0
            for cycle in range(total_cycles + 1):
                # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
                res = pd.read_pickle(f"{cwd}/{dataset}/{cycle}_chi2.pkl")
                points_x += (np.array(res.index)[1:] + start_x).tolist()
                loss_tmp = np.array(res[loss_type])[1:]
                if loss_type == 'theta_prior':
                    loss_tmp = list(-loss_tmp)
                if loss_type == "chi2_pre":
                    loss_tmp = list(loss_tmp * eta)
                if cycle != total_cycles:
                    iterations_min = res.index[np.argmin(res["cost"].to_numpy())]
                    cur_minimum = res.loc[iterations_min][loss_type]
                    if loss_type == 'theta_prior':
                        cur_minimum = -cur_minimum
                    if loss_type == "chi2_pre":
                        cur_minimum = eta * cur_minimum
                    if loss_type == "cost":
                        Rgloss_IDP_minimum = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl").loc[
                            iterations_min].to_numpy().mean()
                        Rgloss_multidomain_minimum = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl").loc[
                            iterations_min].to_numpy().mean()
                        print(loss_type, cycle, cur_minimum, Rgloss_IDP_minimum, Rgloss_multidomain_minimum)
                        dataframe.loc[dataset] = dict(cost=np.round(cur_minimum, 2), Rgloss_IDP=np.round(Rgloss_IDP_minimum, 2), Rgloss_multidomain=np.round(Rgloss_multidomain_minimum, 2))

                ax2.vlines(res.index[0] + start_x, -0.25, np.max(loss_tmp) * 1.4, linestyle=':', color="black")
                loss += list(loss_tmp)
                start_x = points_x[-1]
            ax2.scatter(points_x, loss, s=s, label=f"{labels_dict[loss_type]}, {'min' if loss_type == 'cost' else 'picked'} value: {np.round(cur_minimum, 2)}", marker="o", color=blue)
        if loss_type == "cost":
            dataframe.to_csv(f"{cwd}/dataframe.csv")
    ax2.legend(edgecolor="white", frameon=False)
    ax2.tick_params(axis='y')
    ax2.tick_params(axis='x')
    ax2.set_ylabel("Loss")
    ax2.set_ylim(-0.3, 12.5)

def figS1_3(ax3, dataset, total_cycles):
    cwd = "/home/fancao/CALVADOSCOM"
    datasets = [dataset]  # IDPs, IDPs_allmultidomain, IDPs_multidomainExcludeGS
    ax3.set_xlabel("Iterations")
    for dataset in datasets:
        print("dataset:", dataset)
        start_x = 0
        points_x = []
        chi2_IDPs_rg = []
        chi2_multi_rg = []
        chi2_IDPs_rg_min = 0
        chi2_multi_rg_min = 0
        for cycle in range(total_cycles + 1):
            res = pd.read_pickle(f"{cwd}/{dataset}/{cycle}_chi2.pkl")
            iterations_min = res.index[np.argmin(res["cost"].to_numpy())]
            res_IDP = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl")
            res_multi = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl")
            # columns: Index(['chi2_pre', 'chi2_rg', 'theta_prior', 'lambdas', 'xi', 'cost'], dtype='object')
            points_x += (np.array(res_IDP.index) + start_x).tolist()

            chi2_IDPs_rg += list(np.mean(res_IDP.to_numpy(), axis=1))
            chi2_multi_rg += list(np.mean(res_multi.to_numpy(), axis=1))
            if cycle != total_cycles:
                chi2_IDPs_rg_min = res_IDP.loc[iterations_min].mean()
                chi2_multi_rg_min = res_multi.loc[iterations_min].mean()
            ax3.vlines(start_x, -1, np.max(list(np.mean(res_IDP.to_numpy(), axis=1))+list(np.mean(res_multi.to_numpy(), axis=1)))*1.4, linestyle=':', color="black")
            start_x = points_x[-1]
        ax3.scatter(points_x, chi2_IDPs_rg, s=s, label="$<\chi^2_{R_g}$>-IDPs," + f" picked value: {np.round(chi2_IDPs_rg_min, 3)}", marker="o", color=blue)
        ax3.scatter(points_x, chi2_multi_rg, s=s, label="$<\chi^2_{R_g}$>-MDPs," + f" picked value: {np.round(chi2_multi_rg_min, 3)}", marker="o", color=orange)
    ax3.legend(edgecolor="white", frameon=False)
    ax3.tick_params(axis='y')
    ax3.tick_params(axis='x')
    ax3.set_ylabel("Loss")
    ax3.set_ylim(-10, 550)

def figS1_4(ax4, dataset, total_cycles):
    cwd = "/home/fancao/CALVADOSCOM"
    colors = ["#5566AA","#117733","#44AA66","#55AA22","#668822","#99BB55","#558877","#88BBAA","#AADDCC","#44AA88",
        "#DDCC66","#FFDD44","#FFEE88","#BB0011"]
    print(len(colors))
    datasets_dict = {dataset: total_cycles}
    multidomain_names = list(pd.read_pickle(f"{cwd}/{dataset}/MultiDomainsRgs.pkl").index)
    # ax4.set_title("$<\chi^2_{R_g}$>-MDPs", fontsize=title_fontsize)
    # ax1.set_ylabel("chi2_multi_rg", fontsize=20)
    ax4.set_xlabel("Iterations")
    print(multidomain_names)
    for i, name in enumerate(multidomain_names):
        print(name)
        start_x = 0
        points_x = []
        tmp = []
        ylim = [550, 400, 150, 120, 100]
        for cycle in range(total_cycles + 1):
            ax4.vlines(start_x, -1, ylim[cycle], linestyle=':', color="black")
            res = pd.read_pickle(f"{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl")
            tmp += np.squeeze(res[name].to_numpy()).tolist()
            points_x += (np.array(res.index) + start_x).tolist()
            start_x = points_x[-1]
        ax4.scatter(points_x, np.array(tmp), s=s, label=name, marker="o", color=colors[i])

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys(), edgecolor="white", frameon=False)
    ax4.tick_params(axis='y')
    ax4.tick_params(axis='x')
    ax4.set_ylabel("Loss")

fig = plt.figure(figsize=(5,6.5), dpi=600)
ax1 = plt.subplot2grid((2,4), (0,0), rowspan=1, colspan=2)
ax2 = plt.subplot2grid((2,4), (1,0), rowspan=1, colspan=2)
ax3 = plt.subplot2grid((2,4), (0,2), rowspan=1, colspan=2)
ax4 = plt.subplot2grid((2,4), (1,2), rowspan=1, colspan=2)


dataset = "IDPs_MDPsCOM_2.2_0.08_2"
total_cycles = 4  # cycle3 result was taken
figS1_1(ax1, dataset, total_cycles)
figS1_2(ax2, dataset, total_cycles)
figS1_3(ax3, dataset, total_cycles)
figS1_4(ax4, dataset, total_cycles)
plt.tight_layout()

fig.text(0.005, 0.97, 'A', fontsize=label_size)
fig.text(0.005, 0.47, 'B', fontsize=label_size)
fig.text(0.51, 0.97, 'C', fontsize=label_size)
fig.text(0.51, 0.47, 'D', fontsize=label_size)

plt.show()
fig.savefig(f'{cwd}/paper_multidomainCALVADOS/figS1.pdf', bbox_inches='tight')
