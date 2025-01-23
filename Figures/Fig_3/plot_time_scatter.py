import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

import numpy as np
def fix_tax_names(donors):
    bact_names = list()
    for bact in donors.index:
        list_bact = bact.split(";")
        if len(list_bact) == 7:
            if list_bact[6] == "s__":
                list_bact = list_bact[:6]
        if len(list_bact) == 6:
            if list_bact[5] == "g__":
                list_bact = list_bact[:5]
        if len(list_bact) == 5:
            if list_bact[4] == "f__":
                list_bact = list_bact[:4]
        if len(list_bact) == 4:
            if list_bact[3] == "o__":
                list_bact = list_bact[:3]
        if len(list_bact) == 3:
            if list_bact[2] == "c__":
                list_bact = list_bact[:2]
        if len(list_bact) == 2:
            if list_bact[1] == "p__":
                list_bact = list_bact[:1]

        bact_names.append(";".join(list_bact))
    donors.index = bact_names
    return donors

def check_specific_quality(features,labels,TOGETHER,LIST):
    l0 = labels[labels.values == 0.0]
    l1 = labels[labels.values == 1.0]
    for f in features.columns:
        f0 = features[f].loc[l0.index].values
        f1= features[f].loc[l1.index].values
        boxes = [f0,f1]
        # Create the swarm plot
        plt.figure(figsize=(4, 4))
        #ax = sns.violinplot(data=boxes, palette=["red", "blue"], inner=None)
        #ax = sns.swarmplot(data=boxes,palette=["red", "blue"])
        # sns.boxplot(data=boxes, ax=ax, showfliers=False,
        #             boxprops=dict(facecolor='white', edgecolor='black'))
        ax =sns.boxplot(data=boxes,

                    palette=["red", "blue"])
        add_stat_annotation(ax, data=pd.DataFrame(data=boxes, index=["A", "B"]).T,
                            box_pairs=[("A", "B")],
                            perform_stat_test=True,
                            test="t-test_ind", text_format='star',
                            loc='inside', verbose=2)
        if TOGETHER:
            if LIST == "imic":
                plt.title(f'Mixed-IMIC',fontweight="bold",fontsize=20)
            elif LIST == "lit":
                plt.title(f'Mixed-literature', fontweight="bold", fontsize=20)
        else:
            if LIST == "imic":
                plt.title(f'{name.upper()}-IMIC', fontweight="bold", fontsize=20)
            elif LIST == "lit":
                plt.title(f'{name.upper()}-literature', fontweight="bold", fontsize=20)
        plt.ylabel(f'{f}',fontsize=20)
        plt.tight_layout()
        if TOGETHER:
            if LIST == "imic":
                plt.savefig(f"outline/imic_all_{f}.png")
            elif LIST == "lit":
                plt.savefig(f"outline/literature/all_{f}.png")
        else:
            if LIST == "imic":
                plt.savefig(f"{name}/tax{TAX}/list/imic/{f}.png")
            elif LIST == "lit":
                plt.savefig(f"outline/literature/{name}_{f}.png")

        plt.show()
if __name__ == "__main__":
    mpl.rc('font', family='Arial')
    CONTROL = False
    TAX = 7
    TOGETHER = False
    LIST = "imic"# "imic","lit"
    TASK = "ttest"#"pred"#"ttest"
    MODEL = "XGBOOST"#"RF"
    PLOT =True
    SIZE = 20
    datasets = []
    labels_list = []
    list_names =["PRJNA1130109","PRJEB39500","PRJNA730851","PRJNA345144","PRJNA290729","PRJNA273761","PRJNA301903","PRJNA395569","PRJNA510445","PRJNA806984"]#["PRJEB39500","allergy",'PRJEB6456','PRJEB14529',"gdm","diab"]#['PRJEB6456','PRJEB14529',"gdm"]#[]
    for name in list_names:
        print("#############################")
        print(name)
        if CONTROL:
            met = pd.read_csv(f"{name}/tax{TAX}/CONTROL/all_metrics.csv", index_col=0)
            net_center = pd.read_csv(f"check_visualization/centrality_measures_{TAX}/CONTROL/{name}.csv", index_col=0)
        else:
            met = pd.read_csv(f"{name}/tax{TAX}/all_metrics.csv", index_col=0)
            net_center = pd.read_csv(f"check_visualization/centrality_measures_{TAX}/{name}.csv", index_col=0)
        if name == "allergy":
            net_center.index = [
                i.split(".")[0].replace(" ", "").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                               "").replace(
                    "_4", "").replace("_5", "").replace("_6", "").replace("_7", "").replace("_8", "").replace("_9",
                                                                                                              "").replace(
                    "_10", "") for i in net_center.index]
            met.index = [
                i.split(".")[0].replace(" ", "").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                               "").replace(
                    "_4", "").replace("_5", "").replace("_6", "").replace("_7", "").replace("_8", "").replace("_9",
                                                                                                              "").replace(
                    "_10", "") for i in met.index]


        probs = pd.read_csv(f"outline/time/{name}.csv",index_col=0)


        common_ = list(met.index.intersection(net_center.index))
        met = met.loc[common_]
        net_center = net_center.loc[common_]
        net_center = net_center.groupby(net_center.index).mean()
        met = met.groupby(met.index).mean()

        # Initialize the 'effecting' column in the 'all' DataFrame with zeros
        all = pd.concat([met, net_center], axis=1)

        if name == "gdm":
            all.index = [i.replace(" ;", ";").replace("; ", ";") for i in all.index]
            all.index = [i[:-2] for i in all.index]
        elif name == "diab":
            all.index = [i[:-2] for i in all.index]
        elif name == "mucositis":
            all.index = [i.replace(" ;", ";").replace("; ", ";") for i in all.index]
            all.index = [i[:-2] for i in all.index]
        elif name == "PRJEB6456":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJEB14529":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJEB39500":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA1130109":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA730851":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA345144":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA290729":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA273761":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA301903":
            all.index = [i.replace("_0", "") for i in all.index]
            all = fix_tax_names(all)
        elif name == "PRJNA395569":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA510445":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA806984":
            all.index = [i.replace("_0", "") for i in all.index]
        if name != "PRJEB6456" and name != "PRJEB14529" and name != "PRJEB39500" and name != "PRJNA1130109" and name != "PRJNA730851" and name != "PRJNA345144" and name != "PRJNA290729" and name != "PRJNA273761" and name != "PRJNA301903" and name != "PRJNA395569" and name != "PRJNA510445" and name != "PRJNA806984":
            all = fix_tax_names(all)
            # probs = pd.read_csv(f"{name}/tax7/probs.csv", index_col=0)
            probs = fix_tax_names(probs)
        common = list(probs.index.intersection(all.index))
        all = all.loc[common]
        probs = probs.loc[common]
        all["effecting"] = probs["time"]
        features = all[
            ["Average", "Median","Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness", "Closeness"]]  #
        features = features.dropna(how='any')
        labels = all["effecting"]
        datasets.append(features)
        labels_list.append(labels)

    # Concatenate all datasets and labels
    all_features = pd.concat(datasets)
    all_labels = pd.concat(labels_list)

    # Define the colors and markers
    subplot_colors = ['darkorange', 'darkorange','royalblue', 'royalblue', 'limegreen', 'limegreen', 'limegreen', 'limegreen', 'limegreen']
    markers = ['v', '<', '>', 'P', 'X', '*', 'h', 'H', 'd', '|']

    # Ensure the lengths of subplot_colors and markers match the required lengths
    feature_names = ["Avg. abundance","Med. abundance","S.D. time","S.D. people","Centrality 1","Centrality 2","Centrality 3","Centrality 4","Centrality 5"]
    all_features = all_features.rename(
        columns={"Average": "Avg. abundance", "Median": "Med. abundance", "Std time": "S.D. time",
                 "Std people": "S.D. people", "fvec": "Centrality 1", "In-degree": "Centrality 2",
                 "Out-degree": "Centrality 3", "Betweeness": "Centrality 4", "Closeness": "Centrality 5"})
    assert len(subplot_colors) >= len(feature_names), "Not enough colors defined for subplots"
    assert len(markers) >= len(list_names), "Not enough markers defined for datasets"

    # Create subplots
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))


    # Plot each feature in a different subplot
    for idx, f in enumerate(feature_names):
        ax = axs[idx // 3, idx % 3]
        all_x = []
        all_y = []

        for i, name in enumerate(list_names):
            start_idx = sum([len(labels_list[j]) for j in range(i)])
            end_idx = start_idx + len(labels_list[i])

            # Check if start_idx and end_idx are within bounds
            if start_idx >= len(all_labels) or end_idx > len(all_labels):
                continue

            x = all_labels[start_idx:end_idx]
            y = all_features[f].iloc[start_idx:end_idx]
            if f is "S.D. people":
                # ax.scatter(x, y, label=name_, color=subplot_colors[idx], marker=markers[i])

                valid_indices = y >= 10 ** -3
                x = x[valid_indices]
                y = y[valid_indices]
            elif f is "S.D. time":
                valid_indices = y >= 10 ** -6
                x = x[valid_indices]
                y = y[valid_indices]

            all_x.extend(x)
            all_y.extend(y)
            if name == "allergy":
               nn = "Allergy"
            elif name == "gdm":
                nn = "GDM"
            elif name == "PRJNA345144":
                nn = "Eczema"
            elif name == "PRJEB6456":
               nn = "Early dynamics 1"
            elif name == "PRJEB39500":
                nn = "T2D"
            elif name == "PRJNA1130109":
                nn = "Obesity"
            elif name == "PRJNA730851":
                nn = "Infant allergy"
            elif name == "PRJEB14529":
                nn = "Early dynamics 2"
            elif name == "PRJNA290729":
                nn = "Fatty liver"
            elif name == "PRJNA273761":
               nn = "NEC"
            elif name == "PRJNA301903":
                nn = "Preterm infants"
            elif name == "PRJNA395569":
                nn = "IBDMDB"
            elif name == "PRJNA510445":
               nn = "CF"
            elif name == "PRJNA806984":
               nn = "Healthy"

            if name == "allergy":
                ax.scatter(x, y, label="Allergy", color=subplot_colors[idx], marker=markers[i])
            elif name == "gdm":
                ax.scatter(x, y, label="GDM", color=subplot_colors[idx], marker=markers[i])
            elif name == "diab":
                ax.scatter(x, y, label="Diabimmune", color=subplot_colors[idx], marker=markers[i])
            else:
                ax.scatter(x, y, label=nn, color=subplot_colors[idx], marker=markers[i])

        sns.regplot(x=np.array(all_x), y=np.array(all_y), scatter=False, ax=ax, color=subplot_colors[idx],
                    line_kws={'linewidth': 1})

        ax.set_xlabel("Persistence time (days)", fontsize=17)
        ax.set_ylabel(f, fontsize=20)
        if f in [ "S.D. time", "S.D. people"]:
            ax.set_yscale('log')
            ax.set_xscale('log')
        # if idx == 0:
        #     ax.legend(ncol=2,fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=20)

    plt.tight_layout()
    plt.savefig("outline/time/scatter_strains.png")
    plt.show()
