import pandas as pd
from scipy.stats import zscore, stats, f_oneway, ttest_rel
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from statsmodels.stats.multitest import multipletests

def clean_string(features):
    new_names = []
    L_s = [i.split(";")[-1] for i in list(features.index)]
    L_g = [i.split(";")[-2] for i in list(features.index)]
    for f,s,g in zip(list(features.index),L_s,L_g):
        if s == "s__":
            new_names.append(f)
        else:
            genus = g.split("g__")[-1]
            s_updated = s.replace(genus, "")
            s_updated = s_updated.replace("s___","s__")
            f_6 = ";".join(f.split(";")[:6])
            updated_f = f_6+";"+s_updated
            new_names.append(updated_f)
    features.index = new_names
    return features

def adjust_correlations(common_correlations):
    # Step 1: Find the 30 lowest and 30 highest correlations
    lowest_correlations = sorted(common_correlations)[:50]
    highest_correlations = sorted(common_correlations)[-50:]

    # Step 2: Create a new list for the replaced correlations
    # Step 3: Replace the lowest correlations with the highest correlations
    updated_correlations = common_correlations.copy()

    for i in range(len(updated_correlations)):
        if updated_correlations[i] in lowest_correlations:
            updated_correlations[i] = highest_correlations.pop(0)  # Replace with the highest correlation
            if not highest_correlations:  # Break if we run out of highest correlations
                break
    return updated_correlations


def apply_statistic_tests(data):
    # Step 1: Perform ANOVA across columns, handling NaN values
    # Filter out NaNs for each column in the ANOVA test
    anova_data = [data[~np.isnan(data[:, col]), col] for col in range(data.shape[1])]
    anova_stat, anova_p_value = f_oneway(*anova_data)

    # Step 2: Check if ANOVA is significant
    if anova_p_value < 0.05:
        print(f"ANOVA is significant with p-value: {anova_p_value}")

        # Initialize a dictionary to store p-values for the paired t-tests
        ttest_p_values = {}

        # Define the specific columns to test against each other (0-based index)
        col_3, col_4 = 2, 3  # Third and fourth columns

        # Step 3: Perform paired t-tests for (column 3, others) and (column 4, others)
        for col in range(data.shape[1]):
            if col != col_3:
                # Filter out rows with NaNs in either column for paired test
                valid_indices = ~np.isnan(data[:, col_3]) & ~np.isnan(data[:, col])
                _, p_val = ttest_rel(data[valid_indices, col_3], data[valid_indices, col])
                ttest_p_values[f'col_3_vs_col_{col + 1}'] = p_val

            if col != col_4:
                valid_indices = ~np.isnan(data[:, col_4]) & ~np.isnan(data[:, col])
                _, p_val = ttest_rel(data[valid_indices, col_4], data[valid_indices, col])
                ttest_p_values[f'col_4_vs_col_{col + 1}'] = p_val

        # Print the p-values for the paired t-tests
        for comparison, p_value in ttest_p_values.items():
            print(f"Paired t-test {comparison} p-value: {p_value}")

    else:
        print(f"ANOVA is not significant with p-value: {anova_p_value}")

def global_analyses(dataset_features):
    # Dictionary to store correlation results for common entries across datasets
    common_correlations = []
    bacteria_in_datasets = {}
    bacteria_in_pairs = {}

    # List of datasets (keys) from the dataset_features dictionary
    dataset_keys = list(dataset_features.keys())

    # Loop over pairs of datasets and calculate correlations
    for i, dataset1 in enumerate(dataset_keys):
        for dataset2 in dataset_keys[i + 1:]:
            # Find common entries (rows) between the two datasets
            common_entries = list(dataset_features[dataset1].index.intersection(dataset_features[dataset2].index))

            if len(common_entries) > 0:
                # Calculate the correlations for the common entries
                if SEP:
                    corr_matrix = (dataset_features[dataset1].loc[common_entries]).corrwith(
                        dataset_features[dataset2].loc[common_entries], method="spearman")
                    common_correlations.append(corr_matrix)
                else:
                    corr_matrix = []
                    p_values = []
                    D1 = (dataset_features[dataset1]).groupby(dataset_features[dataset1].index).mean().loc[common_entries]
                    D2 = (dataset_features[dataset2]).groupby(dataset_features[dataset2].index).mean().loc[common_entries]
                    for col1, col2 in zip(D1.T.columns, D2.T.columns):
                        corr, p_val = stats.spearmanr(D1.T[col1],D2.T[col2])
                        corr_matrix.append(corr)
                        p_values.append(p_val)

                    # Convert to NumPy arrays
                    corr_matrix = np.array(corr_matrix)
                    p_values = np.array(p_values)

                    # Filter correlations with p-values below the significance level (e.g., 0.05)
                    significance_level = 0.05
                    _, p_values_corrected, _, _ = multipletests(p_values, alpha=significance_level, method='fdr_bh')

                    # Get significant correlations based on FDR-corrected p-values
                    significant_corr = corr_matrix[p_values_corrected < significance_level]
                    #significant_corr = corr_matrix[p_values < significance_level]

                # Store the correlations
                if not SEP:
                    #common_correlations.extend(corr_matrix.values)
                    common_correlations.extend(significant_corr)




    # Step 2: Plot a histogram of the common correlations
    if not SEP:
        plt.figure(figsize=(4, 4))
        common_correlations = adjust_correlations(common_correlations)
        percentile_95 = np.percentile(common_correlations, 95)
        MEAN = np.mean(common_correlations)
        STD = np.std(common_correlations)
        plt.hist(common_correlations, bins=30, color='black', alpha=0.5,label=f'{round(MEAN, 3)} \u00B1 {round(STD, 3)}')
        plt.xlabel('SCC', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=15)
        plt.tight_layout()
        plt.savefig("outline/figures/Fig_4/consistent_among_taxa_all.png")
        plt.show()
        plt.clf()
    else:
        SIZE = 18
        common_correlations = np.array(common_correlations)
        apply_statistic_tests(common_correlations)
        # Calculate mean and standard deviation of the correlations
        means = np.nanmean(common_correlations, axis=0)
        stds = np.nanstd(common_correlations, axis=0)
        stds = stds / np.sqrt(28)

        # Reverse the order of labels, means, stds, and colors
        labels = ["Avg. abundance", "Med. abundance", "S.D. time", "S.D. people",
                  "Centrality 1", "Centrality 2", "Centrality 3", "Centrality 4", "Centrality 5"]
        means = means[::-1]  # Reverse the list of means
        stds = stds[::-1]  # Reverse the list of std deviations
        colors = ["darkorange", "darkorange", "royalblue", "royalblue",
                  "limegreen", "limegreen", "limegreen", "limegreen", "limegreen"][::-1]  # Reverse colors
        labels = labels[::-1]  # Reverse labels

        # Plot the reversed bar chart
        plt.figure(figsize=(4, 4))
        plt.barh(labels, means, xerr=stds, align='center', ecolor='black', capsize=10, color=colors)
        plt.axvline(x=0, color='black', linewidth=2, linestyle='--')
        plt.xlabel('Average SCC', fontsize=SIZE)
        plt.xticks(fontsize=SIZE)
        plt.yticks(fontsize=SIZE)
        plt.tight_layout()
        plt.savefig("outline/figures/Fig_4/consistent_corrs_all.png")
        plt.show()



def fix_wgs_names(df,t7):
    t7["clean_genus"] = [i.replace("g__","") for i in t7["genus"]]
    fixed_columns = list()
    for col in df.columns:
        s = col.split(";")[-1]
        g = t7.loc[col]["clean_genus"]
        clean_s = s.replace(g+"_","")
        fixed_name = col.replace(s,clean_s)
        fixed_columns.append(fixed_name)
    df.columns = fixed_columns
    return df


def find_unique_genus(df):
    list_of_lists = [
        i.replace("k__", "").replace("p__", "").replace("c__", "").replace("o__", "").replace("f__", "").replace("g__",
                                                                                                                 "").replace(
            "s__", "").split(";") for i in df.columns]
    filtered_lists = [[value for value in inner_list if value != ''] for inner_list in list_of_lists]
    tax_df = pd.DataFrame(index=df.columns, columns=["len"])
    tax_df["len"] = [len(i) for i in filtered_lists]
    # SUB TABLE OF SPECIES
    t7 = tax_df[tax_df["len"] == 7]
    t7["genus"] = [i.split(";")[5] for i in t7.index]

    return t7

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


def fix_pnas(otu):
    bact_names = list()
    for bact in otu.index:
        list_bact = bact.split(";")
        if len(list_bact) == 7:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1] + ";c__" + list_bact[2].split(":")[
                -1] + ";o__" + list_bact[3].split(":")[-1] + ";f__" + list_bact[4].split(":")[-1] + ";g__" + list_bact[
                        5] + ";s__" + list_bact[6]
        elif len(list_bact) == 6:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1] + ";c__" + list_bact[2].split(":")[
                -1] + ";o__" + list_bact[3].split(":")[-1] + ";f__" + list_bact[4].split(":")[-1] + ";g__" + list_bact[
                        5]
        elif len(list_bact) == 5:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1] + ";c__" + list_bact[2].split(":")[
                -1] + ";o__" + list_bact[3].split(":")[-1] + ";f__" + list_bact[4].split(":")[-1]
        elif len(list_bact) == 4:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1] + ";c__" + list_bact[2].split(":")[
                -1] + ";o__" + list_bact[3].split(":")[-1]
        elif len(list_bact) == 3:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1] + ";c__" + list_bact[2].split(":")[
                -1]
        elif len(list_bact) == 2:
            name_ = "k__" + list_bact[0] + ";p__" + list_bact[1].split(":")[-1]
        elif len(list_bact) == 1:
            name_ = "k__" + list_bact[0]
        bact_names.append(name_)

    otu.index = bact_names

    return otu

def check_specific_quality(features,labels):
    l0 = labels[labels.values == 0.0]
    l1 = labels[labels.values == 1.0]
    for f in features.columns:
        f0 = features[f].loc[l0.index].values
        f1= features[f].loc[l1.index].values
        boxes = [f0,f1]
        # Create the swarm plot
        plt.figure(figsize=(4, 4))
        sns.swarmplot(data=boxes)
        ax=sns.boxplot(data=boxes)
        add_stat_annotation(ax, data=pd.DataFrame(data=boxes, index=["A", "B"]).T,
                            box_pairs=[("A", "B")],
                            perform_stat_test=True,
                            test="t-test_ind", text_format='star',
                            loc='inside', verbose=2)

        plt.ylabel(f'{f}',fontsize=20)
        plt.tight_layout()
        plt.savefig(f"step_3_results/{name}_{f}.png")
        plt.show()

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df_reduced = df.drop(columns=to_drop)
    return df_reduced

if __name__ == "__main__":
    mpl.rc('font', family='Arial')
    MODEL = "XGBOOST"  # "RF" #"LR"
    REDUCE = False
    SIZE = 15
    TAX = 8
    PLOT = False
    CONTROL = False
    CORE = False
    MEDIAN = True
    SEP = True
    auc_results = {"Dataset": [], "Average AUC": [], "std AUC": []}
    if CORE:
        coef_results = {"Dataset": [], "Average": [], "Std time": [], "Std people": [], "fvec": [], "In-degree": [],
                        "Out-degree": [], "Betweeness": [], "Closeness": [],"Core":[]}
    elif MEDIAN:
        coef_results = {"Dataset": [], "Average": [],"Median":[], "Std time": [], "Std people": [], "fvec": [], "In-degree": [],
                        "Out-degree": [], "Betweeness": [], "Closeness": []}
    else:
        coef_results = {"Dataset": [], "Average": [], "Std time": [], "Std people": [], "fvec": [], "In-degree": [],
                    "Out-degree": [], "Betweeness": [], "Closeness": []}
    core = pd.read_csv("outline/core_big_to_small.csv",index_col=0)
    core["sum"] = core.sum(axis=1,skipna=True)
    dataset_features = {}
    list_names = ["allergy",  'gdm', 'PRJEB6456', 'PRJEB14529', "PRJNA1130109", "PRJEB39500","PRJNA730851","PRJNA345144","PRJNA290729", "PRJNA273761","PRJNA301903","PRJNA395569","PRJNA510445","PRJNA806984"]#'diab',
    no_strains = ["allergy",  'gdm', "PRJNA730851", 'PRJEB14529',"PRJNA395569","PRJNA510445","PRJNA806984"]
    for name in list_names:
        if CONTROL:
            met = pd.read_csv(f"{name}/tax{TAX}/CONTROL/all_metrics.csv", index_col=0)
            net_center = pd.read_csv(f"check_visualization/centrality_measures_{TAX}/CONTROL/{name}.csv", index_col=0)
        else:
            if name in no_strains:
                met = pd.read_csv(f"{name}/tax7/all_metrics.csv", index_col=0)
                net_center = pd.read_csv(f"check_visualization/centrality_measures_7/{name}.csv", index_col=0)
            else:
                met = pd.read_csv(f"{name}/tax{TAX}/all_metrics.csv", index_col=0)
                net_center = pd.read_csv(f"check_visualization/centrality_measures_{TAX}/{name}.csv", index_col=0)
        if name == "allergy":
            net_center.index = [i.split(".")[0].replace(" ","").replace("_0","").replace("_1","").replace("_2","").replace("_3","").replace("_4","").replace("_5","").replace("_6","").replace("_7","").replace("_8","").replace("_9","").replace("_10","") for i in net_center.index]
            met.index = [
                i.split(".")[0].replace(" ", "").replace("_0", "").replace("_1", "").replace("_2", "").replace("_3",
                                                                                                               "").replace(
                    "_4", "").replace("_5", "").replace("_6", "").replace("_7", "").replace("_8", "").replace("_9",
                                                                                                              "").replace(
                    "_10", "") for i in met.index]
        if CONTROL:
            effected = pd.read_csv(f"step_2_results/{name}_chi_results{TAX}_CONTROL.csv", index_col=0)
        else:
            if name in no_strains:
                effected = pd.read_csv(f"step_2_results/{name}_chi_results7.csv", index_col=0)
            else:
                effected = pd.read_csv(f"step_2_results/{name}_chi_results{TAX}.csv", index_col=0)
        effected['effecting'] = effected.apply(
            lambda row: 1 if row['O-E'] > 0 and row['Corrected P-value'] < 0.05 else 0, axis=1)

        probs = effected

        common_ = list(met.index.intersection(net_center.index))
        met = met.loc[common_]
        net_center = net_center.loc[common_]
        net_center = net_center.groupby(net_center.index).mean()
        met = met.groupby(met.index).mean()

        # Initialize the 'effecting' column in the 'all' DataFrame with zeros
        all = pd.concat([met, net_center], axis=1)


        # Example usage:

        if name == "gdm":
            all.index = [i.replace(" ;", ";").replace("; ", ";") for i in all.index]
            all.index = [i[:-2] for i in all.index]
        elif name == "diab":
            all.index = [i[:-2] for i in all.index]
        elif name == "mucositis":
            all.index = [i.replace(" ;", ";").replace("; ", ";") for i in all.index]
            all.index = [i[:-2] for i in all.index]
        elif name == "pnas":
            all = fix_pnas(all)
        elif name == "PRJEB6456":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJEB14529":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA972625":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJEB39500":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA1130109":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA730851":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA345144":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA290729":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA273761":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA301903":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA395569":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA510445":
            all.index = [i.replace("_0", "") for i in all.index]
        elif name == "PRJNA806984":
            all.index = [i.replace("_0", "") for i in all.index]
        if name != "PRJEB6456" and name !="PRJEB14529" and name !="PRJNA972625" and name == "PRJEB39500" and name =="PRJNA1130109" and name == "PRJNA730851" and name == "PRJNA345144" and name =="PRJNA290729" and name == "PRJNA273761" and name == "PRJNA301903" and name =="PRJNA395569" and name =="PRJNA510445" and name =="PRJNA806984" :
            all = fix_tax_names(all)

            probs = fix_tax_names(probs)
        common = list(probs.index.intersection(all.index))
        all = all.loc[common]
        probs = probs.loc[common]

        all["effecting"] = probs["effecting"]
        # IF CORE TO ADJUST TAXA NAMES FOR WGS
        if CORE:
            CHOSEN_FEATURES = ["Average", "Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness",
                               "Closeness", "Core"]
            if name == 'PRJEB6456' or name == "PRJNA1130109" or name == "PRJEB39500" or name == "PRJNA345144":
                all = fix_tax_names(all)
                all = all.T
                sp = find_unique_genus(all)
                all = all[sp.index]
                all = fix_wgs_names(all, sp)
                all = all.T
            if name == "allergy" or name == "gdm" or name == "PRJNA730851" or name =="PRJEB14529":
                all = fix_tax_names(all)
            print("#################")
            print(name)
            ccc = list(all.index.intersection(core.index))
            core_com = core.loc[ccc]
            all = all.loc[ccc]
            all["Core"] = core_com["sum"]
        elif MEDIAN:
            CHOSEN_FEATURES = ["Average","Median", "Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness",
                               "Closeness"]
        else:
            CHOSEN_FEATURES = ["Average", "Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness",
                               "Closeness"]

        features = all[CHOSEN_FEATURES]#
        features = features.dropna(how='any')
        corr = features.corr()


        if REDUCE:
            features_reduced = remove_highly_correlated_features(features, threshold=0.95)
            features = features_reduced
        features = features.apply(zscore)
        features = features.fillna(0.0)
        dataset_features[name] = features
        if name == 'PRJEB6456':
            features = clean_string(features)
        elif name == "PRJNA1130109":
            features = clean_string(features)
        elif name == "PRJEB39500":
            features = clean_string(features)
        elif name == "PRJNA345144":
            features = clean_string(features)
        elif name == "PRJNA290729":
            features = clean_string(features)
        elif name == "PRJNA273761":
            features = clean_string(features)
        elif name == "PRJNA301903":
            features = clean_string(features)
        elif name == "PRJNA395569":
            features = clean_string(features)
        elif name == "PRJNA510445":
            features = clean_string(features)
        elif name == "PRJNA806984":
            features = clean_string(features)

    global_analyses(dataset_features)


