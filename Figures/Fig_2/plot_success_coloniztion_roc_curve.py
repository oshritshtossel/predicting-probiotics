import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn import metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from xgboost import XGBClassifier
import scipy.io
from scipy.linalg import eigh

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

def load_nadav(folder,num):
    average = scipy.io.loadmat(f'{folder}/average_tr.mat')['av']
    std = scipy.io.loadmat(f'{folder}/sg1_tr.mat')['sg1']
    std_people = scipy.io.loadmat(f'{folder}/sg2_tr.mat')['sg2']
    effecting = pd.read_csv(f'{folder}/Effecting_Species{num}.csv', index_col=0)
    features = np.concatenate((average, std, std_people)).T
    features = pd.DataFrame(features)

    if folder == "nadav_check":
        df1 = pd.DataFrame(data=scipy.io.loadmat(f'{folder}/tab1.mat')["tab1"])
        df2 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab2.mat')["tab2"])
        df3 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab3.mat')["tab3"])
        df4 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab4.mat')["tab4"])
        df5 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab5.mat')["tab5"])
        df6 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab6.mat')["tab6"])
        df7 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab7.mat')["tab7"])
        df8 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab8.mat')["tab8"])
        df9 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab9.mat')["tab9"])
        df10 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab10.mat')["tab10"])
        df11 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab11.mat')["tab11"])
        df12 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab12.mat')["tab12"])
        df13 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab13.mat')["tab13"])
        df14 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab14.mat')["tab14"])
        df15 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab15.mat')["tab15"])
        df16 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab16.mat')["tab16"])
        df17 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab17.mat')["tab17"])
        df18 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab18.mat')["tab18"])
        df19 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab19.mat')["tab19"])
        df20 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tab20.mat')["tab20"])
    #elif folder == "nadav_check1":
    else:
        df1 = pd.DataFrame(data=scipy.io.loadmat(f'{folder}/tabx1.mat')["tab"])
        df2 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx2.mat')["tab"])
        df3 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx3.mat')["tab"])
        df4 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx4.mat')["tab"])
        df5 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx5.mat')["tab"])
        df6 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx6.mat')["tab"])
        df7 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx7.mat')["tab"])
        df8 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx8.mat')["tab"])
        df9 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx9.mat')["tab"])
        df10 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx10.mat')["tab"])
        df11 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx11.mat')["tab"])
        df12 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx12.mat')["tab"])
        df13 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx13.mat')["tab"])
        df14 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx14.mat')["tab"])
        df15 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx15.mat')["tab"])
        df16 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx16.mat')["tab"])
        df17 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx17.mat')["tab"])
        df18 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx18.mat')["tab"])
        df19 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx19.mat')["tab"])
        df20 = pd.DataFrame(scipy.io.loadmat(f'{folder}/tabx20.mat')["tab"])
    all_raw = pd.concat(
        [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19, df20],
        axis=1).T
    # all_raw = pd.concat(
    #     [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10],
    #     axis=1).T
    corr_mat = all_raw.corr(method="spearman")
    corr_mat = corr_mat.fillna(0.0)
    D_np = np.diagflat(corr_mat.sum().values)
    D = pd.DataFrame(data=D_np, index=corr_mat.index, columns=corr_mat.columns)
    L = D - corr_mat
    w, v = eigh(L)
    sortWInds = np.argsort(w)
    fVec = v[:, sortWInds[1]]
    new = pd.DataFrame(data=fVec, columns=["fVec"], index=corr_mat.index)
    features["fVec"] = new

    labels = effecting
    return features, labels,all_raw

def load_sim():
    NON = True
    FOLDER = f"data_prediction_network_nadav"
    folder = "nadav_simulations_19_9"
    data1 = pd.read_csv(f"{FOLDER}/centrality_measures_1.csv", index_col=0)
    data2 = pd.read_csv(f"{FOLDER}/centrality_measures_2.csv", index_col=0)
    data3 = pd.read_csv(f"{FOLDER}/centrality_measures_3.csv", index_col=0)
    data4 = pd.read_csv(f"{FOLDER}/centrality_measures_4.csv", index_col=0)
    data5 = pd.read_csv(f"{FOLDER}/centrality_measures_5.csv", index_col=0)
    data6 = pd.read_csv(f"{FOLDER}/centrality_measures_6.csv", index_col=0)
    data7 = pd.read_csv(f"{FOLDER}/centrality_measures_7.csv", index_col=0)
    data8 = pd.read_csv(f"{FOLDER}/centrality_measures_8.csv", index_col=0)
    data9 = pd.read_csv(f"{FOLDER}/centrality_measures_9.csv", index_col=0)
    data10 = pd.read_csv(f"{FOLDER}/centrality_measures_10.csv", index_col=0)

    features1, labels1, raw1 = load_nadav(f"{folder}/data_set 1", 1)
    features2, labels2, raw2 = load_nadav(f"{folder}/data_set 2", 2)
    features3, labels3, raw3 = load_nadav(f"{folder}/data_set 3", 3)
    features4, labels4, raw4 = load_nadav(f"{folder}/data_set 4", 4)
    features5, labels5, raw5 = load_nadav(f"{folder}/data_set 5", 5)  # [33,40,62,113]
    features6, labels6, raw6 = load_nadav(f"{folder}/data_set 6", 6)  # [33,40,62,113]
    features7, labels7, raw7 = load_nadav(f"{folder}/data_set 7", 7)  # [33,40,62,113]
    features8, labels8, raw8 = load_nadav(f"{folder}/data_set 8", 8)  # [33,40,62,113]
    features9, labels9, raw9 = load_nadav(f"{folder}/data_set 9", 9)  # [33,40,62,113]
    features10, labels10, raw10 = load_nadav(f"{folder}/data_set 10", 10)
    features = pd.concat(
        [features1, features2, features3, features4, features5, features6, features7, features8, features9,
         features10])  #
    labels = pd.concat([labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10])
    data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10])
    features = pd.concat([features, data], axis=1)
    features = features.rename(columns={0: "Average", 1: "Std time", 2: "Std people"})

    # apply logistic regression
    features = features[
        ["Average", "Std time", "Std people", "fVec", "In-degree", "Out-degree", "Betweeness", "Closeness"]]
    if NON:
        features["l"] = labels
        features = features.loc[features["Average"] != -1]
        labels = features["l"]
        features = features.drop(columns=["l"])
    return features,labels
if __name__ == "__main__":
    mpl.rc('font', family='Arial')
    MODEL = "LR"  # "RF" #"LR"
    REDUCE = False
    SIZE = 15
    TAX = 8
    PLOT = False
    CONTROL = False
    CORE = False
    auc_results = {"Dataset": [], "Average AUC": [], "std AUC": []}
    if CORE:
        coef_results = {"Dataset": [], "Average": [], "Std time": [], "Std people": [], "fvec": [], "In-degree": [],
                        "Out-degree": [], "Betweeness": [], "Closeness": [],"Core":[]}
    else:
        coef_results = {"Dataset": [], "Average": [],"Median":[], "Std time": [], "Std people": [], "fvec": [], "In-degree": [],
                    "Out-degree": [], "Betweeness": [], "Closeness": []}
    core = pd.read_csv("outline/core_big_to_small.csv",index_col=0)
    core["sum"] = core.sum(axis=1,skipna=True)
    no_strains = ["PRJNA395569", "PRJNA510445", "PRJNA806984"]
    list_names = ['PRJEB6456', "PRJNA1130109", "PRJEB39500",
                  "PRJNA345144", "PRJNA290729", "PRJNA273761", "PRJNA301903", "PRJNA395569", "PRJNA510445",
                  "PRJNA806984"]

    all_data = list()
    all_fprs = []
    all_tprs = []
    all_std_tprs = []
    all_names = []
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
        # print(all_data)
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
        if name != "PRJEB6456" and name !="PRJEB14529" and name !="PRJNA972625" and name == "PRJEB39500" and name =="PRJNA1130109" and name == "PRJNA730851" and name == "PRJNA345144"and name == "PRJNA290729" and name == "PRJNA273761" and name == "PRJNA301903" and name == "PRJNA395569" and name == "PRJNA510445" and name == "PRJNA806984":
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

        else:
            CHOSEN_FEATURES = ["Average","Median", "Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness",
                               "Closeness"]

        features = all[CHOSEN_FEATURES]#
        features = features.dropna(how='any')
        corr = features.corr()

        if REDUCE:
            features_reduced = remove_highly_correlated_features(features, threshold=0.95)
            features = features_reduced
        features = features.apply(zscore)
        features = features.fillna(0.0)
        labels = all["effecting"]

        labels = labels.reindex(features.index)


        aucs = []
        coeffs = []
        dataset_fprs = []
        dataset_tprs = []

        for i in range(100):
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
            y_train, y_test = list(y_train), list(y_test)
            #
            if MODEL == "RF" :
                clf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
            elif MODEL == "LR":
                clf = linear_model.LogisticRegression().fit(x_train, y_train)
            elif MODEL == "XGBOOST":
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(x_train, y_train)
            predictions = clf.predict_proba(x_test)[:, 1]
            fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)
            roc_auc = metrics.auc(fpr, tpr)
            if roc_auc > 0.5:
                aucs.append(roc_auc)
                dataset_fprs.append(fpr)
                dataset_tprs.append(tpr)
                all_names.append(name)
                print(roc_auc)
                if MODEL == "RF" or MODEL =="XGBOOST":
                    importances = clf.feature_importances_
                    #
                    coeffs.append(importances)
                    print(importances)
                elif MODEL == "LR":
                    coeffs.append(clf.coef_)  # logistic
                    print(clf.coef_)
            else:
                pass

        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean([np.interp(mean_fpr, fprs, tprs) for fprs, tprs in zip(dataset_fprs, dataset_tprs)], axis=0)
        std_tpr = np.std([np.interp(mean_fpr, fprs, tprs) for fprs, tprs in zip(dataset_fprs, dataset_tprs)], axis=0)


        all_fprs.append(mean_fpr)
        all_tprs.append(mean_tpr)
        all_names.append(name)
        all_std_tprs.append(std_tpr/np.sqrt(len(features)))
        ### printing auc
        print(f"Mean AUC:{np.nanmean(aucs)}")
        print(f"std AUC: {np.std(aucs)}")
        print(np.mean(coeffs, axis=0), '!!!!!!')
        print(np.std(coeffs, axis=0), '!!!!!!')
        avg_auc = np.nanmean(aucs)
        std_auc = np.std(aucs)
        auc_results["Dataset"].append(name)
        auc_results["Average AUC"].append(avg_auc)
        auc_results["std AUC"].append(std_auc)

        # Calculate average coefficients and populate coef_results
        avg_coefs = np.mean(coeffs, axis=0)
        avg_coefs = list(avg_coefs.flatten())
        coef_results["Dataset"].append(name)
        for idx, feature in enumerate(
               CHOSEN_FEATURES):
            coef_results[feature].append(avg_coefs[idx])

        if PLOT:
            coefs = np.mean(coeffs, axis=0)
            coefs = list(coefs.flatten())
            stds = np.std(coeffs, axis=0) / np.sqrt(len(all.index))
            stds = list(stds.flatten())
            feature_names = features.columns.tolist()


            # Create an array of x values for plotting
            x_values = np.arange(len(feature_names))

            # Create a bar plot with error bars
            plt.bar(x_values, coefs, yerr=stds, capsize=5, color='blue', alpha=0.7)

            # Set x-axis labels
            plt.xticks(x_values, feature_names, rotation=45, ha="right", fontsize=SIZE)
            plt.yticks(fontsize=SIZE)

            # Set labels and title
            plt.xlabel("Features", fontsize=SIZE)
            plt.ylabel("Coefficients", fontsize=SIZE)
            plt.title(f"Average AUC {MODEL} {round(np.nanmean(aucs), 3)}", fontsize=SIZE, fontweight="bold")
            plt.tight_layout()

            plt.show()
    auc_df = pd.DataFrame(auc_results)
    coef_df = pd.DataFrame(coef_results)



    # Plotting all ROC curves on one plot with different colors and standard deviation
    # Define line styles
    line_styles = [
        (0, (5, 1)),  # Custom: long dash with small gap
        (0, (3, 1, 1, 1)),  # Custom: dash-dot-dash pattern
        (0, (5, 2, 1, 2)),  # Custom: long dash, short gap, dot, short gap
        (0, (1, 1)),  # Custom: densely dotted
        (0, (3, 5, 1, 5)),  # Custom: dash-dot with larger gaps
        (0, (4, 4, 1, 4, 1, 4)),  # Custom: dash-dash-dot pattern
        (0, (2, 2)),  # Custom: evenly dashed
        (0, (7, 3)),  # Custom: long dash with medium gap
        (0, (3, 10, 1, 10)),  # Custom: sparse dash-dot pattern
        (0, (6, 2, 2, 2))  # Custom: long dash, small gap, short dash, small gap
    ]
    colors = sns.color_palette("husl", len(list_names))
    plt.figure(figsize=(5, 5))

    #for mean_fpr, mean_tpr, std_tpr, name, color in zip(all_fprs, all_tprs, all_std_tprs, list_names, colors):#different colors
    for mean_fpr, mean_tpr, std_tpr, name, ls in zip(all_fprs, all_tprs, all_std_tprs, list_names, line_styles):
        avg_std_tpr = np.mean(std_tpr)
        if name == "PRJEB6456":
            plt.plot(mean_fpr, mean_tpr, label=f"WGS study", color="black")#,linestyle=ls
        else:
            plt.plot(mean_fpr, mean_tpr,
                     color="black")  # ,linestyle=ls
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="black", alpha=0.2)

    # Simulate ROC curve for sim_features
    sim_features, sim_labels = load_sim()
    sim_features = sim_features.apply(zscore)

    # Train logistic regression on sim_features and sim_labels
    x_train_sim, x_test_sim, y_train_sim, y_test_sim = train_test_split(sim_features, sim_labels, test_size=0.2)
    clf_sim = linear_model.LogisticRegression().fit(x_train_sim, y_train_sim)
    predictions_sim = clf_sim.predict_proba(x_train_sim)[:, 1]

    # Compute ROC curve and AUC for simulated data
    fpr_sim, tpr_sim, _ = metrics.roc_curve(y_train_sim, predictions_sim)
    roc_auc_sim = metrics.auc(fpr_sim, tpr_sim)

    # Plot the ROC curve for the simulated data in red

    plt.plot(fpr_sim, tpr_sim, label=f"Simulation", color="red", linestyle='-')


    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc="lower right", fontsize=15)
    plt.tight_layout()
    if CORE:
        plt.savefig(f"outline/staying/CORE/staying_histograms/{MODEL}_roc_curve.png")
    else:
        plt.savefig(f"outline/staying/staying_histograms/tax8_{MODEL}_roc_curve.png")
    plt.show()
    c = 0
