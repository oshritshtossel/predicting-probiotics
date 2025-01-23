import pandas as pd
import numpy as np
from scipy.linalg import eigh
import re
def calculate_sg2(df):
    """
    Calculate the average standard deviation over population for each taxon in the study
    :param df: Pandas dataframe of processed microbiome with additional of:'person' and 'time'
    such that the 'person' column indicates the host identity, and the 'time' column indicates the time step of the sampel
    :return: A Series with the average std over hosts of each taxon.
    """
    dict_vars = dict()
    if name == "allergy":
        for p,df_p in df.groupby("person"):
            df_p['time'] = df_p['time'].rank(method='dense').astype(int)
    for t, df_ in df.groupby("time"):
        if len(df_)>1:
            dict_vars[t] = df_.std()
    df_vars = pd.DataFrame(dict_vars)
    if name == "allergy":
        df_vars = df_vars.drop(["person", "time"])
    elif name == "PRJNA395569":
        df_vars = df_vars.drop(["time"])
    var_mean = df_vars.mean(axis=1, skipna=True)
    return var_mean

def calculate_sg1(df):
    """
    Calculate the standard deviation over time each taxon in the study
    :param df: pandas dataframe of processed microbiome with additional of:'person' and 'time'
    such that the 'person' column indicates the host identity, and the 'time' column indicates the time step of the sampel
    :return: a Series with the average std over time of each taxon.
    """
    bigger3 =0
    dict_vars = dict()
    for t, df_ in df.groupby("person"):
        if len(df_)>1:
            sss = df_.std()
            #sss =sss.fillna(0.0)
            dict_vars[t] = sss
        if len(df_)>=2:
            bigger3 = bigger3+1
    df_vars =  pd.DataFrame(dict_vars)
    if name == "allergy":
        df_vars = df_vars.drop(["person","time"])
    elif name == "PRJNA395569":
        df_vars = df_vars.drop(["time"])
    var_mean = df_vars.mean(axis=1, skipna=True)
    print(bigger3)

    return var_mean

def calculate_fvec(otu):
    """
    Calculate the Fiedler vector for each taxon
    :param otu: Pandas dataframe with processed microbiome data
    :return:  Series with the average Fiedler vector for each taxon
    """
    # Calculate the Spearman correlation matrix for
    corr_mat = otu.corr(method="spearman")
    corr_mat = corr_mat.fillna(0.0)
    D_np = np.diagflat(corr_mat.sum().values)
    D = pd.DataFrame(data=D_np, index=corr_mat.index, columns=corr_mat.columns)
    L = D - corr_mat
    w, v = eigh(L)
    sortWInds = np.argsort(w)
    fVec = v[:, sortWInds[1]]
    new = pd.DataFrame(data=fVec, columns=["fVec"], index=corr_mat.index)
    new = new.groupby(new.index).mean()
    return new

def separate_number_letter(index):
    """
    Fix format for a specific dataset
    :param index: indexes
    :return: number, letter
    """
    match = re.match(r'(\d+)([A-Za-z]?)', index)
    if match:
        number = match.group(1)
        letter = match.group(2)
        return number, letter
    return None, None

if __name__ == "__main__":
    # List with datasets names
    list_names =["PRJNA806984"]
    # Taxonomy level
    TAX = 7
    CONTROL = False
    for name in list_names:
        if TAX == 7 or TAX == 8:
            if TAX == 8:
                otu = pd.read_csv(f"{name}/tax8/tax8_log_sub PCA.csv",index_col=0)
            if name == "gdm" or name=="mucositis":
                otu = pd.read_csv(f"{name}/tax7/ready_otu.csv",index_col=0)
                if CONTROL:
                    meta = pd.read_csv(f"gdm/tax{TAX}/mapping.csv", index_col=0)
                    meta["name"] = meta["Group"] + "-STOOL-" + meta["trimester"]
                    meta.index = meta["name"]
                    meta0 = meta[meta["Tag"] == 0.0]
                    common = list(meta0.index.intersection(otu.index))
                    otu = otu.loc[common]
            elif name == "allergy":
                otu = pd.read_csv(f"{name}/tax7/tax7_log_subpca.csv",index_col=0)
                meta = pd.read_csv(f"{name}/tax7/metadata_merge_all_with_ok140.csv", index_col=0)
                meta = meta.loc[otu.index]
                otu["person"] = meta["ParticipentCode"]
                otu["time"] = meta["Date"]
                otu = otu.dropna()
                otu['time'] = pd.to_datetime(otu['time'])
            elif name == "diab":
                otu = pd.read_csv(f"{name}/tax7/ready_otu.csv", index_col=0)
                if CONTROL:
                    meta = pd.read_csv(f"diab/tax{TAX}/metadata_diabimmune.csv", index_col=0)
                    meta["fixed"] = meta["subjectID"] + "-" + [str(i) for i in meta["collection_month"]]
                    meta.index = meta["fixed"]
                    meta["total_allergy"] = meta[
                        ['allergy_milk', 'allergy_egg', 'allergy_peanut', 'allergy_dustmite', 'allergy_cat',
                         'allergy_dog',
                         'allergy_birch', 'allergy_timothy']].sum(axis=1)
                    meta0 = meta[meta["total_allergy"] == 0.0]
                    COMMON = meta0.index.intersection(otu.index)
                    otu = otu.loc[COMMON]
            elif name == "PRJEB6456":
                if TAX == 7:
                    otu = pd.read_csv(f"PRJEB6456/tax{TAX}/tax7_log_subpca.csv", index_col=0)
                else:
                    otu = pd.read_csv(f"PRJEB6456/tax{TAX}/tax8_log_sub PCA.csv",index_col=0)
                tag = pd.read_csv("PRJEB6456/tax7/tag.csv", index_col=0)
                otu["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
                otu["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
                # Create a dictionary to map the values
                mapping = {'B': 0, 'M': 1, '4M': 4, '12M': 12}

                # Replace the values in the "time" column
                otu['time'] = otu['time'].replace(mapping)
            elif name == "PRJEB14529":
                otu = pd.read_csv("PRJEB14529/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJEB14529/tax7/tag.csv", index_col=0)
                # Remove negative times
                tag = tag[tag["month_of_life"] >= 0.0]
                otu = otu.loc[tag.index]
                otu["person"] = tag["host_subject_id"]
                otu["time"] = tag["month_of_life"]
                tag["name"] = [str(otu["person"].loc[i]) + "-" + str(otu["time"].loc[i]) for i in otu.index]
                otu.index = tag["name"]
                otu = otu.groupby(otu.index).first()
            elif name == "PRJNA972625":
                otu = pd.read_csv("PRJNA972625/tax7/tax7_log_subpca.csv",index_col=0)
                tag = pd.read_csv("PRJNA972625/tax7/meta.csv",index_col=0)
                tag['Sample Name'] = [i.split("_")[-1] for i in tag['Sample Name']]
                tag[['person', 'time']] = tag['Sample Name'].apply(lambda x: pd.Series(separate_number_letter(x)))
                tag["time"] = tag['time'].apply(lambda x: 0.0 if x is '' else x)
                mapping = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0,'E':5.0,'F':6.0,'G':7.0}
                # Replace the values in the "time" column
                tag['time'] = tag['time'].replace(mapping)
                otu["person"] = tag["person"]
                otu["time"] = tag["time"]
            elif name == "PRJEB39500":
                if TAX == 7:
                    otu = pd.read_csv("PRJEB39500/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJEB39500/tax7/tag.csv", index_col=0)
                otu.index = [i.split(".")[0] for i in otu.index]
                cc = list(otu.index.intersection(tag.index))
                otu = otu.loc[cc]
                tag = tag.loc[cc]
                otu["person"] = [i.split("_")[0] for i in tag["Sample_Name"]]
                otu["time"] = [i.split("_")[-1].split("-")[0] for i in tag["Sample_Name"]]
                mapping = {'M0': 0, 'M7d': 2, 'M24h': 1}

                # Replace the values in the "time" column
                otu['time'] = otu['time'].replace(mapping)
            elif name == "PRJNA1130109":
                if TAX ==7:
                    otu = pd.read_csv("PRJNA1130109/tax7/tax7_log_subpca.csv",index_col=0)
                tag = pd.read_csv("PRJNA1130109/tax7/tag.csv",index_col=0)
                otu["person"] = [i.split("-")[0] for i in tag["Sample Name"]]
                otu["time"] = [i.split("tp")[-1] for i in tag["timepoint"]]
            elif name == "PRJNA730851":
                otu = pd.read_csv("PRJNA730851/tax7/tax7_log_subpca.csv", index_col=0)
                otu.index = [i.split("_")[0] for i in otu.index]
                otu = otu.groupby(otu.index).sum()
                tag = pd.read_csv("PRJNA730851/tax7/tag.csv", index_col=0)

                otu["person"] = [i.split(".")[1] for i in tag["Sample Name"]]
                otu["time"] = [int(i.split(".")[2]) for i in tag["Sample Name"]]
            elif name =="PRJNA345144":
                if TAX ==7:
                    otu = pd.read_csv("PRJNA345144/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJNA345144/tax7/tag.csv", index_col=0)
                otu["time"] = tag["Host_Age"]
                otu["person"] = tag["Patient_ID"]
                mapping = {'At_Birth': 0, '3_months': 3, '12_months': 12, '24_months': 24}

                # Replace the values in the "time" column
                otu['time'] = otu['time'].replace(mapping)
            elif name == "PRJNA290729":
                if TAX ==7:
                    otu = pd.read_csv("PRJNA290729/tax7/tax7_log_subpca.csv",index_col=0)
                tag = pd.read_csv("PRJNA290729/tax7/meta.csv", index_col=0)
                otu["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
                otu["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
            elif name == "PRJNA273761":
                if TAX == 7:
                    otu = pd.read_csv("PRJNA273761/tax7/tax7_log_subpca.csv",index_col=0)
                tag = pd.read_csv("PRJNA273761/tax7/meta.csv", index_col=0)
                tag["person"] = [i.split("infant ")[-1] for i in tag["isolation_source"]]
                tag["time"] = [i.split(" from infant")[0].replace("fecal sample ", "") for i in tag["isolation_source"]]
                tag = tag[tag["time"] != "fecal sample"]
                tag["time"] = [int(i) for i in tag["time"]]
                common = tag.index.intersection(otu.index)
                tag = tag.loc[common]
                otu = otu.loc[common]
                otu["time"] = tag["time"]
                otu["person"] = tag["person"]
            elif name == "PRJNA301903":
                if TAX == 7:
                    otu = pd.read_csv("PRJNA301903/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJNA301903/tax7/meta.csv", index_col=0)
                # ONLY WGS
                tag = tag[tag["Assay Type"] == "WGS"]
                c = list(otu.index.intersection(tag.index))
                otu = otu.loc[c]
                tag = tag.loc[c]
                otu["person"] = tag["host_subject_id"]
                otu["time"] = tag["host_day_of_life"]
            elif name == "PRJNA510445":
                if TAX == 7:
                    otu = pd.read_csv("PRJNA510445/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJNA510445/tax7/meta.csv", index_col=0)
                otu.index = [i.split("_")[0] for i in otu.index]
                common = list(otu.index.intersection(tag.index))
                tag = tag.loc[common]
                otu = otu.loc[common]
                otu["person"] = [i.split("-")[0] for i in tag["Sample Name"]]
                otu["time"] = [int(i.split("-")[-1].replace("M", "")) for i in tag["Sample Name"]]
            elif name == "PRJNA395569":
                otu = pd.read_csv("PRJNA395569/tax7/tax7_log_subpca.csv",index_col=0)
                tag = pd.read_csv("PRJNA395569/tax7/meta.csv", index_col=0)
                # only WGS
                tag = tag[tag["Assay Type"] == "WGS"]
                otu.index = [i.split("_")[0] for i in otu.index]
                common = list(tag.index.intersection(otu.index))
                otu = otu.loc[common]
                tag = tag.loc[common]
                otu["time"] = tag['Collection_Date']
                otu['time'] = pd.to_datetime(otu['time'])
                otu["person"] = tag["host_subject_id"]
            elif name == "PRJNA806984":
                otu = pd.read_csv("PRJNA806984/tax7/tax7_log_subpca.csv", index_col=0)
                tag = pd.read_csv("PRJNA806984/tax7/meta.csv", index_col=0)
                otu.index = [i.split("_")[0].split(".")[0] for i in otu.index]
                common = list(otu.index.intersection(tag.index))
                otu = otu.loc[common]
                otu = otu.groupby(otu.index).sum()
                tag = tag.loc[common]
                otu["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
                otu["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
                c=0


        else:
            otu = pd.read_csv(f"{name}/otu_common_species.csv",index_col=0)

            if name == "gdm":
                otu["person"] = [i.split("-")[0] for i in otu.index]
                otu["time"] = [i.split("-")[2] for i in otu.index]
            elif name == "allergy":
                meta = pd.read_csv(f"{name}/tax7/metadata_merge_all_with_ok140.csv", index_col=0)
                meta = meta.loc[otu.index]
                otu["person"] = meta["ParticipentCode"]
                otu["time"] = meta["Date"]
                otu = otu.dropna()
                otu['time'] = pd.to_datetime(otu['time'])
            else:
                otu["person"] = [i.split("_")[0] for i in otu.index]
                otu["time"] = [i.split("_")[1] for i in otu.index]

        all_metrics = pd.DataFrame(index=otu.columns,columns=["Average","Median","Std time","Std people","fvec"])
        std_time = calculate_sg1(otu)
        std_people = calculate_sg2(otu)
        avg = otu.mean()
        med = otu.median()
        fvec = calculate_fvec(otu)
        if name == "diab":
            if TAX == 7:
                std_time = std_time.drop(["org_ind","time"])
                std_people = std_people.drop(["org_ind","time"])
                avg = avg.drop(["org_ind","time"])*(10**10)
                fvec = fvec.drop(["org_ind","time"])
                all_metrics= all_metrics.drop(["org_ind","time","person"])
            else:
                avg = avg.drop([ "time"]) * (10 ** 10)

        elif name == "gdm":
            if TAX == 7:
                all_metrics = all_metrics.drop(["time", "person"])
        elif name == "PRJNA395569":
            all_metrics = all_metrics.drop(["time", "person"])
        elif name =="PRJNA806984":
            all_metrics = all_metrics.drop(["time", "person"])
        elif name == "allergy":
            avg = avg.drop(["person"])
            med = med.drop(["person"])
            fvec = fvec.drop(["person"])
            all_metrics = all_metrics.drop(["time", "person"])
        elif name == "PRJNA1130109":
            all_metrics = all_metrics.drop(["time", "person"])
        elif name == "PRJNA290729":
            all_metrics = all_metrics.drop(["time", "person"])
            avg = avg.drop(["time"])
            med = med.drop(["time"])
        else:
            if TAX == 7:
                std_time = std_time.drop(["time"])
                std_people = std_people.drop([ "time"])
                avg = avg.drop(["time"])
                med = med.drop(["time"])
                fvec = fvec.drop(["time"])
                all_metrics = all_metrics.drop(["time", "person"])
            else:
                avg = avg.drop(["time"])
        all_metrics["Average"] = avg
        all_metrics["Median"] = med
        all_metrics["Std time"] = std_time
        all_metrics["Std people"] = std_people
        all_metrics["fvec"] = fvec
        if TAX == 7 or TAX == 8:
            if CONTROL:
                all_metrics.to_csv(f"{name}/tax{TAX}/CONTROL/all_metrics.csv")
            else:
                all_metrics.to_csv(f"{name}/tax{TAX}/all_metrics.csv")
        else:
            all_metrics.to_csv(f"{name}/tax6/all_metrics.csv")





        c=0