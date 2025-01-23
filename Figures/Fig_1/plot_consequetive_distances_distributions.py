import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr
import numpy as np
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
import re
import seaborn as sns


def recognize_staying_bact(df):
    dists = []
    for row in range(len(df.index)-1):
        dist = np.linalg.norm(df.iloc[row+1] - df.iloc[row])
        dists.append(dist)

    return dists

def recognize_disapear(df,bact):
    count_disapear_pairs = list()
    count_t = list()
    for row in range(len(df)-1):
        mid_count = list()
        mid_count_t = list()
        t0 = df[bact].iloc[row]
        t1 = df[bact].iloc[row+1]
        if t0 != 0:
            mid_count_t.append(bact)
            if t1 ==0:
                mid_count.append(bact)
    count_disapear_pairs.append(len(mid_count))
    count_t.append(len(mid_count_t))
    return count_disapear_pairs,count_t

def recognize_appear(df,bact):
    count_appear_pairs = list()
    count_t = list()
    for row in range(len(df) - 1):
        mid_count = list()
        mid_count_t = list()
        t0 = df[bact].iloc[row]
        t1 = df[bact].iloc[row + 1]
        if t0 == 0:
            mid_count_t.append(bact)
            if t1 != 0:
                mid_count.append(bact)
    count_appear_pairs.append(len(mid_count))
    count_t.append(len(mid_count_t))
    return count_appear_pairs, count_t


def find_prob_to_stay(df, NAME):
    list_dists = []

    for p, df_p in df.groupby("person"):
        if len(df_p) < 2:
            continue
        else:
            if NAME == "allergy":
                df_p['time'] = df_p['time'].rank(method='dense').astype(int)
            df_p = df_p.sort_values(by='time')
            del df_p["time"]
            del df_p["person"]
            # Generate distances for each person
            dists = recognize_staying_bact(df_p)
            list_dists.extend(dists)

    # Convert the list of distances to a numpy array
    list_dists = np.array(list_dists).reshape(-1, 1)

    # Fit a Gaussian Mixture Model (GMM) with 2 components
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(list_dists)

    # Predict which Gaussian each point belongs to
    labels = gmm.predict(list_dists)

    # Separate the distances into two groups based on the GMM labels
    list_dists_1 = list_dists[labels == 0]
    list_dists_2 = list_dists[labels == 1]

    # Plot KDE for each group in a different color
    plt.figure(figsize=FIGSIZE)

    # Plot first Gaussian in blue
    sns.kdeplot(list_dists_1.flatten(), color='black', label='G1', fill=True)

    # Plot second Gaussian in red
    sns.kdeplot(list_dists_2.flatten(), color='grey', label='G2', fill=True)

    # Set title based on NAME
    if NAME == "allergy":
        plt.title("Allergy", fontsize=20, fontweight="bold")
    elif NAME == "gdm":
        plt.title("GDM", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA345144":
        plt.title("Eczema", fontsize=20, fontweight="bold")
    elif NAME == "PRJEB6456":
        plt.title("Early dynamics 1", fontsize=20, fontweight="bold")
    elif NAME == "PRJEB39500":
        plt.title("T2D", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA1130109":
        plt.title("Obesity", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA730851":
        plt.title("Infant allergy", fontsize=20, fontweight="bold")
    elif NAME == "PRJEB14529":
        plt.title("Early dynamics 2", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA290729":
        plt.title("Fatty liver", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA273761":
        plt.title("NEC", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA301903":
        plt.title("Preterm infants", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA395569":
        plt.title("IBDMDB", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA510445":
        plt.title("CF", fontsize=20, fontweight="bold")
    elif NAME == "PRJNA806984":
        plt.title("Healthy", fontsize=20, fontweight="bold")

    else:
        plt.title(NAME, fontsize=20, fontweight="bold")

    # Set axis labels and ticks
    plt.xlabel('Distance', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)  # Add a legend for the two Gaussians

    # Save and show the plot
    plt.tight_layout()
    if FIGSIZE == (5, 5):
        plt.savefig(f"outline/supp/dist_hist/for_1/{NAME}_2.png")
    plt.savefig(f"outline/supp/dist_hist/{NAME}_2.png")
    plt.show()

    return list_dists.flatten()



def test_gmm_fit(list_dists, max_components=3):
    # Convert list_dists to a numpy array and reshape it
    list_dists_reshaped = np.array(list_dists).reshape(-1, 1)

    try:
        # Fit a single Gaussian (1 component)
        gmm1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(list_dists_reshaped)
        log_likelihood1 = gmm1.score(list_dists_reshaped) * len(list_dists)
        bic1 = gmm1.bic(list_dists_reshaped)
        aic1 = gmm1.aic(list_dists_reshaped)

        # Fit a mixture of Gaussians (max_components)
        gmmN = GaussianMixture(n_components=max_components, covariance_type='full', random_state=42)
        gmmN.fit(list_dists_reshaped)
        log_likelihoodN = gmmN.score(list_dists_reshaped) * len(list_dists)
        bicN = gmmN.bic(list_dists_reshaped)
        aicN = gmmN.aic(list_dists_reshaped)


        LR_stat = 2 * (log_likelihoodN - log_likelihood1)

        # Degrees of freedom: difference in the number of parameters
        dof = (max_components - 1) * 3  # Assuming 3 parameters (mean, variance, weight) per component
        p_value = chi2.sf(LR_stat, df=dof)

        print(f"Log-likelihood (1 component): {log_likelihood1}")
        print(f"Log-likelihood ({max_components} components): {log_likelihoodN}")
        print(f"BIC ({max_components} components): {bicN}")
        print(f"AIC ({max_components} components): {aicN}")
        print(f"Likelihood Ratio Statistic: {LR_stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print(f"Reject the null hypothesis. A mixture of {max_components} Gaussians fits significantly better.")
        else:
            print(f"Fail to reject the null hypothesis. A single Gaussian might be sufficient.")

            # Extract the means and standard deviations of each component
        means = gmmN.means_.flatten()
        std_devs = np.sqrt(gmmN.covariances_).flatten()

        for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
            print(f"Gaussian {i + 1}: Mean = {mean}, Std Dev = {std_dev}")

    except Exception as e:
        print(f"An error occurred: {e}")


def test_gmm_fit_2_3(list_dists, max_components=3):
    # Convert list_dists to a numpy array and reshape it
    list_dists_reshaped = np.array(list_dists).reshape(-1, 1)

    try:
        # Store results for each number of components
        results = {}

        # Fit GMMs with 1, 2, and 3 components and collect metrics
        for n_components in range(1, max_components + 1):
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42).fit(
                list_dists_reshaped)
            log_likelihood = gmm.score(list_dists_reshaped) * len(list_dists)
            bic = gmm.bic(list_dists_reshaped)
            aic = gmm.aic(list_dists_reshaped)

            results[n_components] = {
                'gmm': gmm,
                'log_likelihood': log_likelihood,
                'bic': bic,
                'aic': aic
            }

        # Print AIC and BIC for all models
        for k, res in results.items():
            print(
                f"Components: {k} | Log-Likelihood: {res['log_likelihood']:.2f} | AIC: {res['aic']:.2f} | BIC: {res['bic']:.2f}")

        # Perform likelihood ratio test for models with increasing components
        for k in range(2, max_components + 1):
            LL1 = results[k - 1]['log_likelihood']
            LL2 = results[k]['log_likelihood']
            LR_stat = 2 * (LL2 - LL1)
            dof = (k - 1) * 3  # 3 parameters per extra component (mean, variance, weight)
            p_value = chi2.sf(LR_stat, df=dof)

            print(f"\nTesting {k - 1} vs {k} components:")
            print(f"Likelihood Ratio Statistic: {LR_stat:.2f}, p-value: {p_value:.4f}")

            if p_value < 0.05:
                print(f"Significant improvement with {k} components.")
            else:
                print(f"No significant improvement beyond {k - 1} components.")

        # Extract and print parameters (means and std devs) of the best model
        best_model = results[max_components]['gmm']
        means = best_model.means_.flatten()
        std_devs = np.sqrt(best_model.covariances_).flatten()

        print("\nBest Model Parameters:")
        for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
            print(f"Gaussian {i + 1}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")

def test_number_samples_in_each_GMM(list_dists, max_components=3):
    # Convert list_dists to a numpy array and reshape it
    list_dists_reshaped = np.array(list_dists).reshape(-1, 1)

    try:
        # Fit a single Gaussian (1 component)
        gmm1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42).fit(list_dists_reshaped)
        log_likelihood1 = gmm1.score(list_dists_reshaped) * len(list_dists)


        # Fit a mixture of Gaussians (max_components)
        gmmN = GaussianMixture(n_components=max_components, covariance_type='full', random_state=42)
        gmmN.fit(list_dists_reshaped)
        log_likelihoodN = gmmN.score(list_dists_reshaped) * len(list_dists)
        bicN = gmmN.bic(list_dists_reshaped)
        aicN = gmmN.aic(list_dists_reshaped)

        # Likelihood ratio test
        LR_stat = 2 * (log_likelihoodN - log_likelihood1)
        dof = (max_components - 1) * 3  # Assuming 3 parameters (mean, variance, weight) per component
        p_value = chi2.sf(LR_stat, df=dof)

        print(f"Log-likelihood (1 component): {log_likelihood1}")
        print(f"Log-likelihood ({max_components} components): {log_likelihoodN}")
        print(f"BIC ({max_components} components): {bicN}")
        print(f"AIC ({max_components} components): {aicN}")
        print(f"Likelihood Ratio Statistic: {LR_stat}")
        print(f"p-value: {p_value}")
        if p_value < 0.05:
            print(f"Reject the null hypothesis. A mixture of {max_components} Gaussians fits significantly better.")
        else:
            print(f"Fail to reject the null hypothesis. A single Gaussian might be sufficient.")

        # Extract the means and standard deviations of each component
        means = gmmN.means_.flatten()
        std_devs = np.sqrt(gmmN.covariances_).flatten()

        for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
            print(f"Gaussian {i + 1}: Mean = {mean}, Std Dev = {std_dev}")

        # Get the probability that each sample belongs to each Gaussian
        probas = gmmN.predict_proba(list_dists_reshaped)

        # Calculate the fraction of samples that belong to the first Gaussian
        fraction_first_gaussian = np.mean(probas[:, 0])
        print(f"Fraction of samples belonging to the first Gaussian: {fraction_first_gaussian}")

        # Calculate the number of samples that belong to the second Gaussian
        samples_second_gaussian = np.sum(probas[:, 1] > 0.5)
        print(f"Number of samples belonging predominantly to the second Gaussian: {samples_second_gaussian}")

    except Exception as e:
        print(f"An error occurred: {e}")


def separate_number_letter(index):
    match = re.match(r'(\d+)([A-Za-z]?)', index)
    if match:
        number = match.group(1)
        letter = match.group(2)
        return number, letter
    return None, None


def find_prob_disapearing(df,title,NAME,freq=True):
    no_zeros = True
    SIZE = 15
    t_df = pd.DataFrame(index=list(set(df["person"])), columns=df.columns)
    dis_df = pd.DataFrame(index=list(set(df["person"])), columns=df.columns)
    for p, df_p in df.groupby("person"):
        if len(df_p) < 2:
            continue
        else:
            if NAME == "allergy":
                df_p['time'] = df_p['time'].rank(method='dense').astype(int)
            df_p = df_p.sort_values(by='time')
            del df_p["time"]
            del df_p["person"]
            for bact in df_p.columns:
                count_dis,count_t = recognize_disapear(df_p, bact)
                n_dis = sum(count_dis)
                n_t = sum(count_t)
                dis_df[bact][p] =n_dis
                t_df[bact][p] =n_t
    del dis_df["person"]
    del dis_df["time"]
    # calc prob
    dis = dis_df.sum()
    t = t_df.sum()
    t = t[t.values != 0]
    dis = dis.loc[t.index]
    probs = dis /t
    # Plot hist
    plt.hist(probs)
    plt.xlabel("Probability to disappear", fontsize=SIZE)
    plt.ylabel("Frequency", fontsize=SIZE)
    plt.xticks(fontsize=SIZE)
    plt.yticks(fontsize=SIZE)
    plt.title(f"{NAME.upper()}-{title}",fontsize=SIZE,fontweight="bold")
    plt.tight_layout()
    if CONTROL:
        plt.savefig(f"step_2_results/{NAME.upper()}-{title}_dis_CONTROL.png")
    else:
        plt.savefig(f"step_2_results/{NAME.upper()}-{title}_dis.png")
    plt.show()
    if freq:
        del df["person"]
        del df["time"]
        avg_freq = df.mean()
        if no_zeros:
            df_no_zeros = df.replace(0, pd.NA)
            avg_freq = df_no_zeros.mean()
        avg_freq = avg_freq.groupby(avg_freq.index).mean()
        common = list(avg_freq.index.intersection(probs.index))
        avg_freq = avg_freq.loc[common]
        scc, p = spearmanr(probs, avg_freq)
        plt.scatter(probs,avg_freq,label=f"SCC: {round(scc,3)} p-value: {round(p,3)}")
        plt.legend(fontsize=SIZE)
        plt.xlabel("Probability to disappear", fontsize=SIZE)
        plt.ylabel("Average species' frequency", fontsize=SIZE)
        plt.xticks(fontsize=SIZE)
        plt.yticks(fontsize=SIZE)
        plt.title(f"{NAME.upper()}-{title}", fontsize=SIZE, fontweight="bold")
        plt.tight_layout()
        if CONTROL:
            plt.savefig(f"step_2_results/{NAME.upper()}-{title}_scatter_no_zeros_CONTROL.png")
        else:
            plt.savefig(f"step_2_results/{NAME.upper()}-{title}_scatter_no_zeros.png")
        plt.show()


def find_avg_distance_between_time_steps(df, NAME):
    distance_df = pd.DataFrame(index=list(set(df["person"])), columns=df.columns)

    for p, df_p in df.groupby("person"):
        if len(df_p) < 2:
            continue
        else:
            if NAME == "allergy":
                df_p['time'] = df_p['time'].rank(method='dense').astype(int)
            df_p = df_p.sort_values(by='time')

            for bact in df_p.columns.drop(['time', 'person']):
                time_diffs = df_p[bact].diff().dropna()
                if not time_diffs.empty:
                    avg_distance = time_diffs.mean()
                    distance_df[bact][p] = avg_distance

    # Calculate the average distance across the population for each taxon
    avg_distance_per_taxon = distance_df.mean()
    avg_distance_per_taxon = avg_distance_per_taxon.dropna()
    avg_distance_per_taxon.to_csv(f"outline/staying_DIST/{NAME}.csv")
    return avg_distance_per_taxon

def find_prob_appearing(df,title,NAME):
    SIZE = 15
    t_df = pd.DataFrame(index=list(set(df["person"])), columns=df.columns)
    dis_df = pd.DataFrame(index=list(set(df["person"])), columns=df.columns)
    for p, df_p in df.groupby("person"):
        if len(df_p) < 2:
            continue
        else:
            if NAME == "allergy":
                df_p['time'] = df_p['time'].rank(method='dense').astype(int)
            df_p = df_p.sort_values(by='time')
            del df_p["time"]
            del df_p["person"]
            for bact in df_p.columns:
                count_dis, count_t = recognize_appear(df_p, bact)
                n_dis = sum(count_dis)
                n_t = sum(count_t)
                dis_df[bact][p] = n_dis
                t_df[bact][p] = n_t
    del dis_df["person"]
    del dis_df["time"]
    # calc prob
    dis = dis_df.sum()
    t = t_df.sum()
    t = t[t.values != 0]
    dis = dis.loc[t.index]
    probs = dis / t
    # Plot hist
    plt.hist(probs)
    plt.xlabel("Probability to appear", fontsize=SIZE)
    plt.ylabel("Frequency", fontsize=SIZE)
    plt.xticks(fontsize=SIZE)
    plt.yticks(fontsize=SIZE)
    plt.title(f"{NAME.upper()}-{title}",fontsize=SIZE,fontweight="bold")
    plt.tight_layout()
    if CONTROL:
        plt.savefig(f"step_2_results/{NAME.upper()}-{title}_app_CONTROL.png")
    else:
        plt.savefig(f"step_2_results/{NAME.upper()}-{title}_app.png")
    plt.show()
    return probs


def check_chi(probs_01, probs_011, df):
    N = len(df)
    common = list(probs_01.index.intersection(probs_011.index))
    df = df[common]
    probs_01 = probs_01.loc[common]
    probs_001 = probs_011.loc[common]

    p1 = (df != 0.0).sum() / len(df)
    E = probs_01 * p1 * N
    O = probs_001 * N

    # Handle zero expected values
    non_zero_E_mask = E != 0
    E_non_zero = E[non_zero_E_mask]
    O_non_zero = O[non_zero_E_mask]

    # Calculate the Chi-squared statistic for non-zero expected values
    chi_square_stat = (O_non_zero - E_non_zero) ** 2 / E_non_zero
    # Ensure the chi_square_stat contains valid numbers
    chi_square_stat = chi_square_stat.astype(float)
    p_values = 1 - chi2.cdf(chi_square_stat, df=1)

    # Initialize arrays to hold results
    chi_square_stat_all = np.zeros_like(E)
    p_values_all = np.ones_like(E)

    # Fill the arrays with computed values where E is non-zero
    chi_square_stat_all[non_zero_E_mask] = chi_square_stat
    p_values_all[non_zero_E_mask] = p_values

    # Multiple testing correction (Benjamini-Hochberg)
    _, corrected_p_values_all, _, _ = multipletests(p_values_all, method='fdr_bh')

    # Create a DataFrame with the results
    results = pd.DataFrame({
        'O': O,
        'E': E,
        'O-E': O - E,
        'Chi-squared Statistic': chi_square_stat_all,
        'P-value': p_values_all,
        'Corrected P-value': corrected_p_values_all
    }, index=common)

    return results

def adjust_time(df):
    df = df.copy()  # Make a copy of the dataframe to avoid modifying the original
    df['time'] = df.groupby('person').cumcount() + 1  # +1 to start counting from 1
    return df



if __name__ == "__main__":
    mpl.rc('font', family='Arial')
    FIGSIZE = (5,5)
    LIST_NAME =["gdm","allergy","PRJNA345144","PRJEB6456","PRJEB39500","PRJNA1130109","PRJNA730851","PRJEB14529","PRJNA972625","PRJNA290729","PRJNA273761","PRJNA301903","PRJNA395569","PRJNA510445","PRJNA806984"] #["mucositis","gdm","allergy","diab","PRJEB6456","PRJEB39500","PRJNA1130109","PRJNA730851","PRJEB14529",PRJNA345144,"PRJNA972625"]
    CONTROL = False
    for tax in [7]:#6,
        if tax == 7:
            title = "species"
        else:
            title = "genus"

        for NAME in LIST_NAME:
            print(NAME)
            df = pd.read_csv(f"{NAME}/tax{tax}/tax{tax}_relative_sum.csv",index_col=0)
            if NAME == "gdm":
                df["person"] = [i.split("-")[0] for i in df.index]
                df["time"] = [i.split("-")[-1] for i in df.index]
                if CONTROL:
                    meta = pd.read_csv(f"{NAME}/tax{tax}/mapping.csv",index_col=0)
                    meta["name"] = meta["Group"]+"-STOOL-"+meta["trimester"]
                    meta.index = meta["name"]
                    meta0 = meta[meta["Tag"] ==0.0]
                    df = df.loc[meta0.index]

            elif NAME =="mucositis":
                df["person"] = [i.split("W")[0] for i in df.index]
                df["time"] = [int(i.split("W")[1]) for i in df.index]
            elif NAME == "allergy":
                meta = pd.read_csv(f"{NAME}/tax{tax}/metadata_merge_all_with_ok140.csv",index_col=0)
                meta = meta.loc[df.index]
                df["person"] = meta["ParticipentCode"]
                df["time"] = meta["Date"]
                df = df.dropna()
                df['time'] = pd.to_datetime(df['time'])
            elif NAME == "diab":
                meta = pd.read_csv(f"{NAME}/tax{tax}/metadata_diabimmune.csv",index_col=0)
                meta.index = meta["SampleID"]
                meta = meta.loc[df.index]
                df["person"] = meta["subjectID"]
                df["time"] = meta["collection_month"]
                if CONTROL:
                    meta["total_allergy"] = meta[['allergy_milk', 'allergy_egg', 'allergy_peanut', 'allergy_dustmite','allergy_cat', 'allergy_dog', 'allergy_birch', 'allergy_timothy']].sum(axis=1)
                    meta0 = meta[meta["total_allergy"] == 0.0]
                    df = df.loc[meta0.index]

            elif NAME == "pnas":
                meta = pd.read_csv(f"{NAME}/tax{tax}/pnas_meta_data.csv",index_col=0)
            elif NAME == "PRJEB6456":
                meta = pd.read_csv(f"{NAME}/tax{tax}/tag.csv",index_col=0)
                df["person"] = [i.split("_")[0] for i in meta["Sample Name"]]
                df["time"] = [i.split("_")[-1] for i in meta["Sample Name"]]
                # Create a dictionary to map the values
                mapping = {'B': 0, 'M': 1, '4M': 4, '12M': 12}

                # Replace the values in the "time" column
                df['time'] = df['time'].replace(mapping)
            elif NAME == "PRJEB14529":
                meta = pd.read_csv("PRJEB14529/tax7/tag.csv", index_col=0)

                # Remove negative times
                meta = meta[meta["month_of_life"] >= 0.0]
                df = df.loc[meta.index]
                df["person"] = meta["host_subject_id"]
                df["time"] = meta["month_of_life"]
                meta["name"] = [str(df["person"].loc[i]) + "-" + str(df["time"].loc[i]) for i in df.index]
                df.index = meta["name"]
                df = df.groupby(df.index).first()
            elif NAME == "PRJNA972625":
                tag = pd.read_csv("PRJNA972625/tax7/meta.csv", index_col=0)
                tag['Sample Name'] = [i.split("_")[-1] for i in tag['Sample Name']]
                tag[['person', 'time']] = tag['Sample Name'].apply(lambda x: pd.Series(separate_number_letter(x)))
                tag["time"] = tag['time'].apply(lambda x: 0.0 if x is '' else x)
                mapping = {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'E': 5.0, 'F': 6.0, 'G': 7.0}
                # Replace the values in the "time" column
                tag['time'] = tag['time'].replace(mapping)
                df["person"] = tag["person"]
                df["time"] = tag["time"]
            elif NAME == "PRJEB39500":
                tag = pd.read_csv("PRJEB39500/tax7/tag.csv", index_col=0)
                df.index = [i.split(".")[0] for i in df.index]
                cc = list(df.index.intersection(tag.index))
                df = df.loc[cc]
                tag = tag.loc[cc]
                df["person"] = [i.split("_")[0] for i in tag["Sample_Name"]]
                df["time"] = [i.split("_")[-1].split("-")[0] for i in tag["Sample_Name"]]
                mapping = {'M0': 0, 'M7d': 2, 'M24h': 1}

                # Replace the values in the "time" column
                df['time'] = df['time'].replace(mapping)
            elif NAME == "PRJNA1130109":
                tag = pd.read_csv("PRJNA1130109/tax7/tag.csv",index_col=0)
                df["person"] = [i.split("-")[0] for i in tag["Sample Name"]]
                df["time"] = [i.split("tp")[-1] for i in tag["timepoint"]]
            elif NAME == "PRJNA730851":
                df.index = [i.split("_")[0] for i in df.index]
                df = df.groupby(df.index).sum()
                tag = pd.read_csv("PRJNA730851/tax7/tag.csv",index_col=0)

                df["person"] = [i.split(".")[1] for i in tag["Sample Name"]]
                df["time"] = [int(i.split(".")[2]) for i in tag["Sample Name"]]
            elif NAME =="PRJNA345144":
                tag = pd.read_csv("PRJNA345144/tax7/tag.csv",index_col=0)
                df["time"] = tag["Host_Age"]
                df["person"] = tag["Patient_ID"]
                mapping = {'At_Birth': 0, '3_months': 3, '12_months': 12,'24_months':24}

                # Replace the values in the "time" column
                df['time'] = df['time'].replace(mapping)
            elif NAME == "PRJEB61646":
                tag = pd.read_csv("PRJEB61646/tax7/tag.csv",index_col=0)
                common = list(df.index.intersection(tag.index))
                df["person"] = tag["Sample Name"]
                df = adjust_time(df)
            elif NAME == "PRJNA290729":
                tag = pd.read_csv("PRJNA290729/tax7/meta.csv", index_col=0)
                df["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
                df["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
            elif NAME == "PRJNA273761":
                tag = pd.read_csv("PRJNA273761/tax7/meta.csv", index_col=0)
                tag["person"] = [i.split("infant ")[-1] for i in tag["isolation_source"]]
                tag["time"] = [i.split(" from infant")[0].replace("fecal sample ", "") for i in tag["isolation_source"]]
                tag = tag[tag["time"] != "fecal sample"]
                tag["time"] = [int(i) for i in tag["time"]]
                common = tag.index.intersection(df.index)
                tag = tag.loc[common]
                df = df.loc[common]
                df["time"] = tag["time"]
                df["person"] = tag["person"]
            elif NAME == "PRJNA301903":
                tag = pd.read_csv("PRJNA301903/tax7/meta.csv", index_col=0)
                # ONLY WGS
                tag = tag[tag["Assay Type"] == "WGS"]
                c = list(df.index.intersection(tag.index))
                tag = tag.loc[c]
                df = df.loc[c]
                df["person"] = tag["host_subject_id"]
                df["time"] = tag["host_day_of_life"]
            elif NAME == "PRJNA806984":
                tag = pd.read_csv("PRJNA806984/tax7/meta.csv", index_col=0)
                df.index = [i.split("_")[0].split(".")[0] for i in df.index]
                common = list(df.index.intersection(tag.index))
                df = df.loc[common]
                df = df.groupby(df.index).sum()
                tag = tag.loc[common]
                df["person"] = [i.split("_")[0] for i in tag["Sample Name"]]
                df["time"] = [i.split("_")[-1] for i in tag["Sample Name"]]
            elif NAME == "PRJNA395569":
                tag = pd.read_csv("PRJNA395569/tax7/meta.csv", index_col=0)
                # only WGS
                tag = tag[tag["Assay Type"] == "WGS"]
                df.index = [i.split("_")[0] for i in df.index]
                common = list(tag.index.intersection(df.index))
                df = df.loc[common]
                tag = tag.loc[common]
                df["time"] = tag['Collection_Date']
                df['time'] = pd.to_datetime(df['time'])
                df["person"] = tag["host_subject_id"]
            elif NAME == "PRJNA510445":
                tag = pd.read_csv("PRJNA510445/tax7/meta.csv", index_col=0)
                df.index = [i.split("_")[0] for i in df.index]
                common = list(df.index.intersection(tag.index))
                tag = tag.loc[common]
                df = df.loc[common]
                df["person"] = [i.split("-")[0] for i in tag["Sample Name"]]
                df["time"] = [int(i.split("-")[-1].replace("M", "")) for i in tag["Sample Name"]]


            list_dists = find_prob_to_stay(df,NAME)
            print(NAME)
            test_gmm_fit_2_3(list_dists, max_components=3)
            test_number_samples_in_each_GMM(list_dists, max_components=2)





