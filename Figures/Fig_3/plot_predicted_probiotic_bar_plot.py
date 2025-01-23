import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
def fix_bact_issue(df):
    df["Taxa"] = [i.replace("p__Bacteroidota", "p__Bacteroidetes") for i in df["Taxa"]]
    df1 = df[["Taxa", "Avg_Patchy_Score"]]
    df2 = df[["Taxa", "Count", "Count_Total"]]
    df1 = df1.groupby(df2["Taxa"]).mean()
    df2 = df2.groupby(df2["Taxa"]).sum()
    df2["Avg_Patchy_Score"] = df1["Avg_Patchy_Score"]
    df2["Fraction"] = df2["Count"] / df2["Count_Total"]
    df2 = df2.sort_values("Count", ascending=False)
    df2["Taxa"] =df2.index
    return df2
def clean_names(df):
    new_names = []
    for i in df["Taxa"]:
        if len(i.split(";")) < 8:
            t = i.split(";")[-1]
        else:
            t = i.split(";")[-2] + ";" + i.split(";")[-1]
        new_names.append(t)
    df["Taxa"] = new_names
    return df

def clean_names_list(l):
    new_names = []
    for i in l:
        if len(i.split(";")) < 8:
            t = i.split(";")[-1]
        else:
            t = i.split(";")[-2] + ";" + i.split(";")[-1]
        new_names.append(t)

    return new_names

def fix_names(cosp_10):
    fixed = []
    for i in list(cosp_10["Taxa"].values):
        if "t__" in i:
            i = i + " (t)"
        else:
            i = i + " (s)"
        fixed.append(i)
    fixed = [i.replace("s__", "").replace(";t__", " ").replace("_", " ") for i in fixed]
    cosp_10["fTaxa"] = fixed
    return cosp_10


def clean_plot(cosp_10,ngs_list,lit,save,xlabel):
    lit_common_p = list(set(cosp_10["Taxa"]).intersection(lit.index))
    cosp_10 = clean_names(cosp_10)
    common_clean = clean_names_list(lit_common_p)
    cosp_10 = cosp_10.sort_values("Avg_Patchy_Score", ascending=True)
    # Plot for cos_p (std_p)
    fig, ax1 = plt.subplots(figsize=(10, 15))



    # Plot "Avg_Patchy_Score" from `cosp_10` as horizontal bars
    cosp_10 = fix_names(cosp_10)
    cosp_10 = cosp_10.groupby(cosp_10["fTaxa"]).max()
    cosp_10 = cosp_10.sort_values("Avg_Patchy_Score", ascending=True)
    colors = [
        'deeppink' if taxa in common_clean
        else 'mediumpurple' if any(ngs in taxa for ngs in ngs_list)
        else 'royalblue'
        for taxa in cosp_10["Taxa"]
    ]
    ax1.barh(cosp_10.index, cosp_10["Avg_Patchy_Score"], color=colors, alpha=0.5, label="Avg SM Score")
    ax1.set_xlabel(xlabel, fontsize=20, color="black")
    ax1.tick_params(axis="x", labelsize=20, colors="black")
    ax1.tick_params(axis="y", labelsize=19, labelright=True, labelleft=False)
    ax1.set_ylim(-0.5, len(cosp_10.index) - 0.5)

    # Create a secondary x-axis for "Fraction"
    ax2 = ax1.twiny()
    ax2.plot(cosp_10["Fraction"], cosp_10.index, 'o', color="royalblue", label="Fraction")
    ax2.set_xlabel("Fraction", fontsize=20, color="black")
    ax2.tick_params(axis="x", labelsize=20, colors="black")


    # Tight layout
    plt.tight_layout()
    plt.savefig(save)#
    plt.show()

def remove_not_leaves_species(df):
    df["level"] = [len(i.split(";")) for i in df.index]
    species = df[df["level"]==7]
    strains = df[df["level"]==8]
    names_to_drop = []
    for s in species.index:
        for t in strains.index:
            if s in t:
                names_to_drop.append(s)
    names_to_drop = list(set(names_to_drop))
    species = species.drop(names_to_drop)
    fixed = pd.concat([strains,species])
    return fixed




# Set font
mpl.rc('font', family='Arial')

# Load data
lit = pd.read_csv("outline/figures/Fig_4/all/from_lit_patchy_scores_p.csv",index_col=0)
cos_p = pd.read_csv("outline/figures/Fig_4/all/tax8_no_genus_patchy_std_p_df_no_percentile.csv", index_col=0)

cos_p = fix_bact_issue(cos_p)
cos_t = pd.read_csv("outline/figures/Fig_4/all/tax8_no_genus_patchy_std_t_df_no_percentile.csv", index_col=0)

cos_t = fix_bact_issue(cos_t)
cos_p = remove_not_leaves_species(cos_p)
cos_t = remove_not_leaves_species(cos_t)

cosp_10 = cos_p[cos_p["Count"]>2][cos_p["Fraction"]>=0.6][["Taxa", "Avg_Patchy_Score", "Fraction"]]
cost_10 = cos_t[cos_t["Count"]>2][cos_t["Fraction"]>=0.6][["Taxa", "Avg_Patchy_Score", "Fraction"]]

ngs_list = ["s__Faecalibacterium_prausnitzii","s__Akkermansia_muciniphila","s__Bacteroides_fragilis","s__Roseburia_intestinalis","s__Prevotella_copri","s__Clostridium_butyricum","s__Parabacteroides_distasonis","s__Veillonella_sp_CAG_933"]
clean_plot(cosp_10,ngs_list,lit,"outline/figures/Fig_4/tax8_top10_cosp.png","S.D. people")
clean_plot(cost_10,ngs_list,lit,"outline/figures/Fig_4/tax8_top10_cost.png","S.D. time")

