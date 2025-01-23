import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import ttest_ind
from statannot import add_stat_annotation

def plot_swarm(ax, LIST_FEATURES, coefs, palette, y_label,y_limits, title, show_xticklabels=False):
    boxes = [coefs[feature] for feature in LIST_FEATURES]

    aub_indices = [0, 1]  # Indices for AUB categories
    cud_indices = [2, 3]
    others_indices = [4, 5, 6, 7, 8]

    aub_data = [value for i in aub_indices for value in boxes[i]]
    cud_data = [value for i in cud_indices for value in boxes[i]]
    others_data = [value for i in others_indices for value in boxes[i]]

    t_stat, p_value = ttest_ind(aub_data, cud_data, equal_var=False)
    t_stat_, p_value_ = ttest_ind(others_data, cud_data, equal_var=False)

    sns.swarmplot(data=boxes, palette=palette, ax=ax)
    sns.boxplot(data=boxes, ax=ax, showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black'))

    add_stat_annotation(ax,
                        data=pd.DataFrame(data=boxes,
                                          index=["A", "B", "C", "D", "E", "F", "G", "H", "I"]).T,
                        box_pairs=[("A", "D"), ("I", "D")],
                        perform_stat_test=False,
                        pvalues=[p_value, p_value_],
                        test=None,
                        text_format='star',
                        loc='inside',
                        verbose=2,
                        fontsize='large')

    for text in ax.texts:
        if "*" in text.get_text():
            text.set_fontsize(18)

    if show_xticklabels:
        ax.set_xticks(np.arange(len(LIST_FEATURES)))
        ax.set_xticklabels(LIST_FEATURES, fontsize=20, family='Arial', rotation=90)
    else:
        ax.set_xticks([])

    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_label_coords(-0.15, 0.5)


def plot_swarm1(ax, LIST_FEATURES, coefs, y_label,y_limits, LIST_SIM_COEFFS, LIST_SIM_STDS, core=False, show_xticklabels=False):
    # Define the custom palette
    if core:
        custom_palette = ["darkorange", "royalblue", "royalblue", "limegreen", "limegreen", "limegreen", "limegreen",
                          "limegreen", "hotpink"]
    else:
        custom_palette = ["darkorange", "darkorange", "royalblue", "royalblue", "limegreen", "limegreen", "limegreen",
                          "limegreen", "limegreen"]

    # Prepare data for the swarm and box plot
    boxes = [coefs[c] for c in LIST_FEATURES]

    # Combine data into AUB and CUD groups
    aub_indices = [0, 1]
    cud_indices = [2, 3]
    others_indices = [4, 5, 6, 7, 8]

    aub_data = [value for i in aub_indices for value in boxes[i]]
    cud_data = [value for i in cud_indices for value in boxes[i]]
    others_data = [value for i in others_indices for value in boxes[i]]

    t_stat, p_value = ttest_ind(aub_data, cud_data, equal_var=False)
    t_stat_, p_value_ = ttest_ind(others_data, cud_data, equal_var=False)

    # Create the swarm plot
    sns.swarmplot(data=boxes, palette=custom_palette, ax=ax)

    # Create the box plot
    sns.boxplot(data=boxes, ax=ax, showfliers=False, boxprops=dict(facecolor='white', edgecolor='black'))

    # Add the red bars for LIST_SIM_COEFFS with error bars for LIST_SIM_STDS
    x_positions = np.arange(len(LIST_SIM_COEFFS))
    ax.bar(x_positions, LIST_SIM_COEFFS, width=0.3, color='red', alpha=0.5, yerr=LIST_SIM_STDS, capsize=5,
           edgecolor='black', error_kw=dict(ecolor='red', lw=2))

    # Add statistical annotations
    add_stat_annotation(ax,
                        data=pd.DataFrame(data=boxes,
                                          index=["A", "B", "C", "D", "E", "F", "G", "H", "I"]).T,
                        box_pairs=[("A", "D"), ("I", "D")],
                        perform_stat_test=False,
                        pvalues=[p_value, p_value_],
                        test=None,
                        text_format='star',
                        loc='inside',
                        verbose=2,
                        fontsize='large')

    for text in ax.texts:
        if "*" in text.get_text():
            text.set_fontsize(18)

    # Customize the ticks and labels
    if show_xticklabels:
        ax.set_xticks(np.arange(len(LIST_FEATURES)))
        ax.set_xticklabels(LIST_FEATURES, fontsize=20, family='Arial', rotation=90)
    else:
        ax.set_xticks([])

    ax.set_ylabel(y_label, fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_label_coords(-0.15, 0.5)


def main():
    alphabet = "CDEFGHIJKLMNOPQRSTUVWXYZ"
    mpl.rc('font', family='Arial')

    CORE = False
    LIST_FEATURES = ["Average", "Median", "Std time", "Std people", "fvec", "In-degree", "Out-degree", "Betweeness", "Closeness"]
    FIXED_LIST_FEATURES = ["Avg. abundance","Med. abundance","S.D. time","S.D. people","Centrality 1","Centrality 2","Centrality 3","Centrality 4","Centrality 5"]
    LIST_SIM_COEFFS = [-1.26403126, -1.19672316, 0.62440778, 2.23420323, 0.08272307, 0.1646745,
                       0.36547394, 0.20477729, -0.35508581]
    LIST_SIM_STDS = [0.06588279, 0.05898468, 0.15590326, 0.11733244, 0.12211742, 0.09113237,
                     0.08924931, 0.19242778, 0.09348941]
    tasks = ["Changing ME", "RPT", "iFMT"]
    models = ["LR", "RF", "XGBOOST"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), sharex=True)
    custom_palette = ["darkorange", "darkorange", "royalblue", "royalblue", "limegreen", "limegreen", "limegreen", "limegreen", "limegreen"]

    for i, task in enumerate(tasks):
        for j, model in enumerate(models):
            ax = axes[i, j]

            if task == "Changing ME":
                if model != "LR":
                    #coefs = pd.read_csv(f"outline/only_wgs/staying_med/tax_8coef_{model}.csv", index_col=0)
                    coefs = pd.read_csv(f"outline/staying_ABOVE_med/coef_{model}_8.csv", index_col=0)
                else:
                    coefs = pd.read_csv(f"outline/staying_ABOVE_med/coef_LR_8.csv", index_col=0)

                coefs = coefs.rename(
                    columns={"Average": "Avg. abundance", "Median": "Med. abundance", "Std time": "S.D. time",
                             "Std people": "S.D. people", "fvec": "Centrality 1", "In-degree": "Centrality 2",
                             "Out-degree": "Centrality 3", "Betweeness": "Centrality 4", "Closeness": "Centrality 5"})

            else:
                coefs = pd.read_csv(f"outline/med_new_wgs/{task}/tax8_coef_{model}.csv", index_col=0)
                coefs = coefs.rename(
                    columns={"Average": "Avg. abundance", "Median": "Med. abundance", "Std time": "S.D. time",
                             "Std people": "S.D. people", "fvec": "Centrality 1", "In-degree": "Centrality 2",
                             "Out-degree": "Centrality 3", "Betweeness": "Centrality 4", "Closeness": "Centrality 5"})

            y_label = "Coefficients" if model == "LR" else "Feature importance"
            if model == "LR":
                y_limits = (-1.0,3.0)
            elif model == "RF":
                y_limits = (0.0,0.5)
            elif model == "XGBOOST":
                y_limits = (0.0,0.7)
            title = f"{task} - {model}"
            show_xticklabels = (i == 2)
            if i ==0 and j==0:
                plot_swarm1(ax, FIXED_LIST_FEATURES, coefs, y_label,y_limits, LIST_SIM_COEFFS, LIST_SIM_STDS, core=False, show_xticklabels=show_xticklabels)
            else:
                plot_swarm(ax, FIXED_LIST_FEATURES, coefs, custom_palette, y_label,y_limits, title, show_xticklabels=show_xticklabels)
            # Add a large letter to each subplot
            letter = alphabet[i * 3 + j]
            ax.text(-0.1, 1.1, letter, transform=ax.transAxes, fontsize=20, fontfamily='Times New Roman',
                        weight='bold', va='top', ha='left')
    plt.tight_layout()
    plt.savefig("outline/figures_st/Fig_3/combined_swarmplot_3_3.png")
    plt.show()

if __name__ == '__main__':
    main()
