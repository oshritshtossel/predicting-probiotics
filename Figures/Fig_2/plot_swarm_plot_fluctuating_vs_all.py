import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import matplotlib as mpl


# Function to add significance stars
def add_significance_annotation(ax, p_values, positions):
    for i, p_val in enumerate(p_values):
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
        else:
            stars = "ns"

        x1, x2 = positions[i]
        y = data['AUC'].max() + 0.02
        h = 0.02 if stars != "ns" else -0.02  # Lower "ns" by using negative h
        ax.plot([x1, x2], [y, y], color='black', lw=1.5)
        ax.text((x1 + x2) * 0.5, y + h, stars, ha='center', va='bottom', color='black', fontsize=16)


if __name__ == "__main__":
    mpl.rc('font', family='Arial')
    LIST_TASKS = ["Changing ME", "RPT", "iFMT"]
    LIST_MODELS = ["LR", "RF", "XGBOOST"]

    data_list = []

    for TASK in LIST_TASKS:
        for MODEL in LIST_MODELS:
            if TASK == "RPT" or TASK == "iFMT":
                patchy = pd.read_csv(f"outline/figures/Fig_3/{TASK}/tax8_aucs_{MODEL}_patchy_wgs.csv", index_col=0)["Average AUC"]
                all_data = pd.read_csv(f"outline/med_new_wgs/{TASK}/tax8_aucs_{MODEL}.csv", index_col=0)["Average AUC"]
            else:
                patchy = pd.read_csv(f"outline/figures/Fig_3/wgs/Changing ME/tax8_aucs_{MODEL}_patchy.csv",index_col=0)["Average AUC"]
                all_data = pd.read_csv(f"outline/only_wgs/staying_med/tax_8_aucs_{MODEL}.csv",index_col=0)["Average AUC"]



            patchy_df = pd.DataFrame({
                'AUC': patchy,
                'Type': 'Fluctuating',
                'Model': MODEL,
                'Task': TASK
            })

            all_df = pd.DataFrame({
                'AUC': all_data,
                'Type': 'All',
                'Model': MODEL,
                'Task': TASK
            })

            data_list.append(patchy_df)
            data_list.append(all_df)

    data = pd.concat(data_list)

    # Create a 'Model_Task' column for x-axis
    data['Model_Task'] = data['Model'] + "\n" + data['Task']

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Create the boxplot with no colors (transparent)
    sns.boxplot(x="Model_Task", y="AUC", hue="Type", data=data,
                palette=["white", "white"],  # White color to remove box color
                dodge=True, linewidth=1.5, showfliers=False)  # No outliers

    # Overlay the swarm plot on top of the boxplot
    sns.swarmplot(x="Model_Task", y="AUC", hue="Type", data=data,
                  palette={"Fluctuating": "royalblue", "All": "black"},
                  dodge=True, edgecolor="k", size=6)

    # Set xtick labels for models only (LR, RF, XGBOOST) without task labels
    model_ticks = ["LR", "RF", "XGBOOST"] * len(LIST_TASKS)  # Repeated for each task
    plt.xticks(ticks=[i for i in range(len(model_ticks))], labels=model_ticks, fontsize=15)

    plt.ylabel("Average AUC", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("")

    # Remove duplicate legends (by hue in boxplot)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend().remove()  # Remove the default legend

    # Custom legend for dots (Patchy and All)
    plt.legend(handles[2:4], labels[2:4], title="Data Type", loc="lower left",fontsize=20)

    # Significance testing between Patchy and All for each model-task pair
    p_values = []
    positions = []
    for i, task in enumerate(LIST_TASKS):
        for j, model in enumerate(LIST_MODELS):
            task_model_data = data[(data['Task'] == task) & (data['Model'] == model)]
            patchy_data = task_model_data[task_model_data['Type'] == 'Fluctuating']['AUC']
            all_data = task_model_data[task_model_data['Type'] == 'All']['AUC']

            stat, p_value = mannwhitneyu(patchy_data, all_data)
            p_values.append(p_value)
            positions.append((j + i * 3 - 0.2, j + i * 3 + 0.05))  # Adjust x-coordinates for annotations

    # Add significance annotations
    add_significance_annotation(plt.gca(), p_values, positions)

    # Tight layout and show plot
    plt.tight_layout()
    plt.savefig("outline/figures/Fig_3/patchy_swarm_wgs_strains.png")
    plt.show()

