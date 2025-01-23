import pandas as pd
import matplotlib.pyplot as plt

# Set the font to Arial and figure size
SIZE = 20
plt.rcParams['font.family'] = 'Arial'

# Load the dataset
df = pd.read_csv("nadav_check/figures/updated/Fig_1/fractions_gaussians.csv", index_col=0)
# filter for 16S datasets
df = df.drop(["GDM","Early dynamics 2","Infant allergy"],axis=0)
# Plot the horizontal stacked bar plot
ax = df.plot(kind="barh", stacked=True, color=['black', 'grey'], figsize=(10, 5))  # Stacked horizontal bar plot
plt.xlabel("Fraction of samples per gaussian", fontsize=SIZE)
plt.ylabel(" ", fontsize=SIZE)
plt.xticks(fontsize=SIZE)
plt.yticks(fontsize=SIZE)
plt.tight_layout()

# Add stars to the bars (3 stars for all datasets except "Obesity" which gets 1 star)
for idx, dataset_name in enumerate(df.index):
    # Calculate the maximum width for placing stars
    max_width = df.iloc[idx, 0] + df.iloc[idx, 1]

    if dataset_name.lower() == "t2d" or dataset_name.lower() == "fatty liver" or dataset_name.lower() == "ibdmdb":  # For Obesity, Healthy, CF, add 1 star
        ax.text(max_width + 0.005, idx, "*", fontsize=SIZE, va='center', color='black')
    elif dataset_name.lower() == "infant allergy" or dataset_name.lower() == "nec" or dataset_name.lower() =="preterm infants" or dataset_name.lower() =="cf":
        ax.text(max_width + 0.005, idx, "-", fontsize=SIZE, va='center', color='black')
    elif dataset_name.lower() == "early dynamics 2" or dataset_name.lower() == "healthy":
        ax.text(max_width + 0.005, idx, "H", fontsize=SIZE, va='center', color='black')
    else:  # For all other datasets, add 3 stars
        ax.text(max_width + 0.005, idx, "n.s", fontsize=SIZE, va='center', color='black')  # Adjusted closer


plt.legend().set_visible(False)
# Save and show the plot
plt.savefig("nadav_check/figures/updated/Fig_1/fractions_gaussians_with_stars.png", bbox_inches='tight')
plt.show()