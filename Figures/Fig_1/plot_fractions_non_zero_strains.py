import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.stats import linregress
mpl.rc('font', family='Arial')
def plot_hist(data,bins):
    hist, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Find the first non-zero bin
    first_non_zero_idx = np.argmax(hist > 0)

    # Adjust the bin centers and histogram counts to start from the first non-zero bin
    bin_centers = bin_centers[first_non_zero_idx:]
    hist = hist[first_non_zero_idx:]
    hist = hist
    return hist,bin_centers

SIZE = 20
LOG = True
NORM = True
COLORS = False
EFFECTING = False


TAX = 7
if TAX == 8:
    LIST_NAMES = ["PRJNA1130109", "PRJEB39500", 'PRJEB6456', "PRJNA345144", "PRJNA290729", "PRJNA395569", "PRJNA301903",
                  "PRJNA510445", "PRJNA806984"]
else:
    LIST_NAMES = ["Allergy", 'GDM_OLD', 'PRJEB14529', "PRJNA730851", 'PRJEB6456', "PRJNA1130109", "PRJEB39500",
                  "PRJNA345144",
                  "PRJNA290729",  "PRJNA301903", "PRJNA395569", "PRJNA510445", "PRJNA806984"]#"PRJNA273761",
NUM = 8
# Initialize the plot
plt.figure(figsize=(5, 5))
freq = pd.read_csv("nadav_check/22_7/species_frac/freqs.csv",index_col=0)
if COLORS:
    # Generate colors for each dataset
    colors = sns.color_palette("husl", NUM)
    highlight_color = 'grey'
else:
    # Define 9 different line styles
    line_styles = ['-', '--', ':', '-.', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)), (0, (1, 1)),
                   (0, (3, 1, 1, 1, 1, 1))]

# Collect all histogram data for regression
all_hist = []
all_bin_centers = []

# Create 10 bins logarithmically spaced between the minimum and maximum of your data
min_value = 1e-3  # Example minimum value, adjust as necessary
max_value = 1e0   # Example maximum value, adjust as necessary
bins = np.logspace(np.log10(min_value), np.log10(max_value), 5)
freq_vals = freq.values.flatten()
sim_hist,sim_bins = plot_hist(freq_vals,bins)

# Plot each dataset's histogram as a line
for i, NAME in enumerate(LIST_NAMES):
    if TAX == 8 and NAME != "PRJNA395569" and NAME != "PRJNA510445" and NAME != "PRJNA806984":
        df = pd.read_csv(f"nadav_check/22_7/species_frac/{NAME}_8.csv", index_col=0)
    elif NAME == "PRJNA510445" or NAME == "PRJNA806984":
        df = pd.read_csv(f"nadav_check/22_7/species_frac/{NAME}_7.csv", index_col=0)
    else:
        df = pd.read_csv(f"nadav_check/22_7/species_frac/{NAME}.csv", index_col=0)
    if EFFECTING and NAME != "GGMP":
        effecting = pd.read_csv(f"nadav_check/22_7/metrics/staying/{NAME}.csv",index_col=0)
        c=0
    # Assuming you want to plot the histogram of all values in the dataframe
    data = df.values.flatten()
    if COLORS:
        if i < NUM:
            color = colors[i]
        else:
            color = highlight_color

    # Plot histogram as a line
    #plt.hist(data,bins)
    hist, bins = np.histogram(data, bins=bins, density=True)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Find the first non-zero bin
    first_non_zero_idx = np.argmax(hist > 0)

    # Adjust the bin centers and histogram counts to start from the first non-zero bin
    bin_centers = bin_centers[first_non_zero_idx:]
    hist = hist[first_non_zero_idx:]

    # Append to all_hist and all_bin_centers
    all_hist.extend(hist)
    all_bin_centers.extend(bin_centers)

    if COLORS:
        plt.plot(bin_centers, hist, color=color, label=NAME)
    else:
        line_style = line_styles[i % len(line_styles)]
        plt.plot(bin_centers, hist, color='black', label=NAME) # linestyle=line_style,
        if EFFECTING and NAME != "GGMP":
            effecting_frac =effecting['frac'][effecting["staying"] == 1]
            plt.scatter(effecting_frac, [10**-2] * len(effecting_frac), color="blue", alpha=0.5)


# Customize the plot
plt.plot(sim_bins,sim_hist,label= "Simulation",color="red", lw=5)
if TAX == 7:
    plt.xlabel("Fraction of non-zero species", fontsize=SIZE)
else:
    plt.xlabel("Fraction of non-zero strains", fontsize=SIZE)


# Remove zeros from all_hist and corresponding values from all_bin_centers
filtered_hist_bin_centers = [(h, bc) for h, bc in zip(all_hist, all_bin_centers) if h != 0]
all_hist, all_bin_centers = zip(*filtered_hist_bin_centers)

# Perform regression on log-transformed data
log_bin_centers = np.log10(all_bin_centers)
log_hist = np.log10(all_hist)
slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_hist)
log_bin_centers = np.log10(all_bin_centers)
log_hist = np.log10(all_hist)
slope, intercept, r_value, p_value, std_err = linregress(log_bin_centers, log_hist)

# Plot the regression line
regression_line = 10**(intercept + slope * log_bin_centers)
plt.plot(10**log_bin_centers, regression_line, color='blue', linestyle='--', label=f'Regression Line (slope={slope:.2f})')

if LOG:
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks([10 ** i for i in range(-2, 0)], [f"$10^{{{i}}}$" for i in range(-2, 0)],fontsize=SIZE)
    plt.yticks(fontsize=SIZE)
plt.ylabel("Frequency", fontsize=SIZE)
handles, labels = plt.gca().get_legend_handles_labels()
desired_labels = ["Simulation", f"Regression Line (slope={slope:.2f})"]
filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l in desired_labels]
handles, labels = zip(*filtered_handles_labels)
plt.legend(handles, labels,fontsize=12)

plt.tight_layout()
if LOG:
    #pass
    if TAX ==8:
        plt.savefig(f"nadav_check/27_11/species_frac/all_his_semi_log_{TAX}.png")
    else:
        plt.savefig(f"nadav_check/27_11/species_frac/all_his_semi_log_{TAX}.png")
else:
    if TAX == 8:
        plt.savefig(f"nadav_check/27_11/species_frac/all_hist_{TAX}.png")
    else:
        plt.savefig(f"nadav_check/27_11/species_frac/all_hist_{TAX}.png")
plt.show()
