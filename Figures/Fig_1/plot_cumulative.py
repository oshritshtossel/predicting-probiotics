import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rc('font', family='Arial')
SIZE = 20
LOG = False
NORM =True
COLOR = False
TAX = 7
# List of dataset names
if TAX == 8:
    LIST_NAMES = ['PRJEB6456', "PRJNA1130109", "PRJEB39500","PRJNA345144","PRJNA290729","PRJNA273761","PRJNA301903","PRJNA395569","PRJNA510445","PRJNA806984"]
else:
    LIST_NAMES = ["Allergy", 'GDM_OLD', 'PRJEB14529', "PRJNA730851", 'PRJEB6456', "PRJNA1130109", "PRJEB39500", "PRJNA345144",
     "PRJNA290729", "PRJNA273761", "PRJNA301903", "PRJNA395569", "PRJNA510445", "PRJNA806984"]
NUM = 10
# Initialize the plot
plt.figure(figsize=(5, 5))



if COLOR:
    colors = sns.color_palette("husl", NUM)

# Specify a different color for the 8th dataset
    highlight_color = 'grey'

# Define 10 different line styles
line_styles = [
    '-',                  # Solid line
    '--',                 # Dashed line
    ':',                  # Dotted line
    '-.',                 # Dash-dot line
    (0, (5, 1)),          # Custom: long dash with small gap
    (0, (3, 1, 1, 1)),    # Custom: dash-dot-dash pattern
    (0, (5, 2, 1, 2)),    # Custom: long dash, short gap, dot, short gap
    (0, (1, 1)),          # Custom: densely dotted
    (0, (3, 5, 1, 5)),    # Custom: dash-dot with larger gaps
    (0, (4, 4, 1, 4, 1, 4))  # Custom: dash-dash-dot pattern
]
sim = pd.read_csv("nadav_check/22_7/cumulative/simulation_14_8.csv",index_col=0)
sim.index = [round(np.exp(i),1) for i in sim.index]
sim_100 = sim.loc[100]['y']
sim = sim.iloc[:1000]
sim_normalized = sim["y"]/sim_100

for i, NAME in enumerate(LIST_NAMES):
    df = pd.read_csv(f"nadav_check/22_7/cumulative/{NAME}.csv", index_col=0)
    if NAME == "GDM_OLD":
        nn = "GDM"
    elif NAME == "PRJNA345144":
        nn = "Eczema"
    elif NAME == "PRJEB6456":
        nn = "Early dynamics 1"
    elif NAME == "PRJEB39500":
        nn = "T2D"
    elif NAME == "PRJNA1130109":
        nn = "Obesity"
    elif NAME == "PRJNA730851":
        nn = "Infant allergy"
    elif NAME == "PRJEB14529":
        nn = "Early dynamics 2"
    elif NAME == "PRJNA290729":
        nn = "Fatty liver"
    elif NAME == "PRJNA273761":
        nn = "NEC"
    elif NAME == "PRJNA301903":
        nn = "Preterm infants"
    elif NAME == "PRJNA395569":
        nn = "IBDMDB"
    elif NAME == "PRJNA510445":
        nn = "CF"
    elif NAME == "PRJNA806984":
        nn = "Healthy"
    x = df["Number of samples"]
    y = df["Non zero species"]
    if COLOR:
        if i < NUM:
            color = colors[i]
        else:
            color = highlight_color
    if NORM:
        if NAME == "PRJNA290729":
            y100 = (df["Non zero species"][df["Number of samples"] == 92]).values[0]
        elif NAME == "PRJNA273761":
            y100 = (df["Non zero species"][df["Number of samples"] == 60]).values[0]
        else:
            y100 = (df["Non zero species"][df["Number of samples"] == 100]).values[0]

        y_normalized = y/y100

        if COLOR:
            plt.plot(x, y_normalized, color=color, label=NAME)
        else:
            line_style = line_styles[i % len(line_styles)]
            plt.plot(x, y_normalized, color='black')
    else:
        plt.plot(x, y, color=color, label=NAME)

# Customize the plot
sim_handle = plt.scatter(sim.index,sim_normalized,color="red")
plt.xlabel("Number of samples",fontsize=SIZE)
if TAX ==7:
    plt.ylabel("Normalized non zero species",fontsize=SIZE)
else:
    plt.ylabel("Normalized non zero strains", fontsize=SIZE)
plt.xscale('log')
#plt.title("Cumulative Non-Zero Species Per Sample",fontsize=SIZE)
plt.xticks(fontsize=SIZE)
plt.yticks(fontsize=SIZE)

handles, labels = plt.gca().get_legend_handles_labels()

# Create two separate legends
legend_wgs = [handle for handle, label in zip(handles, labels) if label != "Simulation"]
legend_sim = [sim_handle]
plt.legend(legend_wgs, [label for label in labels if label != "Simulation"], fontsize=12, loc='upper left')
plt.legend(legend_sim, ["Simulation"], fontsize=15, loc='lower right')
plt.tight_layout()

plt.savefig(f"nadav_check/27_11/cumulative/cumulative_all_in_1_normalized_line_styles_{TAX}.png")
plt.show()
