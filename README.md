# Shifting from Abundance and Centrality - Fluctuating Microbes are Key Drivers of Long Lasting Microbiome Manipulation
This code is attached to the paper "Shifting from Abundance and Centrality - Fluctuating Microbes are Key Drivers of Long Lasting Microbiome Manipulation". 
For computing scores for fluctuating microbes, you can use our website, [micrOS](https://micros.math.biu.ac.il/).
To search for the scores of a specific taxon in our database, visit [micrOS](https://micros.math.biu.ac.il/) and navigate to the [Predicted Next Gen Probiotic Taxa](https://micros.math.biu.ac.il/Pngt).
The full probiotics table is also available in the **example** directory as **"probiotics.csv"**.

## Python Dependencies (for computing microbial features)
```python
pandas
numpy
scipy
re
scikit-learn
tqdm
networkx
```


## Computing microbial features
By using the code in the **features_calculation** directory.
- **calc_abundance_and_fluctuation_measures.py -** For computing the features of *log average abundance*, *log median abundance*, *standard deviation over time (S.D. time)*, and *standard deviation over population (S.D. people)* and *Fiedler vector*.
- **calc_centrality_measures.py -** For building the predicted interaction network based on the predicting the differences of the log abundances of a certain taxon based on the log abundances in the previous time step. Then 4 centrality measures are computed to the interaction network: *in-degree*, *out-degree*, *betweenness centrality*, and *closeness centrality*.

### Example use:
There is an example dataset in the example directory ("tax8_log_sub PCA.csv" & "tag.csv"). 
- By running the **calc_abundance_and_fluctuation_measures.py** script, one can produce a pandas dataframe containing the  *log average abundance*, *log median abundance*, *standard deviation over time (S.D. time)*, and *standard deviation over population (S.D. people)* and *Fiedler vector* features.
- By running the **calc_centrality_measures.py** script, one can produce a pandas dataframe containing the  *in-degree*, *out-degree*, *betweenness centrality*, and *closeness centrality* features.

**For others datasets, follow the format of the uploaded example data and update the file pathes as explained in the code documentation**

## Figures
#### Fig_1
- **plot_consecutive_distances_distribution.py -** Produces histogram of Euclidean distances between consecutive time points of real studies and fits the data into 2 Gaussians and colors each Gaussian in a different color. (**Fig. 1B**).
- **plot_fraction_of_samples_per_gaussian.py -** Plots the fraction of samples per gaussian distribution across different studies. All cohort distributions are significantly multimodal, as confirmed by the F-test. n.s. indicates that the host phenotype does not explain the multimodality, "\*" represents a significant association between the phenotype and multimodality, and "-" indicates studies where phenotype data is unavailable. "H" signifies cohorts where all samples are from healthy individuals (**Fig. 1C**).
- **plot_consecutive_distances_simulation.py -** Produces histogram of Euclidean distances between consecutive time points of GLVM simulation and fits the data into 2 Gaussians and colors each Gaussian in a different color. (**Fig. 1D**).
- **plot_fraction_non_zero_strains.py -** Produces log-log histogram showing the fraction of non-zero strains in populations from real-world cohorts (black) and simulated datasets (red). The strain distribution follows a log-normal pattern (blue trend line), demonstrating power-law-like behavior in strain abundance across populations (**Fig. 1E**).
- **plot_cumulative.py -** Plots the normalized count of non-zero strains as a function of sample size in real-world cohorts (black) and simulated populations (red) (**Fig. 1F**).

#### Fig_2
- **plot_success_colonization_roc_curve.py -** Plots the ROC AUCs for predicting RSM taxa (colonization success measured by high $P(Change)$) using the LR model across 10 WGS cohorts (black) and in a simulation (red) (**Fig. 2A**).
- **plot_swarm_plot_fluctuating_vs_all.py -** Plots the comparison of prediction performance using only fluctuating features (blue) versus all features (black). Each point represents the average AUC from the test set of a different study. No significant differences were observed between models using all features and those using fluctuating measures alone. Light grey represents predictions of RSM defined by colonization success, pink represents RSM defined by phenotype change (RPT), and beige represents RSM defined by phenotype change (influential FMT) (**Fig. 2B**).
- **plot_swarm_plot_coefficients.py -** Plots the coefficents or FI scores of predicting RSM by different models and varied definitons. Each RSM definition is shown in different colors (light grey for colonization success, pink for RPT, and beige for influential FMT taxa). These features include predictions of taxa with high likelihood to colonize (light purple C-E), RPT (pink F-H), or influential FMT (beige I-K) across different models, including LR (logistic regression) models (C,F,I), RF models (D,G,J), and XGBOOST models (E,H,K). Feature contributions are measured by coefficients in LR models and feature importance (FI) in RF and XGBOOST models. Blue features consistently show significantly higher positive contributions across all tasks. Stars indicate p-values from paired t-tests comparing blue features with orange and green features such that \*-p<0.05, \*\*-p<0.01, \*\*\*-p<0.001 (**Fig. 2C-K**).

#### Fig_3
- **plot_time_scatter.py -** Plots scatter plot between each microbial features (abundance, fluctuation , centrality) and persistence time of different taxa (**Fig. 3A**).
- **plot_consistency_checks.py -** (1) Plots significant correlations (after FDR correction) of shared taxon feature distributions across 10 WGS studies. The distribution is skewed to the right, with an average correlation of 0.833±0.372, indicating a strong positive relationship. (2) Plots consistency of microbial features across WGS studies measured by the average SCC for each measure across the common taxa present in the cohorts. Fluctuation is significantly the most consistent feature (blue). Black error bars represent the standard deviation of the SCCs for each measure (**Fig. 3B-C**).
- **plot_predicted_probiotic_bar_plot.py -** Plots predicted next-generation probiotic taxa based on S.D. across individuals, consistent in at least three WGS studies and within the top 5th percentile. Bar plots represent the average S.D. of taxa (bottom x-axis), while dots represent the consistency of fluctuation as a fraction of studies in which the taxa rank in the top 5 percentile (top x-axis). Pink bars correspond to taxa already known as probiotics (RPT), purple bars represent taxa identified as reported next-generation probiotic candidates, and blue bars indicate newly identified taxa. Note: for taxa reported at the species level, all strains within that species share the same color (**Fig. 3E**).


