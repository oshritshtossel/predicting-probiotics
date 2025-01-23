import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import seaborn as sns
import matplotlib as mpl
def test_gmm_fit_and_separate_distances(list_dists, max_components=3):
    # Convert list_dists to a numpy array and reshape it
    list_dists_reshaped = np.array(list_dists).reshape(-1, 1)

    try:
        # Fit a mixture of Gaussians (max_components)
        gmmN = GaussianMixture(n_components=max_components, covariance_type='full', random_state=42)
        gmmN.fit(list_dists_reshaped)

        # Predict which Gaussian each distance belongs to
        labels = gmmN.predict(list_dists_reshaped)

        # Separate the distances into lists based on the Gaussian labels
        distances_by_gaussian = [list_dists_reshaped[labels == i].flatten().tolist() for i in range(max_components)]

        # Extract the means and standard deviations of each component
        means = gmmN.means_.flatten()
        std_devs = np.sqrt(gmmN.covariances_).flatten()

        # Plot the histogram of distances
        plt.figure(figsize=(10, 6))
        plt.hist(list_dists, bins=30, color='skyblue', edgecolor='black', alpha=0.6, density=True, label='Histogram')

        # Generate KDE curves for each Gaussian component
        x = np.linspace(min(list_dists), max(list_dists), 1000)
        for i, (mean, std_dev) in enumerate(zip(means, std_devs)):
            pdf = norm.pdf(x, mean, std_dev)
            plt.plot(x, pdf, label=f'Gaussian {i + 1}: μ={mean:.2f}, σ={std_dev:.2f}', lw=2)

        plt.title("Histogram and KDE of Euclidean Distances Between Consecutive Time Points")
        plt.xlabel("Euclidean Distance")
        plt.ylabel("Density")
        plt.yscale("log")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig("akiva/old_sim/dist_hist_kde.png")
        plt.show()

        # Print and return the distances separated by Gaussian
        for i, distances in enumerate(distances_by_gaussian):
            print(f"\nGaussian {i + 1}:")
            print(f"Mean = {means[i]:.2f}, Std Dev = {std_devs[i]:.2f}")
            print(f"Number of Points = {len(distances)}")
            print(f"Distances: {distances[:10]}... (showing first 10 values)")

        return distances_by_gaussian

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == '__main__':
    mpl.rc('font', family='Arial')
    FOLDER = "combined_sim_3"#"combined_sim/combined_sim"#
    all_distances = []
    for i in range(1, 11):
        data = loadmat(f'akiva/{FOLDER}/data{i}.mat')['mm3']

        # Iterate over each person
        for person_idx in range(data.shape[2]):
            person_data = data[:, :, person_idx]  # Extract data for a specific person (shape: 100 x 20)
            person_distances = []

            # Iterate over time points, calculating distances between consecutive points
            for time_idx in range(1, person_data.shape[1]):
                # Get the data for consecutive time points
                point_t1 = person_data[:, time_idx - 1]
                point_t2 = person_data[:, time_idx]

                # Calculate Euclidean distance and add it to the list
                dist = euclidean(point_t1, point_t2)
                person_distances.append(dist)

            # Append the distances for this person to the main list
            all_distances.extend(person_distances)  # Flatten the distances as we iterate

    # Convert list of lists to a 1D array for easier handling
    flattened_distances = np.array(all_distances)

    # Test GMM fit and separate distances by Gaussian
    distances_by_gaussian = test_gmm_fit_and_separate_distances(flattened_distances, max_components=2)
    plt.figure(figsize=(5, 5))

    # Plot first Gaussian in blue
    sns.kdeplot(distances_by_gaussian[1], color='red', label='G1', fill=True)
    sns.kdeplot(distances_by_gaussian[0], color='lightcoral', label='G2', fill=True)

    plt.title("Simulation",size=20,fontweight="bold")
    plt.xlabel('Distance', fontsize=20)
    plt.ylabel('Density', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(f"akiva/{FOLDER}/dist_hist_components.png")
    plt.show()
