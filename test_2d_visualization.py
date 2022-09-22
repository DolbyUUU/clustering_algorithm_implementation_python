from GaussianMM import GaussianMM
from KMeans import KMeans
from MeanShift import MeanShift

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture


# Import datasets.
filename = 'data_2d_test/cluster_data_text/cluster_data_dataA_X.txt'
data_a_x = np.genfromtxt(filename, delimiter='\t')
data_a_x = np.transpose(data_a_x)
data_a_x = data_a_x[~np.isnan(data_a_x).any(axis=1)]
filename = 'data_2d_test/cluster_data_text/cluster_data_dataB_X.txt'
data_b_x = np.genfromtxt(filename, delimiter='\t')
data_b_x = np.transpose(data_b_x)
data_b_x = data_b_x[~np.isnan(data_b_x).any(axis=1)]
filename = 'data_2d_test/cluster_data_text/cluster_data_dataC_X.txt'
data_c_x = np.genfromtxt(filename, delimiter='\t')
data_c_x = np.transpose(data_c_x)
data_c_x = data_c_x[~np.isnan(data_c_x).any(axis=1)]


def plot_clusters(samples, labels, title):
    """
    Plot the results of clustering analysis.
    """
    filtered_label0 = samples[labels == 0]
    filtered_label1 = samples[labels == 1]
    filtered_label2 = samples[labels == 2]
    filtered_label3 = samples[labels == 3]
    filtered_labels = samples[labels >= 4]
    plt.scatter(filtered_label0[:, 0] , filtered_label0[:, 1] , color = 'red')
    plt.scatter(filtered_label1[:, 0] , filtered_label1[:, 1] , color = 'green')
    plt.scatter(filtered_label2[:, 0] , filtered_label2[:, 1] , color = 'blue')
    plt.scatter(filtered_label3[:, 0] , filtered_label3[:, 1] , color = 'yellow')
    plt.scatter(filtered_labels[:, 0] , filtered_labels[:, 1] , color = 'grey')
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title(title)
    plt.savefig(title.replace(".", "_").replace(" ", "_").replace(",", "")
        .replace("=", "").replace("(", "").replace(")", "").lower())
    plt.close()



# 2d dataset a.

meanshift = MeanShift(bandwidth=2.1, max_iter=200, tol=1e-4)
cluster_labels, cluster_centers = meanshift.fit(data_a_x)
plot_clusters(data_a_x, cluster_labels, "Data pattern blobs, mean-shift (my implementation)")

kmeans = KMeans(k=4, tol=1e-4, max_iter=200, n_init=500)
cluster_labels, cluster_centers = kmeans.fit(data_a_x)
plot_clusters(data_a_x, cluster_labels, "Data pattern blobs, k-means (my implementation)")

gmm = GaussianMM(k=4, max_iter=200, n_init=500, regul=1e-6)
cluster_labels, cluster_centers, cluster_weights = gmm.fit(data_a_x)
plot_clusters(data_a_x, cluster_labels, "Data pattern blobs, EM-GMM (my implementation)")

# gmm = mixture.GaussianMixture(
#     n_components=4, covariance_type="full", random_state=0, n_init=300
#     ).fit(data_a_x)
# plot_clusters(data_a_x, gmm.predict(data_a_x), "Data pattern blobs, EM-GMM (scikit-learn)")

# 2d dataset b.

meanshift = MeanShift(bandwidth=1.65, max_iter=200, tol=1e-4)
cluster_labels, cluster_centers = meanshift.fit(data_b_x)
plot_clusters(data_b_x, cluster_labels, "Data pattern sticks, mean-shift (my implementation)")

kmeans = KMeans(k=4, tol=1e-4, max_iter=200, n_init=500)
cluster_labels, cluster_centers = kmeans.fit(data_b_x)
plot_clusters(data_b_x, cluster_labels, "Data pattern sticks, k-means (my implementation)")

gmm = GaussianMM(k=4, max_iter=200, n_init=500, regul=1e-6)
cluster_labels, cluster_centers, cluster_weights = gmm.fit(data_b_x)
plot_clusters(data_b_x, cluster_labels, "Data pattern sticks, EM-GMM (my implementation)")

# gmm = mixture.GaussianMixture(
#     n_components=4, covariance_type="full", random_state=0, n_init=300
#     ).fit(data_b_x)
# plot_clusters(data_b_x, gmm.predict(data_b_x), "Data pattern sticks, EM-GMM (scikit-learn)")

# 2d dataset c.

meanshift = MeanShift(bandwidth=2.0, max_iter=200, tol=1e-4)
cluster_labels, cluster_centers = meanshift.fit(data_c_x)
plot_clusters(data_c_x, cluster_labels, "Data pattern moons and stars, mean-shift (my implementation)")

kmeans = KMeans(k=4, tol=1e-4, max_iter=200, n_init=500)
cluster_labels, cluster_centers = kmeans.fit(data_c_x)
plot_clusters(data_c_x, cluster_labels, "Data pattern moons and stars, k-means (my implementation)")

gmm = GaussianMM(k=4, max_iter=200, n_init=500, regul=1e-6)
cluster_labels, cluster_centers, cluster_weights = gmm.fit(data_c_x)
plot_clusters(data_c_x, cluster_labels, "Data pattern moons and stars, EM-GMM (my implementation)")

# gmm = mixture.GaussianMixture(
#     n_components=4, covariance_type="full", random_state=0, n_init=300
#     ).fit(data_c_x)
# plot_clusters(data_c_x, gmm.predict(data_c_x), "Data pattern moons and stars, EM-GMM (scikit-learn)")
