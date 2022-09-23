import numpy as np
import random

def k_means(samples, n_clusters, n_init=200, max_iter=300, tol=1e-4):
    """
    Implementation of K-Means clustering algorithm.
    """

    i_iter_lst = []
    inertia_lst = []
    labels_lst = []
    cluster_centers_lst = []

    x = np.asarray(samples)
    n = len(x)
    k = n_clusters
    d = len(x[0])

    # Trial with different cluster center seeds.
    i_init = 0
    while i_init < n_init:

        # Initialize cluster assignments and cluster centers.
        z = np.zeros((n, k))
        y = np.zeros(n)
        mu = np.asarray(random.choices(x, k=k)) # Random cluster center seeds.
        mu_old = np.zeros((k, d))

        # Iterate until convergence.
        i_iter = 0
        while i_iter < max_iter:
            if np.linalg.norm(mu_old - mu) > tol:
                mu_old = mu[:]

                # Assign clusters.
                for i in range(n):
                    min_dist = np.linalg.norm(x[i] - mu[0])
                    for j in range(k):
                        dist = np.linalg.norm(x[i] - mu[j])
                        if dist < min_dist:
                            min_dist = dist
                    for j in range(k):
                        dist = np.linalg.norm(x[i] - mu[j])
                        if dist == min_dist:
                            z[i, j] = 1
                    for j in range(k):
                        if z[i, j] == 1:
                            y[i] = j

                # Estimate cluster centers.
                sum_zx = np.zeros((k, d))
                for i in range(n):
                    zx = np.outer(z[i,], x[i])
                    sum_zx += zx
                sum_z = np.zeros(k)
                for i in range(n):
                    sum_z += z[i,]
                mu = sum_zx / sum_z[:, None]

            else:
                break
            i_iter += 1

        # Calculate inertia.
        sq_dist_center = np.zeros(n)
        for i in range(n):
            y_i = int(y[i])
            sq_dist_center[i] = np.linalg.norm(x[i] - mu[y_i])**2
        inertia = np.sum(sq_dist_center)

        i_init += 1
        i_iter_lst.append(i_iter)
        inertia_lst.append(inertia)
        labels_lst.append(y)
        cluster_centers_lst.append(mu)

    # Select best trial with lowest inertia.
    for i in range(n_init):
        if inertia_lst[i] == min(inertia_lst):
            best_i_iter = i_iter_lst[i]
            best_labels = labels_lst[i]
            best_cluster_centers = cluster_centers_lst[i]

    # Output clustering results.
    print(f"Iteration number: {best_i_iter}")
    print(f"Sample labels: {best_labels}")
    print(f"Cluster centers: {best_cluster_centers}")
    return best_labels, best_cluster_centers
