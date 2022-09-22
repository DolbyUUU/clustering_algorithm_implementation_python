import numpy as np

class MeanShift:

    def __init__(self, bandwidth, max_iter=300, tol=1e-4):
        '''
        Define a mean-shift algorithm with known kernel bandwidth.
        '''
        self.bandwidth = bandwidth
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        x = np.array(X)
        n, dim = X.shape
        cluster_centers_lst = []
        cluster_labels = np.zeros(n)
        cluster_centers_old = np.zeros(dim)
        # Initialize each sample.
        for t in range(n):
            cluster_centers = x[t]
            # Iterate until convergence.
            i_iter = 0
            while i_iter < self.max_iter:
                if np.linalg.norm(cluster_centers_old - cluster_centers) > self.tol:
                    cluster_centers_old = cluster_centers[:]
                    sum_xi_weight = np.zeros(dim)
                    sum_weight = np.zeros(dim)
                    for i in range(n):
                        r = np.linalg.norm((cluster_centers - x[i]) / self.bandwidth) ** 2
                        weight = 0.5 * np.exp(-0.5 * r)
                        xi_weight = x[i] * weight
                        sum_xi_weight += xi_weight
                        sum_weight += weight
                    cluster_centers = sum_xi_weight / sum_weight
                else:
                    break
                i_iter += 1
            cluster_centers_lst.append(np.around(cluster_centers, decimals=2))
        cluster_centers = np.unique(cluster_centers_lst, axis=0)
        # Generate sample labels.
        for i in range(n):
            for j in range(len(cluster_centers)):
                if np.array_equal(cluster_centers_lst[i], cluster_centers[j]):
                    cluster_labels[i] = j
        cluster_labels = cluster_labels.astype(int)
        print(f"\nCluster centers: {cluster_centers}")
        return cluster_labels, cluster_centers
