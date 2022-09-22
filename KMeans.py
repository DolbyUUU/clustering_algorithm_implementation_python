import numpy as np

class KMeans:

    def __init__(self, k, tol=1e-4, max_iter=100, n_init=100):
        '''
        Define a k-means algorithm with known number of clusters.
        '''
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        
    def fit(self, X):
        '''
        Initialize randomly, train model with input data, and select trial with smallest cost.
        '''
        n, dim = X.shape
        X_values = np.array(X)
        X_labels = np.zeros(n)
        costs = np.zeros(self.n_init)
        X_labels_lst, centers_lst = [], []
        for i in range(self.n_init):
            centers = X[np.random.choice(n, size=self.k, replace=False)]
            for j in range(self.max_iter):            
                old_centers = np.copy(centers)
                distances = self.compute_distances(X_values, centers, n)
                X_labels = self.label_samples(distances)
                centers = self.compute_centers(X_values, X_labels, dim)
                clusters_stable = np.abs(centers - old_centers) < self.tol
                if np.all(clusters_stable) != False:
                    break
            X_labels_lst.append(X_labels)
            centers_lst.append(centers)
            costs[i] = self.compute_cost(X_values, X_labels, centers)
        best_trial_index = costs.argmin()
        best_cost = costs[best_trial_index]
        best_X_labels = X_labels_lst[best_trial_index]
        best_centers = centers_lst[best_trial_index]
        print(f"Best result in {self.n_init} random trials:")
        print(f"\nCluster centers: {best_centers}")
        return best_X_labels, best_centers

    def compute_distances(self, X, centers, n):
        '''
        Calculate distances of samples to their cluster center.
        '''
        distances = np.zeros((n, self.k))
        for cluster_center_index, cluster_center in enumerate(centers):
            distances[:, cluster_center_index] = np.linalg.norm(X - cluster_center, axis = 1)
        return distances
    
    def label_samples(self, distances):
        '''
        Assign and label samples to their closest cluster center.
        '''
        return distances.argmin(axis = 1)

    def compute_centers(self, X, labels, dim):
        '''
        Estimate and update cluster centers as the means of their members.
        '''
        centers = np.zeros((self.k, dim))
        for center_index, center in enumerate(centers):
            members = X[labels == center_index]
            if len(members):
                centers[center_index, :] = members.mean(axis = 0)
        return centers
    
    def compute_cost(self, X, labels, centers):
        '''
        Calculate sum of distances of samples to their cluster center. 
        Squared distance can be used instead of linear distance.
        '''
        cost = 0
        for center_index, center in enumerate(centers):
            members = X[labels == center_index]
            cost += np.linalg.norm(members - center, axis = 1).sum()
        return cost
