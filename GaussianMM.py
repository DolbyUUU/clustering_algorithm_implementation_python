import numpy as np
from scipy.stats import multivariate_normal

class GaussianMM:
    
    def __init__(self, k, max_iter=100, n_init=100, regul=1e-6):
        '''
        Define a Gaussian mixture model with known number of clusters and dimensions.
        '''
        self.k = k
        self.max_iter = max_iter
        self.n_init = n_init
        self.regul = regul

    def init_em(self, X):
        '''
        Initialization for EM algorithm.
        '''
        # Random initialization of Gaussian mean mu with uniform[-10, 10].
        self.data = X
        self.num_points = X.shape[0]
        self.dim = X.shape[1]
        init_mu = np.random.rand(self.k, self.dim) * 20 - 10
        self.mu = init_mu
        init_sigma = np.zeros((self.k, self.dim, self.dim))
        for i in range(self.k):
            init_sigma[i] = np.identity(self.dim)
        self.sigma = init_sigma
        init_pi = np.ones(self.k) / self.k
        self.pi = init_pi
        self.z = np.zeros((self.num_points, self.k))

    def e_step(self):
        '''
        E-step of EM algorithm.
        '''
        for j in range(self.k):
            self.z[:, j] = self.pi[j] * multivariate_normal.pdf(self.data, mean=self.mu[j], cov=self.sigma[j])
            # self.z[:, j] = np.dot(self.pi[j], multivariate_normal.pdf(self.data, mean=self.mu[j], cov=self.sigma[j]))
        self.z /= self.z.sum(axis=1, keepdims=True)

    def m_step(self):
        '''
        M-step of EM algorithm.
        '''
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for j in range(self.k):
            minus = np.expand_dims(self.data, axis=1) - self.mu[j]
            squared = np.matmul(minus.transpose([0, 2, 1]), minus)
            self.sigma[j] = np.matmul(squared.transpose(1, 2, 0), self.z[:, j])
            self.sigma[j] /= sum_z[j]
            # Regularization of the covariance matrices sigma_k.
            self.sigma[j] += np.identity(self.dim) * self.regul

    def log_likelihood(self, X):
        '''
        Compute the log-likelihood of X under current parameters.
        '''
        ll = []
        for d in X:
            tot = 0
            for j in range(self.k):
                tot += self.pi[j] * multivariate_normal.pdf(d, mean=self.mu[j], cov=self.sigma[j])
            ll.append(np.log(tot))
        data_ll = np.sum(ll)
        return data_ll

    def predict(self, X):
        '''
        Predict the cluster assignments of input data.
        '''
        probas = []
        cluster_labels = []
        cluster_centers = []
        cluster_weights = []
        for i in range(len(X)):
            probas.append([multivariate_normal.pdf(X[i], mean=self.mu[j], cov=self.sigma[j]) for j in range(self.k)])
        for proba in probas:
            cluster_labels.append(proba.index(max(proba)))
        for j in range(self.k):
            cluster_centers.append(self.mu[j])
        for j in range(self.k):
            cluster_weights.append(self.pi[j])
        return np.array(cluster_labels), np.array(cluster_centers), np.array(cluster_weights)

    def fit(self, X):
        '''
        Initialize randomly, train the model with input data, and select trial with largest data log-likelihood.
        '''
        i_iter = 0
        data_ll_lst, cluster_labels_lst, cluster_centers_lst, cluster_weights_lst = [], [], [], []
        while self.n_init >= i_iter:
            i_iter += 1
            self.init_em(X)
            for i in range(self.max_iter):
                self.e_step()
                self.m_step()
            cluster_labels, cluster_centers, cluster_weights = self.predict(X)
            data_ll = self.log_likelihood(X)
            data_ll_lst.append(data_ll)
            cluster_labels_lst.append(cluster_labels)
            cluster_centers_lst.append(cluster_centers)
            cluster_weights_lst.append(cluster_weights)
        for i in range(len(data_ll_lst)):
            if data_ll_lst[i] == max(data_ll_lst):
                best_data_ll = data_ll_lst[i]
                best_cluster_labels = cluster_labels_lst[i]
                best_cluster_centers = cluster_centers_lst[i]
                best_cluster_weights = cluster_weights_lst[i]
        print(f"Best result in {self.n_init} random trials:")
        print(f"\nCluster centers: {best_cluster_centers}")
        print(f"\nCluster weights: {best_cluster_weights}")
        return best_cluster_labels, best_cluster_centers, best_cluster_weights
