import numpy as np

class MeanShift:

    def __init__(self, bandwidth, stop_thresh=1e-4, cluster_thresh=1e-2):
        '''
        Define a mean-shift algorithm with known kernel bandwidth.
        '''
        self.bandwidth = bandwidth
        self.stop_thresh = stop_thresh
        self.cluster_thresh = cluster_thresh

    def fit(self, X):
        shift_points = np.array(X)
        shifting = [True] * X.shape[0]
        while True:
            max_dist = 0
            for i in range(len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self.shift_point(shift_points[i], X)
                dist = self.distance(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > self.stop_thresh
            if (max_dist < self.stop_thresh):
                break
        cluster_labels, cluster_centers = self.cluster_points(shift_points.tolist())
        cluster_labels = np.int64(cluster_labels)
        return cluster_labels, cluster_centers

    def shift_point(self, point, X):
        shift_x = 0
        shift_y = 0
        scale = 0
        for p in X:
            dist = self.distance(point, p)
            weight = self.gaussian_kernel(dist, self.bandwidth)
            shift_x += p[0] * weight
            shift_y += p[1] * weight
            scale += weight
        shift_x = shift_x / scale
        shift_y = shift_y / scale
        return [shift_x, shift_y]

    def cluster_points(self, X):
        cluster_labels = []
        cluster_labelx = 0
        cluster_centers = []
        for i, point in enumerate(X):
            if (len(cluster_labels) == 0):
                cluster_labels.append(cluster_labelx)
                cluster_centers.append(point)
                cluster_labelx += 1
            else:
                for center in cluster_centers:
                    dist = self.distance(point, center)
                    if (dist < self.cluster_thresh):
                        cluster_labels.append(cluster_centers.index(center))
                if (len(cluster_labels) < i + 1):
                    cluster_labels.append(cluster_labelx)
                    cluster_centers.append(point)
                    cluster_labelx += 1
        return cluster_labels, cluster_centers

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def gaussian_kernel(self, distance, bandwidth):
        return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)
