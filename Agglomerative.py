import numpy as np

class Agglomerative:

    def __init__(self, k, linkage="average"):
        '''
        Define a mean-shift algorithm with known number of clusters.
        Revised from https://github.com/susobhang70/clustering_algorithms.
        '''
        self.k = k
        self.linkage = linkage

    def euclidean_distance(self, vec1, vec2):
        return np.sqrt(sum((vec1 - vec2) ** 2))

    def min_distance(self, clust1, clust2, distances):
        min_dist = 1e20
        for i in clust1.node_vector:
            for j in clust2.node_vector:
                try:
                    distance = distances[(i,j)]
                except:
                    try:
                        distance = distances[(j,i)]
                    except:
                        distance = self.euclidean_distance(clust1.vec, clust2.vec)
                if distance < min_dist:
                    min_dist = distance
        return min_dist

    def fit(self, X):
        n, dim = X.shape
        data = np.array(X)
        # Cluster the rows of the data matrix
        distances = {}
        currentclustid = -1
        # Cluster nodes are initially just the individual rows
        nodes = [ClusterNode(np.array(data[i]), id=i) for i in range(len(data))]
        while len(nodes) > self.k:
            lowestpair = (0,1)
            closest = self.euclidean_distance(nodes[0].vec,nodes[1].vec)
            # Loop through every pair looking for the smallest distance
            for i in range(len(nodes)):
                for j in range(i+1,len(nodes)):
                    if (nodes[i].id,nodes[j].id) not in distances:
                        if self.linkage == "single":
                            distances[(nodes[i].id,nodes[j].id)] = self.min_distance(nodes[i], nodes[j], distances)
                        elif self.linkage == "average":
                            distances[(nodes[i].id,nodes[j].id)] = self.euclidean_distance(nodes[i].vec,nodes[j].vec)
                        else:
                            print("This linkage criterion is unavailble!")
                    d = distances[(nodes[i].id,nodes[j].id)]
                    if d < closest:
                        closest = d
                        lowestpair = (i,j)
            # Calculate the average of the two nodes
            len0 = len(nodes[lowestpair[0]].node_vector)
            len1 = len(nodes[lowestpair[1]].node_vector)
            mean_vector = [(len0*nodes[lowestpair[0]].vec[i] + len1*nodes[lowestpair[1]].vec[i])/(len0 + len1) \
                            for i in range(len(nodes[0].vec))]
            # Create the new cluster node
            new_node = ClusterNode(
                np.array(mean_vector), 
                currentclustid, 
                left = nodes[lowestpair[0]], 
                right = nodes[lowestpair[1]], 
                distance = closest, 
                node_vector = nodes[lowestpair[0]].node_vector + nodes[lowestpair[1]].node_vector
                )
            # Cluster ids that are not in the original set are negative
            currentclustid -= 1
            del nodes[lowestpair[1]]
            del nodes[lowestpair[0]]
            nodes.append(new_node)
        clusters = []
        cluster_centers = []
        cluster_labels = - np.zeros(n)
        for i in range(self.k):
            cluster_centers.append(nodes[i].vec)
            clusters.append(nodes[i].node_vector)
        for j in range(n):
            for i in range(self.k):
                if j in clusters[i]:
                    cluster_labels[j] = i
        return cluster_centers, cluster_labels

class ClusterNode:

    def __init__(self, vec, id, left=None, right=None, distance=0.0, node_vector=None):
        '''
        Define a cluster node.
        '''
        self.leftnode = left
        self.rightnode = right
        self.vec = vec
        self.id = id
        self.distance = distance
        if node_vector is None:
            self.node_vector = [self.id]
        else:
            self.node_vector = node_vector[:]
