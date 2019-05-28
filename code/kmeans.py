import numpy as np
from copy import deepcopy


class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None




    def update_means(self, means, assignments,features):
        holder = {}
        means = []
        for i in range(self.n_clusters):
            holder[i] = []
        for i in assignments:
            for x in range(self.n_clusters):
                if (i==x):
                    holder[x].append(features[i])
        for i in holder.values():
            means.append(np.mean(np.asarray(i), axis = 0))
        means = np.asarray(means)
        return means

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        bool = False 
        index = np.random.choice(features.shape[0], self.n_clusters, replace=False)  
        means = []
        oldmeans = []
        count = 0
        for i in range(self.n_clusters):
            means.append(features[index[i]])
        for i in range(100):
            count += 1
            assignments = []
            for i in features:
                for x in range(self.n_clusters):
                    dist = np.linalg.norm(i-means[x])
                    assignments.append(dist)
            assignments = np.asarray(assignments)
            assignments = assignments.reshape(-1,self.n_clusters)
            #print(assignments)
            assignments = np.argmin(assignments, axis = 1)
            
            oldmeans = means
            print(oldmeans)
            #means = self.update_means(means, assignments, features)
            #for i in range(self.n_clusters):
                #means[i] = np.mean(features[assignments == i], axis = 0)
            
            means = [features[assignments == i].mean(axis = 0) for i in range(self.n_clusters)]
            print(means)
            print(count)
            if(np.array_equal(oldmeans,means)):
                break
        self.means = means


        




    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        means = self.means
        assignments = []
        for i in features:
             for x in means:
                dist = np.linalg.norm(i-x)
                assignments.append(dist)
        assignments = np.asarray(assignments)
        assignments = assignments.reshape(-1,self.n_clusters)
        assignments = np.argmin(assignments, axis = 1)
        return assignments

        
    
        




    





    



