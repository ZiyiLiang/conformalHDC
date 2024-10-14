import numpy as np


class NormalClusters():
    def __init__(self, d, num_classes):
        self.d = d   # feature dimensions
        self.num_classes = num_classes

    def sample(self, means, covs, sizes):
        X = []
        Y = []
        for i in range(self.num_classes):
            mean = means[i]
            cov = covs[i]
            n = sizes[i]
            X.append(np.random.multivariate_normal(mean, cov, n))
            Y.append(np.array([i]*n))
        
        return np.concatenate(X), np.concatenate(Y)
    

    