import numpy as np
from utils.dataloader import load
from scipy.spatial.distance import cdist, pdist
class GPR(object):
    def __init__(self) -> None:
        pass

    def se_kernel(self, gamma):
        pass

    def fit(self, X, y):
        # Normalize fitting data
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        self.X = (X-X_mean)/X_std

        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        self.y = (y-y_mean)/y_std

        # self.X_median = np.median(self.X, axis=0)
        # self.y_median = np.median(self.y, axis=0)
        
        median = np.median(pdist(self.X, 'sqeuclidean'))
        dists = cdist(self.X, self.y,
                metric='sqeuclidean')
        K = np.exp(-.5 * dists)

        # dists = cdist(self.X / length_scale, self.Y / length_scale, metric='sqeuclidean')
        # K = np.exp(-.5 * dists)


train_X, train_y = load('../data/trainInterpolate.json')
gpr = GPR()
gpr.fit(train_X, train_y)