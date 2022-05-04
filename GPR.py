import numpy as np
from utils.dataloader import load
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, solve_triangular

GPR_CHOLESKY_LOWER = True

class GPR(object):
    def __init__(self):
        self.alpha = 1e-10

    def se_kernel(self, Xa, Xb):
        # dists_ = pdist(Xa, metric='sqeuclidean')
        # K_ = np.exp(-.5 * gamma * dists_)
        # K_ = squareform(K_)
        # np.fill_diagonal(K, 1)

        dists = cdist(Xa, Xb, metric='sqeuclidean')
        K = np.exp(- dists / (self.median))

        # print(np.equal(K, K_))
        return K

    def fit(self, X, y):
        # Normalize fitting data
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X = (X-self.X_mean)/self.X_std

        self.y_mean = np.mean(y, axis=0)
        self.y_std = np.std(y, axis=0)
        self.y = (y-self.y_mean)/self.y_std

        # print(pdist(X, 'sqeuclidean'))
        self.median = np.median(pdist(X, 'sqeuclidean'))

        K = self.se_kernel(self.X, self.X)
        # noise™
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                f"The kernel, {self.kernel_}, is not returning a positive "
                "definite matrix. Try gradually increasing the 'alpha' "
                "parameter of your GaussianProcessRegressor estimator.",
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y,
            check_finite=False,
        )
        return self
    
    def predict(self, X):
        X = (X-self.X_mean)/self.X_std
        K_trans = self.se_kernel(X, self.X)
        y_mean = K_trans @ self.alpha_
        y_mean = self.y_std * y_mean + self.y_mean
        # V = solve_triangular(
        #     self.L_, K_trans.T, lower=GPR_CHOLESKY_LOWER, check_finite=False
        # )
        
        return y_mean



if __name__ == '__main__':
    train_X, train_y = load('../data/trainInterpolate.json')
    test_X, test_y = load('../data/valExtrapolate.json')
    gpr = GPR()
    train_X = train_X[:100, :]
    train_y = train_y[:100, :]
    train_y = train_y.reshape(-1, 27)
    gpr.fit(train_X, train_y)
    pred_y = gpr.predict(test_X)
    test_y = test_y.reshape(-1,27)
    print(np.max((test_y - pred_y) / test_y))