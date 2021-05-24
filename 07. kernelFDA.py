import sys
import numpy as np
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

cmap = ['red', 'green', 'blue', 'black', 'cyan', 'magenta', 'yellow', 'gray', 'violet', 'lightblue']

class KernelFDA(object):
    def __init__(self, n_components, kernel, gamma=None):
        if not kernel or kernel == "rbf": self.kernel = self._rbf
        elif kernel == 'linear': self.kernel = self._linear

        self.gamma = gamma
        self.max_components = n_components


    def _rbf(self, X1, X2):
        if not self.gamma:
            self.gamma = 1. / X1.shape[1]

        K = rbf_kernel(X1, X2, self.gamma)
        return K


    def _linear(self, X1, X2):
        K = linear_kernel(X1, X2)
        return K


    def _kernel(self, X1, X2):
        K = self.kernel(X1, X2)
        return K # n_X1 = 10, # n_X2 = 20, # n_K = 10 x 20


    def _calc_scatter_between(self, K_cls, K): # M
        K = np.mean(K, 0)  # (N, 0)
        M = np.zeros((len(K), len(K)))
        for i, kc in enumerate(K_cls):
            tmp = (np.mean(kc, 0) - K)[np.newaxis, :] # (1, N)
            M += (np.matmul(tmp.T, tmp) * kc.shape[0]) # (N, N)
        return M


    def _calc_scatter_within(self, K_cls): # N

        N = np.zeros((K_cls[0].shape[1], K_cls[0].shape[1]))
        for kc in K_cls:
            nc = len(kc)
            I = np.eye(nc)
            one = np.ones((nc, nc)) / nc
            tmp = np.matmul(np.matmul(kc.T, I - one), kc) # (N, N)
            N += tmp
        return N


    def _calc_mean_cls(self, X, X_cls): # M_{*}
        K = self._kernel(X, X) # n_K = N x N
        K_cls = [self._kernel(xc, X) for xc in X_cls] # n_K_{0} = m x N

        return K_cls, K


    def _get_evc_separability(self, evc, evl, S_b, S_w):
        w = np.expand_dims(evc.T, 1)
        a = np.squeeze(np.matmul(np.matmul(w, S_b), np.transpose(w, [0, 2, 1])), (1, 2))
        b = np.squeeze(np.matmul(np.matmul(w, S_w), np.transpose(w, [0, 2, 1])), (1, 2))

        idx_nonzero = np.array(np.nonzero(b))[0]
        a = a[idx_nonzero]
        b = b[idx_nonzero]
        sep = a / b
        evl, evc = evl[idx_nonzero], evc[:, idx_nonzero]
        idx = sep.argsort()[::-1]

        return evl[idx], evc[:, idx]


    def fit(self, X, y):
        self.X_fit = X

        uq_y = np.unique(y)

        if not self.max_components:
            self.max_components = np.min([X.shape[1], len(uq_y), self.max_components])

        X_cls = [X[y == uy] for uy in uq_y]

        self.K_cls, self.K = self._calc_mean_cls(X, X_cls)
        self.M = self._calc_scatter_between(self.K_cls, self.K)
        self.N = self._calc_scatter_within(self.K_cls)

        Sigma = np.matmul(np.linalg.pinv(self.N), self.M)
        evl, evc = np.linalg.eig(Sigma)

        evl, evc = self._get_evc_separability(evc, evl, self.M, self.N) # Eigen value, Eigen vector # (N, ), (N, N) <- column vector x N

        self.L = evl[:self.max_components]
        self.A = evc[:, :self.max_components]


    def transform(self, X):
        K = self._kernel(X, self.X_fit) # 커널 계산 --> 커널 크기: (m, N)
        return np.matmul(K, self.A) # (m, max_comp)


    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def kfda_my(X, y):
    kfda = KernelFDA(n_components=2, kernel='linear', gamma=0.1) # exp(-gamma * ||x1-x2||^2) # 데이터에 따라 gamma 값을 조절해야 함
    # kfda.fit(X, y)
    # X_tf = kfda.transform(X)
    X_tf = kfda.fit_transform(X, y)

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('Kernel-FDA')
    fig.show()


def lda_sk(X, y):
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_tf = lda.fit_transform(X, y)

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('LDA')
    fig.show()


def main():
    X, y = load_digits(return_X_y=True)
    X = X / 16.

    lda_sk(X, y)
    kfda_my(X, y)
    plt.show()


if __name__ == "__main__":
    sys.exit(main())