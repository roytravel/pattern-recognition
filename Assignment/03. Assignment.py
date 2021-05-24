"""
Algorithm Implementation
    - PCA
    - kernel-PCA
    - LDA
    - LLE(Locally Linear Embedding)
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.sparse import linalg, eye
from pyamg import smoothed_aggregation_solver
from sklearn import neighbors
from scipy.spatial import KDTree
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import KernelCenterer
from sklearn.utils import extmath
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.spatial.distance import pdist, squareform

cmap = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'violet', 'lightblue']


class Library(object):
    def __init__(self):
        pass

    def pca(self, X, y):
        pca = PCA(n_components=2)
        X_tf = pca.fit_transform(X)

        fig = plt.figure()
        scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('SK-PCA')
        fig.show()


    def kpca(self, X, y):
        kpca = KernelPCA(n_components=2, kernel='cosine')
        X_tf = kpca.fit_transform(X)

        fig = plt.figure()
        scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('SK-kernel-PCA')
        fig.show()


    def lda(self, X, y):
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_tf = lda.fit_transform(X, y)

        fig = plt.figure()
        scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('SK-LDA')
        fig.show()


    def lle(self, X, y):
        lle = LocallyLinearEmbedding(n_neighbors=7)
        X_tf = lle.fit_transform(X)

        fig = plt.figure()
        scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('SK-LLE')
        fig.show()


class Handmade(object):
    def __init__(self):
        pass


    def PCA(self, X, y, num_components=2):

        # 고유값 분해(Eigen Decomposition)를 위해 공분산 행렬 계산
        X_meaned = X - np.mean(X, axis=0)
        covariant_matrix = np.cov(X_meaned, rowvar = False)

        # 공분산 행렬로부터 고유 값(Weight), 고유 벡터 추출(Vector)
        eigen_values, eigen_vectors = np.linalg.eigh(covariant_matrix)
        sorted_index = np.argsort(eigen_values)[::-1] #
        # sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        eigenvector_subset = sorted_eigenvectors[:, 0:num_components]
        X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

        fig = plt.figure()
        scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('HM-PCA')
        fig.show()


    def KPCA(self, X, y, gamma=0.3, n_components=2):
        """
        1) https://opendatascience.com/implementing-a-kernel-principal-component-analysis-in-python/
        2) https://medium.com/@ODSC/implementing-a-kernel-principal-component-analysis-in-python-495f04a7f85f
        3) https://sebastianraschka.com/Articles/2014_kernel_pca.html
        4) https://excelsior-cjh.tistory.com/167
        5) https://dnai-deny.tistory.com/18
        6) https://github.com/susobhang70/kernel_pca_lda/blob/master/KPCA_KLDA.py
        """

        kpca = HM_KernelPCA()

        # calculate euclidean distance matrix
        distance_matrix = kpca.find_distance_matrix(X)

        # find variance of one dimensional distance list
        variance = np.var(kpca.inverse_squareform(distance_matrix))

        # calculate kernel (using rbf kernel)
        gamma = 1 / (2 * variance)
        K = kpca.rbfkernel(gamma, distance_matrix)

        # centering kernel matrix
        mean = np.mean(K, axis=0)
        K_center = K - mean

        # finding eigen vector and eigen value
        eigen_values, eigen_vectors = np.linalg.eig(K_center)
        normalization_root = np.sqrt(eigen_values)
        eigen_vectors = eigen_vectors / normalization_root
        indexes = eigen_values.argsort()[::-1]
        direction_vectors = eigen_vectors[:, indexes[0: len(X)]]
        projected_data = np.dot(K, direction_vectors)

        fig = plt.figure()
        scatter = plt.scatter(projected_data[:, 0], projected_data[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('HM-kernel-PCA')
        fig.show()


    def LDA(self, X, y):
        """
        1)  https://www.section.io/engineering-education/linear-discriminant-analysis/
        2) https://ratsgo.github.io/machine%20learning/2017/03/21/LDA/
        3) https://www.python-engineer.com/courses/mlfromscratch/14-lda/
        4) https://sebastianraschka.com/Articles/2014_python_lda.html
        """
        kfda = KernelFDA(n_components=2, kernel='linear', gamma=0.1)  # exp(-gamma * ||x1-x2||^2) # 데이터에 따라 gamma 값을 조절해야 함
        X_tf = kfda.fit_transform(X, y)

        fig = plt.figure()
        scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('HM-kernel-LDA')
        fig.show()


    def LLE(self, X, y, k_neighbors=7, n_neighbors=2):
    # def LLE(self, X, y, n_neighbors=2):
        """
        1) https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning
        2) https://www.python-course.eu/linear_discriminant_analysis.php
        3) https://github.com/lxcnju/Locally-Linear-Embedding/blob/master/lle.py
        """

        # 데이터 샘플 개수
        n_samples = X.shape[0]

        # pair-wise distance 계산
        dist_mat = pairwise_distances(X)

        # neighbors의 인덱스
        neighbors = np.argsort(dist_mat, axis=1)[:, 1: k_neighbors + 1]

        # neighbor combination matrix
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            mat_z = X[i] - X[neighbors[i]]
            mat_c = np.dot(mat_z, mat_z.transpose())
            w = np.linalg.solve(mat_c, np.ones(mat_c.shape[0]))
            W[i, neighbors[i]] = w / w.sum()

        # 희소 행렬 M
        I_W = np.eye(n_samples) - W
        M = np.dot(I_W.transpose(), I_W)

        # solve the d+1 lowest eigen values
        eigen_values, eigen_vectors = np.linalg.eig(M)
        index = np.argsort(eigen_values)[1: n_neighbors + 1]
        selected_eig_values = eigen_values[index]
        selected_eig_vectors = eigen_vectors[index]
        eig_values = selected_eig_values
        low_X = selected_eig_vectors.transpose() # print(low_X.shape)

        fig = plt.figure()
        scatter = plt.scatter(low_X[:, 0], low_X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
        handles, labels = scatter.legend_elements()
        plt.legend(handles, np.unique(y))
        plt.title('HM-LLE')
        fig.show()



class HM_KernelPCA(object):

    def __init__(self):
        pass

    def inverse_squareform(self, matrix):
        inv_sqfrm = []
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix[i])):
                inv_sqfrm.append(matrix[i][j])
        inv_sqfrm = np.array(inv_sqfrm)
        return inv_sqfrm


    def rbfkernel(self, gamma, distance):
        return np.exp(-gamma * distance)


    def find_distance_matrix(self, data):
        euclid_distance = []
        for i in data:
            distance = []
            for j in data:
                distance.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
            distance = np.array(distance)
            euclid_distance.append(distance)
        euclid_distance = np.array(euclid_distance)
        return euclid_distance



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



def main():
    X, y = load_digits(return_X_y=True)
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X, y = X[ind], y[ind]

    L = Library()
    # L.pca(X, y)
    # L.kpca(X, y)
    # L.lda(X, y)
    L.lle(X, y)

    H = Handmade()
    # H.PCA(X, y)
    # H.KPCA(X, y)
    # H.LDA(X, y)
    H.LLE(X, y)
    plt.show()


if __name__ == '__main__':
    sys.exit(main())