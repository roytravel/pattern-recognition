# Manifold learning -> 데이터 시각화할 때 주로사용.
# 고차원 데이터 분포를 결따라 맞은 차원에 임베딩을 한다.

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import load_digits
from sklearn.manifold import *
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 8x8 = 64차원을 2차원으로 축소하여 시각화.

cmap = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'violet', 'lightblue']

# def mlp_embedding(X, y):
#     X = X / 16.
#     t = y.reshape((-1,1 ))
#     encoder = OneHotEncoder()
#     t = encoder.fit_transform(t).toarray()
#     mlp = mlp_bp.MLP.fit(X, t, 1e-1)
#
#     h = mlp.layers[0](X)
#     h = mlp.layers[1](h, do_act=False)
#
#     X_tf = h
#
#     fig = plt.figure()
#     scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
#     handles, labels = scatter.legend_elements()
#     plt.legend(handles, np.unique(y))
#     plt.title('mlp')
#
#     fig.show()


def isomap(X, y):
    # Isomap 모델에 매개변수가 여러개 있음.
    isomap = Isomap(n_neighbors=30)
    X_tf = isomap.fit_transform(X) # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('Isomap')

    fig.show()

def lle(X, y): # Manifold 모델을 이해하기에 가장 적합한 모델.
    lle = LocallyLinearEmbedding(n_neighbors=30)
    X_tf = lle.fit_transform(X)  # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('LLE')

    fig.show()

def tsne(X, y):
    # 위 두 모델과 큰 연관성은 없음. 하지만 tsne는 lle와 달리 지역적 특정을 반영하면서도
    # 데이터셋 전체 데이터에 대해 즉 global 관계를 포함해서 임베딩함
    tsne = TSNE() # 기본 default embedding 차원: 2, because 시각화에 초점
    X_tf = tsne.fit_transform(X)  # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('t-SNE')

    fig.show()

def pca(X, y):
    pca = PCA(n_components=2)  # 기본 default embedding 차원: 2, because 시각화에 초점
    X_tf = pca.fit_transform(X)  # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('PCA')

    fig.show()

def kernel_pca(X, y):
    # Kernel을 선택하지 않으면 PCA와 동일한 결과를 얻음. default가 linear
    # Kernel에 따라 모델의 성능도 많이 바뀜. 바꿔가면서 실험 필요
    kpca = KernelPCA(n_components=2, kernel='cosine')  # 기본 default embedding 차원: 2, because 시각화에 초점
    X_tf = kpca.fit_transform(X)  # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('Kernel-PCA')

    fig.show()

def lda(X, y):
    lda = LinearDiscriminantAnalysis(n_components=2)  # 기본 default embedding 차원: 2, because 시각화에 초점
    # LDA는 라벨이 필요함
    X_tf = lda.fit_transform(X, y)  # 64차원 데이터 -> 2차원으로 transform됨

    fig = plt.figure()
    scatter = plt.scatter(X_tf[:, 0], X_tf[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(cmap))
    handles, labels = scatter.legend_elements()
    plt.legend(handles, np.unique(y))
    plt.title('LDA')

    fig.show()



def main():

    # load data
    X, y = load_digits(return_X_y=True)

    # Shuffle data
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X, y = X[ind], y[ind]

    isomap(X, y)
    lle(X, y)
    tsne(X, y)
    pca(X, y)
    kernel_pca(X, y)
    lda(X, y)
    # mlp_embedding(X, y)

    plt.show()

if __name__ == "__main__":
    sys.exit(main())
