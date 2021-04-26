import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


def do_pca(X, n_comp):
    pca = PCA(n_comp)
    pca.fit(X)

    # X -> covariance matrix를 계산(Important) -> Eigen decompsition(복잡도 (O(N^3)) -> lambda(Eigen value), priciple compoment(pc) (Eigen vector)
    # 아이젠 벨류가 중요한 의미를 가짐. 크면 클수록 데이터 설명에 중요함
    # 피처를 독립적으로 만듦으로써 난수를 만들기 쉬워짐.
    pc, mean_orig, cov_e = pca.components_, pca.mean_, np.diag(pca.explained_variance_)

    X_tf = pca.transform(X) # 샘플 개수 만큼, n 컴포넌트 만큼 모양이 잡힘(?, n_comp)
    mean_tf = np.mean(X_tf, 0) # 0번 축을 기준으로 평균을 계산. (n_comp, )

    return mean_tf, pc, mean_orig, cov_e


def generate_random_data(mean_tf, pc, mean_orig, cov_e):
    f_rand = np.random.multivariate_normal(mean=mean_tf, cov=cov_e, size=10) # 변환된 평균값, 아이젠 벨류

    # PCA 알고리즘을 역으로 수행하는 것. # X_tf = (X-m) * PC
    # X_tf * PC^T + m = X
    # pc는 eigen face.
    X_gen = np.matmul(f_rand, pc) + mean_orig.reshape((1, -1)) # 원본 데이터 도메인(공간)으로 돌아가게 됨

    return X_gen



def main():
    # 피처의 크기 = (70000, 784) / as_frame = True일 경우 pandas의 Dataframe으로 줌
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

    # 인덱스 생성 후 셔플
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X = X[ind] # X 생성 목적이라 y는 불필요
    X = X[:30000]

    print (X.shape) # (30000, 784)

    # 784 -> 4로 줄어듦을 의미
    mean_tf, pc, mean_orig, cov_e = do_pca(X, 20) # 랜덤 데이터의 생성 퀄리티를 조절 가능. 4 -> ?
    X_gen = generate_random_data(mean_tf, pc, mean_orig, cov_e)

    fig, ax = plt.subplots(1, 10)
    for i in range(10):
        ax[i].imshow(X_gen[i].reshape((28, 28)), cmap='spring')
    plt.show()

    # 이렇게 하면 PCA를 이용해서 랜덤 데이터를 계속해서 생성 가능

if __name__ == "__main__":
    sys.exit(main())