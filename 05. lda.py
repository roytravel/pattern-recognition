import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA




def load_data():
    X, y = load_digits(return_X_y=True)

    # 인덱스 생성 후 셔플
    ind = np.arange(len(X))
    np.random.shuffle(ind)

    X = X[ind]
    y = y[ind]

    TR_SIZE = int(len(X) * 3. / 4.)
    tr_X, te_X = X[:TR_SIZE], X[TR_SIZE:]
    tr_y, te_y = y[:TR_SIZE], y[TR_SIZE:]

    return tr_X, tr_y, te_X, te_y

def transform(tr_X, tr_y, te_X, te_y):
    # LDA를 하면 자연스럽게 차원축소가 일어남. 몇차원까지 축소할지 지정(=2)
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(tr_X, tr_y)

    # 8 x 8 = 64차원. n_components를 20으로 줘도 괜찮은 수준
    # lda = PCA(n_components=20)
    # lda.fit(tr_X, tr_y)



    # 훈련과 테스트셋에 대해 트랜스폼
    tr_X_tf = lda.transform(tr_X)
    te_X_tf = lda.transform(te_X)

    # 시각화
    fig, ax = plt.subplots(1, 5)

    for i in range(5):
        ax[i].imshow(tr_X[i].reshape((8, 8)), cmap='spring')
    fig.show()

    fig = plt.figure()
    color_map = {
        0: 'r',
        1: 'g',
        2: 'b',
        3: 'c',
        4: 'm', # 마젠타
        5: 'y',
        6: 'k', # 검은색
        7: 'grey',
        8: 'lightblue',
        9: 'violet'
    }

    for i in range(10):
        ind = np.where(tr_y == i)[0]
        plt.scatter(tr_X_tf[ind, 0], tr_X_tf[ind, 1], c=color_map[i], label=str(i))
    plt.legend()
    fig.show()

    return tr_X_tf, te_X_tf




def classify(tr_X, tr_y, te_X, te_y, title):
    print (title)

    cls = KNeighborsClassifier(n_neighbors=5)
    cls.fit(tr_X, tr_y)

    print('[-] accuracy:', cls.score(te_X, te_y))


def main():
    
    # (?, 64), (?, )
    tr_X, tr_y, te_X, te_y = load_data()

    tr_X_tf, te_X_tf = transform(tr_X, tr_y, te_X, te_y)


    classify(tr_X, tr_y, te_X, te_y, 'original')
    classify(tr_X_tf, tr_y, te_X_tf, te_y, 'lda')

    plt.show()


if __name__ == "__main__":
    sys.exit(main())