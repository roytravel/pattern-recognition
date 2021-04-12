import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

def plot(data):
    # pandas는 크게 dataframe과 series를 제공.
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.DataFrame(data.target, columns=['species'])

    print (X.iloc[:3, [0, 2]])
    print(y.iloc[:3])

    """To check data-point's distribution to recognize the data visually."""

    pd.plotting.scatter_matrix(X, c=y['species'], diagonal='kde') # kernel density estimation
    # plt.show()

    # (150, 4), (150, ) => (150, 5)
    iris_data = pd.concat([X, y], axis=1)
    iris_data['species'] = iris_data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    sns.pairplot(iris_data, hue='species') # hue means color. # sns is beatiful than pandas's plotting.
    plt.show()

def naive_bayes_partial_features(data):
    print (type(data))
    X = data.data[:, [2, 3]] # 3, 4번째 피쳐만 가지고 사용 학습 예정. 시각화 결과 petal length와 petal width가 학습에 적절.
    y = data.target

    print (X.shape, y.shape)
    print (X[:3]) # 논문 쓸 때 전체 데이터 집합을 X 또는 그리스어로 카이를 사용함
    print (y[:3]) # 레이블은 t또는 y로 표현함. 관행적. """ 0, 1, 2 """

    # y의 레이블이 0, 1, 2 순서대로 정렬되어 있음. 이를 섞기 위해 아래와 같이 인덱스 생성 후 셔플링.
    index = np.arange(len(X))
    np.random.shuffle(index)

    X = X[index]
    y = y[index]

    # len(X) * 3. / 4. => float
    # int(len(X) * 3. / 4.) => int
    TR_SIZE = int(len(X) * 3. / 4.)
    tr_X, val_X = X[:TR_SIZE], X[TR_SIZE:]
    tr_y, val_y = y[:TR_SIZE], y[TR_SIZE:]

    model = GaussianNB()
    model.fit(tr_X, tr_y)

    val_y_pred = model.predict(val_X)
    score = model.score(val_X, val_y)
    print(f'score: {score}')

    # 1개의 plot에 2개의 subplot을 넣음. to check difference between ground-truth and prediction
    fig, ax = plt.subplots(1, 2)

    min_f1, max_f1 = np.min(X[:, 0]), np.max(X[:, 0]) # 가져온 petal length에서 가장 작은 값 큰 값 가져옴
    min_f2, max_f2 = np.min(X[:, 1]), np.max(X[:, 1]) # 가져온 petal width에서 가장 작은 값 큰 값 가져옴
    f1_space = np.linspace(min_f1, max_f1, 100)
    f2_space = np.linspace(min_f2, max_f2, 100)
    f1f1, f2f2 = np.meshgrid(f1_space, f2_space) # 위 두개를 가지고 mesh-grid를 만듦 --> (100, 100), (100, 100)
    feature_space = np.stack([f1f1, f2f2], 2) # (100, 100, 2) //
    feature_space = feature_space.reshape((-1, 2)) # (10000, 2) 위 100x100을 묶어서 하나의 공간으로 표기.
    prob_feature_space = model.predict(feature_space) # (10000, 3) 어느 클래스로 분류될 지 확률 값을 알아 낼 수 있음. 클래스가 3개임.
    prob_feature_space = prob_feature_space.reshape((100, 100, 3))

    ax[0].imshow(prob_feature_space, extent=(f1f1.min(), f1f1.max(), f2f2.min(), f2f2.max()), origin='lower', aspect='auto', alpha=0.3) #투명도 0.3
    ax[1].imshow(prob_feature_space, extent=(f1f1.min(), f1f1.max(), f2f2.min(), f2f2.max()), origin='lower', aspect='auto', alpha=0.3) #투명도 0.3

    ax[0].scatter(val_X[:, 0], val_X[:, 1], c=val_y, cmap=matplotlib.colors.LinearSegmentationColormap.from_list('cmap', colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]))
    ax[1].scatter(val_X[:, 0], val_X[:, 1], c=val_y_pred, cmap=matplotlib.colors.LinearSegmentationColormap.from_list('cmap', colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]))

    ax[0].set_title('ground truth')
    ax[1].set_title('prediction')
    fig.show()
    plt.show()


def main(argv):
    data = load_iris()

    # Plot.
    # plot(data)

    # Create classifier.
    naive_bayes_partial_features(data)

if __name__ == "__main__":

    sys.exit(main(sys.argv))