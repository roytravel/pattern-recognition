# Linear Regression 예제
from sklearn.linear_model import LinearRegression
import numpy as np # n차원 배열을 다루기 위함
import matplotlib.pyplot as plt

def main():
    # 직교 좌표계
    # 유니폼하게 난수 생성(최저값 -5.0, 최고 값 5.0, 난수 개수 50개)
    X = np.random.uniform(low=-5., high=5., size=500)
    # 난수 = 뒤죽박죽 --> 정렬
    X.sort()
    print (type(X))
    # (50,) = 50차원의 데이터 = 50개의 피처 = 50개의 데이터 개수
    print (X.shape)

    # expand_dims = 차원을 1 추가해줌
    X = np.expand_dims(X, 1)
    print (X.shape)

    # 직선 생성 + X의 shape 맞춰줌
    # y = a*x + b + e
    t = 0.5 * X + -1. + np.random.normal(size=[500, 1])
    print (X.shape, t.shape)

    # 학습 데이터 개수
    TR_SIZE = int(len(X) * 3. / 4.)
    
    # Linear 모델 학습에 사용
    tr_X, val_X = X[:TR_SIZE], X[TR_SIZE:]
    tr_t, val_t = t[:TR_SIZE], t[TR_SIZE:]

    model = LinearRegression()
    model.fit(tr_X, tr_t)
    # coef_ = a를 학습, intercept_ = b를 학습
    print(model.coef_, model.intercept_)

    fig = plt.figure()
    # [:, 0] = 가로50, 세로 1 = 한 줄 전부
    plt.scatter(tr_X[:, 0], tr_t[:, ], color='b', label='train')
    plt.scatter(val_X[:, 0], val_t[:, 0], color='r', label='val')

    x_space = np.expand_dims(np.linspace(-5., 5., 500), 1)
    y_p = model.predict(x_space)
    y = 0.5 * x_space -1.

    plt.plot(x_space[:, 0], y[:, 0], color='b', label='true')
    plt.plot(x_space[:, 0], y_p[:, 0], color='r', label='predict')
    # 격자 표시
    plt.grid(True)
    plt.legend()
    plt.show()

    # figure 생성
    # fig = plt.figure()
    # plt.scatter(X, t, color='b')
    # plt.show()

if __name__ == '__main__':
    main()
