from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def main():
    X = np.random.uniform(low=-5., high=5., size=50)
    X.sort()
    X = np.expand_dims(X, 1)
    X = np.concatenate([X, np.ones([len(X), 1])], 1)
    # t = 0.5 * X + -1. + np.random.normal(size=[50, 1])
    W = np.array([0.5, -1.]).T
    t = np.expand_dims(np.matmul(X, W), 1) + np.random.normal(size=[50, 1])
    print(X.shape, t.shape)

    TR_SIZE = int(len(X) * 3. / 4.)
    tr_X, val_X = X[:TR_SIZE], X[TR_SIZE:]
    tr_t, val_t = t[:TR_SIZE], t[TR_SIZE:]

    model = LinearRegression(fit_intercept=False)
    model.fit(tr_X, tr_t)
    print(model.coef_, model.intercept_)

    fig = plt.figure()
    plt.scatter(tr_X[:, 0], tr_t, color='b', label='train')
    plt.scatter(val_X[:, 0], val_t, color='r', label='val')

    x_space = np.expand_dims(np.linspace(-5., 5., 50), 1)
    x_space = np.concatenate([x_space, np.ones([50, 1])], 1)
    y_p = model.predict(x_space)
    y = np.matmul(x_space, W.T)

    print(x_space.shape, y_p.shape)
    plt.plot(x_space[:, 0], y, color='b', label='true')
    plt.plot(x_space[:, 0], y_p, color='r', label='prediction')

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()