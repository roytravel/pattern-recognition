from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

'''
age: age in years
sex
bmi: body mass index
bp: average blood pressure
tc: T-Cells (a type of white blood cells)
ldl: low-density lipoproteins
hdl: high-density lipoproteins
tch: thyroid stimulating hormone
ltg: lamotrigine
glu: blood sugar level

               The MEANS Procedure

 Variable      N            Mean         Std Dev
 �����������������������������������������������
 age         442    -3.64182E-16       0.0476190
 sex         442    1.308343E-16       0.0476190
 bmi         442    -8.04535E-16       0.0476190
 map         442    1.281655E-16       0.0476190
 tc          442    -8.78508E-17       0.0476190
 ldl         442    1.327024E-16       0.0476190
 hdl         442    -4.57465E-16       0.0476190
 tch         442    3.777301E-16       0.0476190
 ltg         442    -3.82858E-16       0.0476190
 glu         442    -3.41288E-16       0.0476190
 y           442     152.1334842      77.0930045
 �����������������������������������������������
'''

class LinearRegressor:
    def __init__(self):
        self. W = None

    def fit(self, X, y):
        '''
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_targets)
        :return:
        '''
        self.W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)


    def predict(self, X):
        if self.W is None:
            raise Exception('undefined the model yet.')

        return np.matmul(X, self.W)

def calculate_mse(y_pred, y_target):
    if len(y_pred.shape) == 1: y_pred = np.expand_dims(y_pred, 1)
    if len(y_target.shape) == 1: y_target = np.expand_dims(y_target, 1)
    return np.mean(np.sum(np.square(y_pred - y_target), 1))

def main():
    # X has been standardized to have mean 0 and std 1 :)
    print('[*] load the dataset')
    X, t = load_diabetes(return_X_y=True)
    print(X.shape, t.shape)
    print(X[0, :])
    X = np.concatenate([X, np.ones([len(X), 1])], 1)
    print()

    print('[*] splitting the dataset')
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    X, t = X[ind], t[ind]
    TR_SIZE = int(len(X) * 3. / 4.)
    tr_X, tr_t = X[:TR_SIZE], t[:TR_SIZE]
    val_X, val_t = X[TR_SIZE:], t[TR_SIZE:]
    print()

    print('[*] training the handmade model')
    my_model = LinearRegressor()
    my_model.fit(tr_X, tr_t)
    val_y_my = my_model.predict(val_X)
    print()

    print('[*] training the model using the sklearn')
    lib_model = LinearRegression(fit_intercept=False)
    lib_model.fit(tr_X, tr_t)
    val_y_lib = lib_model.predict(val_X)
    print()

    print('[*] the predictors for the validation set')
    print(val_y_my[:5])
    print(val_y_lib[:5])
    print()

    print('[*] the model parameters')
    print(my_model.W)
    print(lib_model.coef_)
    print()

    print('[*] calculating the mean squared error')
    print('mse for my model: ', calculate_mse(val_y_my, val_t))
    print('mse for lib model: ', calculate_mse(val_y_lib, val_t))
    print()

    print('[*] calculating the mean squared error with the sklearn')
    print('mse for my model: ', mean_squared_error(val_y_my, val_t))
    print('mse for lib model: ', mean_squared_error(val_y_lib, val_t))


if __name__ == '__main__':
    main()