# -*- coding:utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder


############################################# Base class
class Activation:
    def __init__(self):
        pass

    # 클래스의 인스턴스를 함수처럼 사용할 수 있게 해주는 내장 함수
    def __call__(self, x):
        return self._function(x)

    # 실제 액티베이션 함수들의 함수 꼴을 구현할 것
    def _function(self, x):
        return 0.

    # 함수에 대한 미분값 계산
    def calc_gradients(self, x):
        return 0.


class LossFunction:
    def __init__(self):
        pass

    def __call__(self, y, t):
        return self._function(y, t)

    def _function(self, y, t):
        return 0.

    def calc_gradients(self, y, t):
        return 0.

############################################# Activation functions
# sigmoid, tanh, relu, softplus, prelu, erelu, leaky-relu

class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def _function(self, x):
        f = 1. / (1. + np.exp(-x))

        return f

    # Too simple
    def calc_gradients(self, x):
        f = self(x)
        return np.multiply(f, 1. - f)

# Sigmoid의 확장판
class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def _function(self, x):
        exps = np.exp(x - np.max(x, 1, keepdims=True))
        f = exps / np.sum(exps, 1, keepdims=True)

        return f

    # 조금 복잡하게 생김. 다른 Activation function들과 달리 미분 값이 matrix 형태로 나옴.
    # 이유는 계산 할때 위 아래로 값이 들어가는데 분모에 서메이션이 들어가서 Matrix로 나옴.
    def calc_gradients(self, x):
        f = self(x)
        # 3차원 Tensor 형태로 생성
        g = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        # 마지막 레이어 출력 값이 10차원임(데이터셋에서)

        diag = np.multiply(f, 1. - f)

        for i in range(x.shape[1]):
            g[:, i, i] = diag[:, i]
        for i in range(x.shape[1]):
            for j in range(x.shape[1]):
                if i == j: continue
                g[:, i, j] = - np.multiply(f[:, i], f[:, j])

        return g



class TanH(Activation):
    def __init__(self):
        super(TanH, self).__init__()

    def _function(self, x):
        return 0.

    def calc_gradients(self, x):
        return 0.


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()

    def _function(self, x):
        return 0.

    def calc_gradients(self, x):
        return 0.

############################################# Loss functions
# 최근에 많이 쓰이는 종류가 몇 개 없음.
# 기본 베이스는 크게 두개 cross entropy, mean squared error, . . .

class CrossEntropy(LossFunction):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def _function(self, y, t):
        y = y + 1e-24
        f = -np.sum(np.multiply(t, np.log(y)), 1)

        return f

    def calc_gradients(self, y, t):
        # 라벨 값에 y 값의 역수를 곱해주면 됨.
        return -np.multiply(t, 1. / y)


class MeanSquaredError(LossFunction):
    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def _function(self, y, t):
        return 0.

    def calc_gradients(self, y, t):
        return 0.

############################################# Optimizer
# 최근 많이 사용되는 SGD, Adagrad, Adam, RMSProp ... 등이 존재.
# 그 중 가장 구현에 쉬운 SGD을 구현.
class GradientDescentOptimizer:
    # learning rate를 매개변수로 받음
    def __init__(self, lr):
        self.lr = lr

    # 레이어 별로 학습 할 거임. 따라서 layer를 매개변수로 전달 받음
    def apply_gradients(self, layer):
        for i in range(len(layer.trainable_variables)):
            layer.trainable_variables[i] = layer.trainable_variables[i] - self.lr * layer.gradients[i]
        layer.gradients = None

############################################# Model
# Dense, Convolution, ... 등의 레이어가 존재

class DenseLayer:
    def __init__(self, n_in,n_out, activation=None, name=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation()
        self.name = name
        self.trainable_variables = None #여기에 weight를 집어넣을 예정
        self.gradients = None # 백프로파게이션을 통해 일괄적으로 업데이트

        self._build()
    
    def _build(self):
        self.trainable_variables = [
            # 가우시안 분포를 따르는 난수를 생성해줌
            # 평균과 표준 편차를 넣어주고 뽑아낼 난수의 개수를 지정해야 함.
            np.random.normal(scale=np.sqrt(2. / (self.n_in + self.n_out)), size=(self.n_in, self.n_out)), # Xavier Init
            # Bias 생성. 초기값을 0으로 주는 것이 관례
            np.zeros((1, self.n_out))
        ]

    # dense layer에 대한 forward 방향 연산
    def __call__(self, x):
        # x = m(미니배치 크기) * d(x의 차원)
        self.x = x
        self.a = np.matmul(self.x, self.trainable_variables[0]) + self.trainable_variables[1] # weight + bias
        return self.activation(self.a)

    # 상위 레이어에서 온 그레디언트 값을 받아줘야 함. 백프로파게이션이니까
    def calc_gradients(self, g_high):
        # g_high = m x 10
        # G_act = m x 10 x 10
        # activation에 들어가는 입력값에 대한 그레디언트 계산
        g_act = self.activation.calc_gradients(self.a)

        # intuitive, but can be more simpler
        # 차원 맞춰줌
        if isinstance(self.activation, Softmax):
            g_high = np.expand_dims(g_high, 1)      # m x 1 x 10
            delta = np.multiply(g_high, g_act)      # delta = m x 10 x 10
            delta = np.sum(delta, 2)                # delta = m x 10
        else:
            # 시그모이드인 경우
            delta = np.multiply(g_high, g_act)

        
        self.g_W = np.matmul(self.x.T, delta)
        self.g_b = np.sum(delta, 0, keepdims=True)
        self.gradients = [self.g_W, self.g_b]
        
        # 그레디언트를 밑으로 전달해야 함.
        # 현재 레이어의 델타 값에 그 레이어의 Weight 값을 곱해서 전파를 시킴 (?)!
        
        return np.matmul(delta, self.trainable_variables[0].T)
        


# Layer를 쌓아서 MLP 모델 생성
class MLP:
    def __init__(self, loss_func, learning_rate):
        self.loss_func = loss_func()
        self.opt = GradientDescentOptimizer(learning_rate)
        self._build()

    
    def _build(self):
        self.layers = [
            # 데이터셋의 크기가 8*8임. 입력이 64, 출력이 100
            DenseLayer(64, 100, Sigmoid, name='layer1'),
            DenseLayer(100, 100, Sigmoid, name='layer2'),
            DenseLayer(100, 10, Softmax, name='layer3')
        ]

    def fit(self, x, t):  # Implement backpropagation algortihm / 직관적으로 구현.
        # 입력 값에 대해 모델의 출력값 계산
        y = self(x)

        # 출력값을 기준으로 loss 값을 계산
        loss = self.loss_func(y, t)

        # 위쪽부터 그레디언트를 계산해서 내려감
        g = self.loss_func.calc_gradients(y, t)
        # 위쪽부터 아래 방향으로 그레디언트를 전파
        for l in self.layers[::-1]:
            # 백워드 path의 정 반대 방향
            g = l.calc_gradients(g)

            # 모든 레이어의 그레디언트를 다 계산한다음

        # 모든 방향(?)에 대한 그레디언트를 일괄적으로 적용
        for l in self.layers:
            self.opt.apply_gradients(l)

        return np.mean(loss)


    # Forward 연산. 간단히 표현 가능.
    def __call__(self, x):
        z = x
        # l = layer
        for l in self.layers:
            z = l(z)

        return z

            
    # 입력 값과 라벨 값이 있어야 계산 가능
    def accuracy(self, x, t):
        y = self(x)
        acc = np.mean(np.equal(np.argmax(y, 1), np.argmax(t, 1)).astype(np.float32))
        return acc


def split(data):
    X = data.data / 16. # 값의 범위를 0~1 사이로 normalization
    t = data.target.reshape((-1, 1))

    # [One-Hot Encoding]
    # 0 -> [1, 0, 0 ,0, 0, 0, 0, 0, 0]
    # 1 -> [0, 1, 0 ,0, 0, 0, 0, 0, 0]
    # fit_transform의 매개변수로 전달된 t의 클래스를 유니크하게 뽑아냄.
    encoder = OneHotEncoder()

    # fit_transform은 fit와 transform 함수를 연달아 실행함.
    # fit은 모델을 학습하는 부분. transform은 feature의 모양을 바꾸는 함수.
    t = encoder.fit_transform(t).toarray()

    size_fold = int(len(X) / 5.)

    tr_X, val_X, te_X = X[:size_fold * 3], X[size_fold * 3: size_fold * 4], X[size_fold * 4:]
    tr_t, val_t, te_t = t[:size_fold * 3], t[size_fold * 3: size_fold * 4], t[size_fold * 4:]

    return tr_X, tr_t, val_X, val_t, te_X, te_t



def main():
    data = load_digits()
    tr_X, tr_t, val_X, val_t, te_X, te_t = split(data)

    # 인스턴스 생성
    mlp = MLP(loss_func=CrossEntropy, learning_rate=1e-2)

    # val_y = mlp(val_X)
    # print (val_y.shape)

    # plotting 코드
    fig, ax = plt.subplots(1, 2)
    losses , tr_accs, val_accs = [], [], []

    # 학습 시킴
    n_batch = 100 # 미니배치 크기 100
    n_epoch = 20
    for e in range(n_epoch):
        # 학습 데이터 셋의 크기를 배치 크기로 나누면 순회 몇 번인지 알 수 있음
        for i in range(int(np.ceil(len(tr_X) / n_batch))):
            mb_X, mb_t = tr_X[i * n_batch:(i+1) * n_batch], tr_t[i * n_batch:(i+1) * n_batch]

            loss = mlp.fit(mb_X, mb_t)
            tr_acc = mlp.accuracy(mb_X, mb_t)
            val_acc = mlp.accuracy(val_X, val_t)

            losses.append(loss)
            tr_accs.append(tr_acc)
            val_accs.append(val_acc)

            print ('- epoch: %d step: %d, loss: %f, tr acc: %f, val acc: %f' % (e, i, loss, tr_acc, val_acc))

            ax[0].cla()
            ax[1].cla()

            ax[0].plot(losses, c='r')
            ax[1].plot(tr_accs, c='b')
            ax[1].plot(val_accs, c='r')

            fig.canvas.draw()
            plt.pause(0.001)

        # 한 에폭이 끝날 때 마다 테스트 셋을 이용해서 성능 측정
        te_acc = mlp.accuracy(te_X, te_t)
        print('* end of epoch(%d), te acc: %f' % (e, te_acc))

    fig.show()
    plt.show()

if __name__ == "__main__":

    sys.exit(main())