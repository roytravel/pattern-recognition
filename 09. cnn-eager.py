# tensorflow v2는 eager execution을 제공 = 한국어로는 즉시 실행.
# eager execution을 통해 자유도가 조금 더 높음 but 조금 더 느려짐.
# 기존 tf v1의 문제점 -> python syntax를 이용한 코딩 불가

import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class CNN:

    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        # self.model.compile()을 하지 않고 대신 loss, optimizer 등 전부 따로 사용 --> 코딩 자유도 높이기 위함
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # loss 함수에 정보를 넣어줌. logit = output without activation
        self.opt = tf.keras.optimizers.Adam(1e-3) # 1e-3 = Adam의 default learning rate
        self.model.build()
        self.model.summary()


    def _build_layers(self):
        layers = [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, [3, 3], padding='same', strides=[1, 1], activation='relu'),
            # 커널의 개수 8개, 커널의 크기 = [3 x 3], padding=same은 28x28이 컨볼루션을 거쳐도 28x28이 됨 / (28, 28, 8)
            tf.keras.layers.MaxPool2D([2, 2], [2, 2]),
            # 커널의 크기 보통 2x2, stride도 보통 2x2. 이유는 커널이 이동하는데 겹치는 영역이 없도록 하기 위함. / (14, 14, 8)
            tf.keras.layers.Conv2D(4, [3, 3], padding='same', strides=[1, 1], activation='relu'),  # (14, 14, 4)
            tf.keras.layers.MaxPool2D([2, 2], [2, 2]),  # (7, 7, 4)
            tf.keras.layers.Flatten(),  # (7 * 7  * 4)
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10)
        ]

        return layers

    @tf.function # gpu 연산을 하란 의미로 데코레이터를 붙임. 연산 그래프에 병합이 됨
    def __call__(self, X, training=False): # 즉시 즉시 호출될 때 마다 연산 되기에 다소 느림
        return self.model(X, training=training)

    @tf.function
    def step(self, X, t):
        # for epoch
        #   for update
        #       step()
        # 미니 배치하나에 대해 모델을 업데이트하기 위해 사용되는 함수
        # y_pred = model(x)
        # loss = Loss(y_true, y_pred)
        # grad = d loss / d W
        # opt.apply(grad)위의 과정을 그대로 구현하면 아래와 같음

        with tf.GradientTape() as g:
            y = self(X, training=True) # prediction 값 계산. 학습이니 training=True
            loss = self.loss(t, y)

            vars = self.model.trainable_variables # trainable 파라미터 전부 추출
            grad = g.gradient(loss, vars) # loss에 대해 그레디언트 게산
            self.opt.apply_gradients(zip(grad, vars))

            return loss

    @tf.function
    def evaluate(self, X, t):  # 성능 측정을 위해 데이터 X, 라벨 t
        y = self(X, training=False) # __call__로 구현했기에 self() 형태.
        return tf.reduce_mean(
            tf.keras.metrics.sparse_categorical_accuracy(t, y))  # t = true, y = prediction value




def main():
    (tr_X, tr_t), (te_X, te_t) = tf.keras.datasets.mnist.load_data()
    tr_X, te_x = (tr_X / 255.).reshape((-1, 28, 28, 1)), (te_X / 255.).reshape((-1, 28, 28, 1))
    tr_X, tr_t, val_X, val_t = tr_X[:50000], tr_t[:50000], tr_X[50000:], tr_t[50000:]

    n_epoch = 3
    n_batch = 128

    cnn = CNN()

    fig, ax = plt.subplots(1, 2)
    tr_accuracy, val_accuracy, losses = [], [], []
    for n in range(n_epoch): # epoch을 도는 루프

        e_tr_acc, e_loss = [], []
        for i in range(int(len(tr_X) / n_batch)): # update하는 루프
            mb_X, mb_t = tr_X[i * n_batch:(i+1) * n_batch], tr_t[i * n_batch:(i+1) * n_batch] # iteration 별 배치를 가져올 수 있음
            loss = cnn.step(mb_X, mb_t).numpy() # step을 호출해서 미니 배치를 한 스텝 학습 가능. return 값은 eager~ 형태이나 numpy 붙여주면 넘파이 형태로 변경

            tr_acc = cnn.evaluate(mb_X, mb_t).numpy() # epoch 별 계산을 위함
            e_tr_acc.append(tr_acc)
            e_loss.append(loss)

            if i % 100 == 0:
                print (f"[-] epochs: %d, itr: %d, loss: %f, acc: %f" % (n, i, np.mean(e_loss), np.mean(e_tr_acc)))

        # 하나의 epoch이 끝나면 validation epoch을 측정 가능
        val_acc = cnn.evaluate(val_X, val_t)
        print ('[*] epochs: %d, val acc: %f' % (n, val_acc))
        
        # 위 까지만 돌려도 상관 없으나 모니터링 작업을 위해 fig, ax를 추가
        # epoch이 끝날 때 마다 기록
        tr_accuracy.append(np.mean(e_tr_acc))
        val_accuracy.append(val_acc)
        losses.append(np.mean(e_loss))

        # 플롯이 그려질 부분을 클리닝
        ax[0].cla()
        ax[1].cla()

        ax[0].plot(tr_accuracy, c='b', label='train acc')
        ax[0].plot(val_accuracy, c='r', label='val acc')
        ax[1].plot(losses, c='r', label='loss')
        ax[0].legend()
        ax[1].legend()
        fig.canvas.draw()
        plt.pause(0.1)

    # 모든 학습이 끝나고 test dataset을 이용하여 test accuracy 측정.
    te_acc = cnn.evaluate(te_X, te_t).numpy()
    print ('[*] te acc: %f' % te_acc)

    fig.show()
    plt.show()



if __name__ == "__main__":
    sys.exit(main())