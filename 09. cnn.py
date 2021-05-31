import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CNN:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()


    def _build_layers(self):
        layers = [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(8, [3, 3], padding='same', strides=[1, 1], activation='relu'), # 커널의 개수 8개, 커널의 크기 = [3 x 3], padding=same은 28x28이 컨볼루션을 거쳐도 28x28이 됨 / (28, 28, 8)
            tf.keras.layers.MaxPool2D([2, 2], [2, 2]), # 커널의 크기 보통 2x2, stride도 보통 2x2 그 이유는 커널이 이동하는데 겹치는 영역이 없도록 하기 위함. / (14, 14, 8)
            tf.keras.layers.Conv2D(4, [3, 3], padding='same', strides=[1, 1], activation='relu'), # (14, 14, 4)
            tf.keras.layers.MaxPool2D([2, 2], [2, 2]), # (7, 7, 4)
            tf.keras.layers.Flatten(),              # (7 * 7  * 4)
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

        return layers


    def fit(self, X, t, val_X, val_t, n_epoch, n_batch):
        return self.model.fit(X, t, validation_data=(val_X, val_t), batch_size=n_batch, epochs=n_epoch)


    def evaluate(self, X, t): # 성능 측정을 위해 데이터 X, 라벨 t
        y = self.model(X, training=False)
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(t, y)) # t = true, y = prediction value


def main():
    (tr_X, tr_t), (te_X, te_t) = tf.keras.datasets.mnist.load_data()
    tr_X, te_x = (tr_X / 255.).reshape((-1, 28, 28, 1)), (te_X / 255.).reshape((-1, 28, 28, 1))
    tr_X, tr_t, val_X, val_t = tr_X[:50000], tr_t[:50000], tr_X[50000:], tr_t[50000:]

    cnn = CNN()
    hist = cnn.fit(tr_X, tr_t, val_X, val_t, n_epoch=10, n_batch=100)

    print (cnn.evaluate(te_X, te_t)) # 리스트 형태로 반환됨 [loss, accuracy]

    # 학습 추이를 그래프로 표현
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(hist.history['accuracy'], c='b', label='train acc')
    ax[0].plot(hist.history['val_accuracy'], c='r', label='val acc')
    ax[1].plot(hist.history['loss'], c='r', label='loss')
    ax[0].legend()
    ax[1].legend()
    plt.show()


if __name__ == "__main__":
    sys.exit(main())