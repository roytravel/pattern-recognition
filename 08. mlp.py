import sys
import tensorflow as tf
import matplotlib.pyplot as plt


class MLP:
    def __init__(self):
        self.layers = self._build_layers()
        self.model = tf.keras.Sequential(self.layers) # Sequential 함수에서 레이어를 넣으면 순차적으로 연산해줌. feed forward.

        self.model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                           loss=self.loss, # 직접 구현해서 넣기 or 텐서플로우 내장 함수
                           metrics=[
                               self.accuracy # 직접 구현해서 넣기 or 텐서플로우 내장 함수
                           ])


    def _build_layers(self):
        layers = [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(10)
        ]

        return layers


    def loss(self, y_true, y_pred): # 텐서플로우에서 제공해주는 형태를 위해 y_true, y_pred 추가
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(tf.one_hot(y_true, depth=10, dtype=tf.float32), 1) # (N, 10) N = 배치 크기
        y_pred = tf.nn.softmax(y_pred, 1)

        # 메뉴얼 Cross Entropy 구현
        return -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, tf.math.log(y_pred)), 1))


    def accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(tf.one_hot(y_true, depth=10, dtype=tf.float32), 1) # (N, 10)

        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)), tf.float32))


    def fit(self, tr_X, tr_t, val_X, val_t, n_epoch, n_batch):

        # 아래의 return 값이 history 형태로 전달 됨
        # 모델이 학습하게 되면 여러 Iteration과 epoch을 수행함. 그 때 수행된 loss 값과 metric이 들어가게 됨.
        return self.model.fit(tr_X, tr_t,
                              validation_data=(val_X, val_t),
                              epochs=n_epoch,
                              batch_size=n_batch)

def main():
    # MNIST
    (tr_X, tr_t), (te_X, te_t) = tf.keras.datasets.mnist.load_data()

    # Normalization.
    tr_X, te_X = tr_X / 255., te_X / 255.

    # Validation을 학습 데이터셋에서 일부 분리
    tr_X, val_X, tr_t, val_t = tr_X[:50000], tr_X[50000:], tr_t[:50000], tr_t[50000:]


    model = MLP()
    hist = model.fit(tr_X, tr_t, val_X, val_t, n_epoch=10, n_batch=128)

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(hist.history['accuracy'], c='b', label='train acc')
    ax[0].plot(hist.history['val_accuracy'], c='r', label='val acc')
    ax[1].plot(hist.history['loss'], c='r', label='loss')
    ax[0].legend()
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    sys.exit(main())