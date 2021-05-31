import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class RNN:
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layers())
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        
        # summary를 보려면 input shape가 있어야 하고, 한 번 모델이 build 되어야 함.
        # Build: 연산 그래프를 만드는 과정(C++ 형태로 표현됨) tensorflow는 여러 언어를 지원(js, python etc.). 여러 언어는 하이레벨임. 로우레벨인 C++이 백앤드임.
        # node(vetex) => operation을 의미(덧셈, 뺄셈, 곱하기 등을 노드로 표현), edge를 따라서 데이터가 흐름 (tensor라고함)
        # tensorboard를 통해 확인 가능

        self.model.summary()

    def _build_layers(self):
        # mnist (60000, 28, 28)
        #       (60000, n_timestep, 28) => (None, 28)

        layers = [
            # tf.keras.layers.SimpleRNN()
            # tf.keras.layers.GRU()

            tf.keras.layers.LSTM(64, input_shape=(None, 28)),
            # tf.keras.layers.LSTM(64, input_shape=(None, 28)), # Stacked LSTM Model
            # tf.keras.layers.LSTM(64, input_shape=(None, 28)),
            # tf.keras.layers.LSTM(64, input_shape=(None, 28)),
            tf.keras.layers.Dense(10, activation='softmax')
        ]

        return layers


    def fit(self, X, t, val_X, val_t, n_epoch, n_batch):
        # 리턴값 = history
        return self.model.fit(X, t, validation_data=(val_X, val_t), batch_size=n_batch, epochs=n_epoch)


    def evaluate(self, X, t):
        # Evaluate는 metric에 지정해준 "accuracy"를 기반으로 성능 측정.
        # loss, accuracy
        return self.model.evaluate(X, t)


def main():
    (tr_X, tr_t), (te_X, te_t) = tf.keras.datasets.mnist.load_data()
    tr_X, te_x = (tr_X / 255.)[:, :27], te_X / 255.
    # tr_X (60000, 27, 28), te_X (10000, 28, 28) -> timestep의 길이가 달라지고 그럼에도 불구하고 처리 가능
    tr_X, tr_t, val_X, val_t = tr_X[:50000], tr_t[:50000], tr_X[50000:], tr_t[50000:]


    # hidden/cell states (hidden state 값 = return sequences = True, cell state 값 = return_state = True)
    # LSTM, GRU -> -> -> -> 초기 등장 데이터가 끝 부분에 반영 안됨. 요새 RNN 계열 모델 사용하지 않음.
    # 요새는 Transformer 모델을 많이 사용함 (Attention is all you need). RNN과 다름. 가변 길이를 처리할 수 있도록 디자인함.
    # self-attention 모델을 사용함. sub-layer 느낌으로 들어가 있음.
    # 여기서 비롯된 것이 BERT, GPT1-3, ...
    # forget, input, output gates로 구성됨
    # 3가지 모두에 activation이 들어감. activation과 recurrent_activation 값의 디폴트는 각각 tanh, sigmoid
    lstm = tf.keras.layers.LSTM(64, activation='tanh',
                                recurrent_activation='sigmoid',
                                input_shape=(None, 28),
                                return_sequences=True,
                                return_state=True)
    tr_y, last_hidden, last_cell = lstm(tr_X[:10])
    print(tr_y.shape)
    print (last_hidden.shape)
    print (last_cell.shape)

    # rnn = RNN()
    # hist = rnn.fit(tr_X, tr_t, val_X, val_t, n_epoch=10, n_batch=128)
    #
    # print (rnn.evaluate(te_X, te_t)) # 리스트 형태로 반환됨 [loss, accuracy]
    #
    # # 학습 추이를 그래프로 표현
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(hist.history['accuracy'], c='b', label='train acc')
    # ax[0].plot(hist.history['val_accuracy'], c='r', label='val acc')
    # ax[1].plot(hist.history['loss'], c='r', label='loss')
    # ax[0].legend()
    # ax[1].legend()
    # plt.show()



if __name__ == "__main__":
    sys.exit(main())