# coding: utf-8
# 시스템 접근
import os
import sys
# 텐서플로우 디버깅 메시지 Off.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 행렬 계산
import numpy as np
import pandas as pd
# 이미지 처리
import cv2
from PIL import Image
import matplotlib.pyplot as plt
# 모델 구현
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, Dense, LeakyReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

class SynDetectModel(object):
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layer())
        self.model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
        self.model.summary()
        self.model_checkpoint = ModelCheckpoint('syn_best_model.h5', monitor='loss', mode='min', save_best_only=True)


    def augment_image(self):
        train_path = 'train/'
        test_path = 'test/'

        train_generator = ImageDataGenerator(horizontal_flip=True, vertical_flip=False, shear_range=0.2, zoom_range=0.2, rescale=1 / 255)
        test_generator = ImageDataGenerator(rescale=1 / 255, validation_split=0.2)

        train_set = train_generator.flow_from_directory(train_path, batch_size=16, class_mode='binary', target_size=(300, 300), color_mode='rgb', shuffle=False)
        valid_set = test_generator.flow_from_directory(test_path, batch_size=16, class_mode='binary', target_size=(300, 300), color_mode='rgb', subset='training')
        test_set = test_generator.flow_from_directory(test_path, batch_size=16, class_mode='binary', target_size=(300, 300), color_mode='rgb', subset='validation')

        return train_set, valid_set, test_set

    def _build_layer(self):
        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(300, 300, 3), activation='relu'),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), padding='same', activation='relu'),
            Conv2D(64, (3, 3), padding='same', activation='relu'),

            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=1, activation='sigmoid')]

        return layers

    def loss(self, y_true, y_pred):  # 텐서플로우에서 제공해주는 형태를 위해 y_true, y_pred 추가
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(tf.one_hot(y_true, depth=1, dtype=tf.float32), 1)  # (N, 2) N = 배치 크기, 2 = 라벨 개수
        y_pred = tf.nn.sigmoid(y_pred, 1)
        # 메뉴얼 Cross Entropy 구현
        return -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_true, tf.math.log(y_pred)), 1))


    def accuracy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.squeeze(tf.one_hot(y_true, depth=2, dtype=tf.float32), 1)  # (N, 2)

        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, 1), tf.argmax(y_pred, 1)), tf.float32))

    def is_model_saved(self):
        try:
            self.model = load_model('./hidden/syn_detect_model.h5')
            return True

        except Exception as ex:
            return False


    def fit(self, train_set, valid_set, n_epochs, early_stop, flag):
        if flag == False:
            print ('[*] Start fitting the model')
            history = self.model.fit(train_set, validation_data=valid_set, epochs=n_epochs, callbacks=[early_stop, self.model_checkpoint], verbose=1)
            self.model.save('./hidden/syn_detect_model.h5')
            return history
        else:
            print ('[*] Skip fitting the model')
            return None


    def evaluate(self, valid_set):
        print ('[*] Start evaluating the model')
        return self.model.evaluate(valid_set, verbose=1)


    def predict(self, test_set):
        print ('[*] Start prediction')
        return self.model.predict(test_set, verbose=1)

    def test_predict(self):
        print ('[*] Start inferencing')

        y_pred = list()
        fake_images = os.listdir('./test/training_fake/')
        for i in fake_images:
            test_image = image.load_img('./test/training_fake/' + i, target_size=(300, 300, 3))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0) / 255.
            predict_result = self.model.predict_classes(test_image)
            #predict_result2 = np.argmax(self.model.predict(test_image), axis=-1)
            if predict_result[0][0] == 0:
                print("[*] Fake : {}".format(i))
                y_pred.append(predict_result[0][0])

            elif predict_result[0][0] == 1:
                print("[*] Real : {}".format(i))
                y_pred.append(predict_result[0][0])

        y_true = [0 for i in range(len(fake_images))]
        print (f"[*] F1 Score: {f1_score(y_true, y_pred, average='micro')}")


class LocDetectModel(object):
    def __init__(self):
        self.model = tf.keras.Sequential(self._build_layer())
        self.model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
        self.model.summary()
        self.model_checkpoint = ModelCheckpoint('loc_best_model.h5', monitor='loss', mode='min', save_best_only=True)


    def create_df(self):

        train_path = 'train/'
        test_path = 'test/'

        # Train dataframe
        train_frames = list()
        fake_fnames = os.listdir(train_path + 'training_fake/')
        for i in range(len(fake_fnames)):
            train_frame = list()
            fake_fname_label = fake_fnames[i].split('.')[0].split('_')[-1] # ex) 1010, 0011, 1111, 0111 ...
            train_frame.append(fake_fnames[i])
            train_frame.append(fake_fname_label[0])
            train_frame.append(fake_fname_label[1])
            train_frame.append(fake_fname_label[2])
            train_frame.append(fake_fname_label[3])
            train_frames.append(train_frame)


        # Test dataframe
        test_frames = list()
        fake_fnames = os.listdir(test_path + 'training_fake/')
        for i in range(len(fake_fnames)):
            test_frame = list()
            fake_fname_label = fake_fnames[i].split('.')[0].split('_')[-1]
            test_frame.append(fake_fnames[i])
            test_frame.append(fake_fname_label[0])
            test_frame.append(fake_fname_label[1])
            test_frame.append(fake_fname_label[2])
            test_frame.append(fake_fname_label[3])
            test_frames.append(test_frame)

        train_df = pd.DataFrame(train_frames, columns=['filename', 'left_eye', 'right_eye', 'nose', 'mouth'])
        test_df = pd.DataFrame(test_frames, columns=['filename', 'left_eye', 'right_eye', 'nose', 'mouth'])

        train_df['label'] = train_df['left_eye'] + train_df['right_eye'] + train_df['nose'] + train_df['mouth']
        test_df['label'] = test_df['left_eye'] + test_df['right_eye'] + test_df['nose'] + test_df['mouth']

        return train_df, test_df


    def augment_image(self, train_df, test_df):
        train_gen = ImageDataGenerator(# featurewise_center = True, featurewise_std_normalization = True,
            # width_shift_range=0.2, height_shift_range=0.2, # rescale = 1./255., rotation_range = 15,
            # vertical_flip = False, fill_mode='nearest'
            shear_range=0.2, zoom_range=0.2)

        test_gen = ImageDataGenerator(validation_split=0.2)

        train_set = train_gen.flow_from_dataframe(dataframe=train_df,
                                                  directory="./train/training_fake/", x_cols="filename",
                                                  y_col="label",
                                                  class_mode="categorical",
                                                  target_size=(300, 300),
                                                  batch_size=16,
                                                  save_to_dir='./hidden/preview',
                                                  save_format='jpeg')

        valid_set = test_gen.flow_from_dataframe(dataframe=test_df,
                                                 directory="./test/training_fake/", x_cols="filename",
                                                 y_col="label",
                                                 class_mode="categorical",
                                                 target_size=(300, 300),
                                                 batch_size=16,
                                                 subset='training',
                                                 save_to_dir='./hidden/preview',
                                                 save_format='jpeg')

        test_set = test_gen.flow_from_dataframe(dataframe=test_df,
                                                class_mode="categorical",
                                                directory="./test/training_fake/", x_cols="filename",
                                                y_col="label",
                                                subset="validation",
                                                batch_size=1)
        #print(test_set.class_indices)  # 라벨 확인
        return train_set, valid_set, test_set


    def _build_layer(self):

        layers = [
            Conv2D(32, (3, 3), padding='same', input_shape=(300, 300, 3), activation='relu'),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),

            Conv2D(64, (3, 3), padding='same', activation='relu'),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPool2D(pool_size=(2, 2)),
            BatchNormalization(),

            Flatten(),
            Dense(units=128, activation='relu'),
            Dense(units=15, activation='softmax')]

        return layers

    def loss(self):
        pass

    def is_model_saved(self):
        try:
            self.model = load_model('./hidden/loc_detect_model.h5')
            return True

        except Exception as ex:
            return False


    def fit(self, train_set, valid_set, n_epochs, early_stop, flag):
        if flag == False:
            print('[*] Start fitting the model')
            history = self.model.fit(train_set, validation_data=valid_set, epochs=n_epochs, callbacks=[early_stop, self.model_checkpoint], verbose=1)
            self.model.save('./hidden/loc_detect_model.h5')
            return history
        else:
            self.model = load_model('./hidden/loc_detect_model.h5')
            return self.model

    def accuracy(self):
        pass

    def evaluate(self, valid_set):
        print('[*] Start evaluating the model')
        return self.model.evaluate(valid_set, verbose=1)


    def predict(self, test_set):
        print ('[*] Start prediction')
        return self.model.predict(test_set, verbose=1)


    def test_predict(self, test_set):
        print('[*] Start inferencing')
        print(test_set.class_indices)

        y_pred = list()
        y_true = list()
        fake_images = os.listdir('./test/training_fake/')
        for i in fake_images:
            img = image.load_img('./test/training_fake/' + i, target_size=(300, 300, 3))
            img = image.img_to_array(img)
            img = img.reshape(-1, 300, 300, 3)

            predict_result = self.model.predict_classes(img, verbose=0)

            fname = i.split('.')[0].split('_')[-1]
            real_label = test_set.class_indices[fname]
            y_true.append(real_label)
            y_pred.append(predict_result[0])
            print(i, real_label, predict_result[0])
            # if real_label == predict_result[0]:
            #     y_pred.append(real_label)
            #     # print (i+ ": True")
            # else:
            #     y_pred.append(real_label)
            #     # print (i+ ": False")

        #y_true = [0 for i in range(len(fake_images))]
        print (round(f1_score(y_true, y_pred, average='micro'), 4))

        #predict = np.argmax(locate_model.predict(img2, verbose=1), axis=1)
        #print(predict)


def plot(epochs, history, flag):
    if flag:
        xc = range(epochs)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.figure(1, figsize=(7, 5))
        plt.plot(xc, train_loss)
        plt.plot(xc, val_loss)
        plt.xlabel('Num of Epochs')
        plt.ylabel('Loss')
        plt.title('Train loss vs Validation loss')
        plt.grid(True)
        plt.legend(['train', 'val'])
        plt.style.use(['classic']) # print plt.style.available # use bmh, classic,ggplot for big pictures

        plt.figure(2, figsize=(7, 5))
        plt.plot(xc, train_acc)
        plt.plot(xc, val_acc)
        plt.xlabel('Num of Epochs')
        plt.ylabel('Accuracy')
        plt.title('Train accuracy vs Validation accuracy')
        plt.grid(True)
        plt.legend(['Train', 'Validation'], loc=4)
        plt.style.use(['classic'])
    plt.show()


def main():

    S = SynDetectModel()
    L = LocDetectModel()

    epochs = 100
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    # Synthesis Detection --> True/False
    train_set, valid_set, test_set = S.augment_image()
    flag = S.is_model_saved()
    history = S.fit(train_set, valid_set, epochs, early_stop, flag)

    if flag:
        plot(early_stop.stopped_epoch-1, history, True)

    evaluation = S.evaluate(valid_set) # 모델 평가 (loss, accuracy)
    print(f'  \n[+] Loss : {round(evaluation[0],4)}\n  [+] Accuracy : {round(evaluation[1], 4)}\n  [+] Precision : {round(evaluation[2], 4)}\n  [+] Recall : {round(evaluation[3], 4)}\n  [+] AUC : {round(evaluation[4], 4)}')
    predict = S.predict(test_set) # 모델 예측 (test_set 전체 결과 도출)
    S.test_predict()

    # Location Detection --> Left eye, Right eye, Nose, Mouth
    train_df, test_df = L.create_df()
    train_set, valid_set, test_set = L.augment_image(train_df, test_df)
    flag = L.is_model_saved()
    history2 = L.fit(train_set, valid_set, epochs, early_stop, flag)

    if history2 is not None:
        plot(early_stop.stopped_epoch-1, history2, True)
    evaluation2 = L.evaluate(valid_set)
    print(f'[*] Loss : {round(evaluation2[0], 4)} Accuracy : {round(evaluation2[1], 4)}')
    #predict2 = L.predict(test_set)
    L.test_predict(test_set)


    # # Augmentation
    # # 피처, 라벨 분리
    #y_train = np.array(train_df.drop(['filename'], axis=1))
    #y_test = np.array(test_df.drop(['filename'], axis=1))



if __name__ == "__main__":
    main()