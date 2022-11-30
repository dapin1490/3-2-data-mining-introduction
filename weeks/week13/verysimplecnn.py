import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import time

"""
# 캐글 버전 import

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
import numpy as np
import time
"""

res = open(r"weeks\week13\res\verysimplecnn_result.txt", 'w')

print(f"verysimplecnn.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}")
res.write(f"verysimplecnn.py 실행 시간 : {time.strftime('%Y-%m-%d %X', time.localtime(time.time()))}\n\n")

res.write("cnn 실습\n\nmnist 손글씨 데이터 사용\n\n")

batch_size = 128
num_classes = 10
epochs = 12

res.write(f"batch_size = {batch_size}\nnum_classes = {num_classes}\nepochs = {epochs}\n")

# input image dimensions
img_rows, img_cols = 28, 28

res.write(f"image size : ({img_rows}, {img_cols})\n\n")

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

res.write(f'x_train shape : {x_train.shape}\n')
res.write(f"{x_train.shape[0]} train samples\n")
res.write(f"{x_test.shape[0]} test samples\n\n")

# convert class vectors to binary class matrices
# 원 핫 인코딩
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 시퀀셜 형태로 계층 연결
model = Sequential()
# 2D 컨볼루션 2번, 맥스 풀링, 드롭아웃, 플래튼(2D -> 1D), 은닉층 구성, 드롭아웃, 결과값 출력 위한 은닉층 구성
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

res.write(f"model summary\n{model.summary()}\n\n")

# loss optimizer 컴파일
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# 모델 훈련, 평가
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=1)

print(f"Test loss : {score[0] :.4f}")
print(f"Test accuracy : {score[1] :.4f}")

res.write(f"Test loss : {score[0] :.4f}\n")
res.write(f"Test accuracy : {score[1] :.4f}\n")

res.close()
