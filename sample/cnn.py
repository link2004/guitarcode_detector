import numpy as np
#データの読み込みと前処理
#from keras.utils import np_utils
from keras.utils import to_categorical
#kerasでCNN構築
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pandas as pd
import time
#CGFAmEmDm
f_model = './model'

#時間計測
import time
correct = 10
data = pd.read_csv("merged.csv")
X = data.iloc[:, 0:1024]
Y = data.iloc[:, 1024]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
X_train = X_train.to_numpy()
# X_trainの形状を取得
num_samples, num_features = X_train.shape
# 画像のサイズとチャネル数を指定
image_height = 32
image_width = 32
num_channels = 1
# 新しい形状を計算
new_shape = (num_samples, image_height, image_width, num_channels)
# 全要素数が一致することを確認
assert np.prod(X_train.shape) == np.prod(new_shape)
# reshapeを実行
X_train = X_train.reshape(new_shape)
X_train = X_train.astype('float32')

Y_train = Y_train.to_numpy()
Y_train = to_categorical(Y_train, correct)

X_test = X_test.to_numpy()
# X_testの形状を取得
num_samples_test, num_features_test = X_test.shape
# 新しい形状を計算
new_shape_test = (num_samples_test, image_height, image_width, num_channels)
# 全要素数が一致することを確認
assert np.prod(X_test.shape) == np.prod(new_shape_test)
# reshapeを実行
X_test = X_test.reshape(new_shape_test)
X_test = X_test.astype('float32')

Y_test = Y_test.to_numpy() 
Y_test = to_categorical(Y_test, correct)

model = Sequential()

model.add(Conv2D(filters=10, kernel_size=(3,3),padding='same', input_shape=(32,32,1), activation='relu'))
model.add(Conv2D(32,1,activation='relu'))
model.add(Conv2D(64,1,activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

startTime = time.time()

history = model.fit(X_train, Y_train, epochs=200, batch_size=100, verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test Loss:{0:.3f}'.format(score[0]))
print('Test accuracy:{0:.3}'.format(score[1]))
#処理時間
print("time:{0:.3f}sec".format(time.time() - startTime))

json_string = model.to_json()

model.save('model_CNN.h5')

