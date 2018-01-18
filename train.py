'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import random
import string
from captcha.image import ImageCaptcha

image = ImageCaptcha(width = 100,height = 30)
sample_str = string.digits + string.ascii_uppercase + string.ascii_lowercase
sample_str*=4
index_str = string.digits + string.ascii_uppercase + string.ascii_lowercase

batch_size = 128

epochs = 1

# input image dimensions
img_rows, img_cols = 30, 100
input_shape = (img_rows, img_cols, 1)


x_train = np.zeros([10000,img_rows,img_cols,1])
x_test = np.zeros([1000,img_rows,img_cols,1])
y_train = np.zeros([10000,62*4])
y_test = np.zeros([1000,62*4])

for i in range(10000):
	str = random.sample(sample_str,4)
	img = image.generate_image(str).convert('L')
	img = np.array(img.getdata())
	x_train[i] = np.reshape(img,[30,100,1])/255.0
	for j in range(4):
		y_train[i][index_str.find(str[j])+62*j]=1;
	
for i in range(1000):
	str = random.sample(sample_str,4)
	img = image.generate_image(str).convert('L')
	img = np.array(img.getdata())
	x_test[i] = np.reshape(img,[30,100,1])/255.0
	for j in range(4):
		y_test[i][index_str.find(str[j])+62*j]=1;


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(y_test[0])
print(y_train[0])

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(62*4))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
