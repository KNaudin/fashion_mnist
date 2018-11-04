"""
Use base fashion_mnist
Need to reach 95% performance results
"""

# Actual best is 0.9461 for val_acc

import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, SpatialDropout2D, Dropout, LeakyReLU
from keras.optimizers import Adam, Adadelta, RMSprop
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

batch_size = 242
num_classes = 10
epochs = 500

# Data set of 60,000 28x28 grays cale images of 10 fashion categories, along with a test set of 10,000 images.
((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

data_gen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=.2,
    shear_range=.2
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

learner = Sequential()
# C2D-C2D-MP-Drop-C2D-C2D-MP-Drop-F-D-Drop-D
learner.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
learner.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(3, 3)))
learner.add(SpatialDropout2D(0.2))
learner.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
learner.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(3, 3)))
learner.add(SpatialDropout2D(0.2))
learner.add(Flatten())
learner.add(Dense(32, activation='relu'))
learner.add(Dropout(0.4))

# C2D-C2D-MP-C2D-MP-C2D-C2D-MP-D-Drop-F-D
'''learner.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
learner.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(2, 2)))
learner.add(SpatialDropout2D(0.5))
learner.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(2, 2)))
learner.add(SpatialDropout2D(0.22))
learner.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
learner.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(2, 2)))
learner.add(Dense(64, activation='linear'))
learner.add(SpatialDropout2D(0.15))
learner.add(Flatten())'''

learner.add(Dense(10, activation='softmax'))

learner.summary()

learner.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(amsgrad=True), metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='fashion_learner.hdf5', verbose=1, save_best_only=True, monitor='val_acc')

for i in range(0, 20):
    learner.fit(x_train, y_train, batch_size=batch_size, epochs=25, shuffle=True, verbose=1, validation_data=(x_test, y_test)
                , callbacks=[checkpoint, reduce_lr])

    learner.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size,
                          epochs=25, shuffle=True, verbose=1,  validation_data=(x_test, y_test), callbacks=[checkpoint, reduce_lr])
