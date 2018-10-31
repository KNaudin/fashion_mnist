"""
Use base fashion_mnist
Need to reach 95% performance results
"""

# Actual best is 0.9398 for val_acc

import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, SpatialDropout2D, Dropout
from keras.optimizers import Adam
from keras.datasets import fashion_mnist
from keras.callbacks import ModelCheckpoint
import numpy as np

batch_size = 190
num_classes = 10
epochs = 50

# Data set of 60,000 28x28 grays cale images of 10 fashion categories, along with a test set of 10,000 images.
((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

learner = Sequential()
learner.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
learner.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
learner.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
learner.add(MaxPooling2D(pool_size=(2, 2)))
learner.add(SpatialDropout2D(0.25))
learner.add(Flatten())
learner.add(Dense(32, activation='relu'))
learner.add(Dropout(0.05))
learner.add(Dense(num_classes, activation='softmax'))

learner.summary()

learner.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath='fashion_learner.hdf5', verbose=1, save_best_only=True, monitor='val_acc')

learner.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)
            , callbacks=[checkpoint])
score = learner.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
