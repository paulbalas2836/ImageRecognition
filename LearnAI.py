import numpy
from keras.regularizers import l2
from tensorflow import keras
from keras.utils import np_utils
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

seed = 21
n_split = 20
train_data = numpy.load('data_set.npy', allow_pickle=True)
train_label = numpy.load('data_label.npy', allow_pickle=True)
train_data = train_data.astype('float32')
train_data = train_data / 255.0
train_label = np_utils.to_categorical(train_label)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=4)

class_num = train_label.shape[1]

img_height = 64
img_width = 64
epochs = 25
batch_size = 64
kernel_regularizer = l2(0.0005)
kernel_initializer = "he_normal"
model = keras.Sequential()

model.add(keras.layers.Conv2D(16, 3, input_shape=(img_height, img_width, 3), activation='relu', padding='same',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(32, 3, input_shape=(img_height, img_width, 3), activation='relu', padding='same',
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer))
model.add(keras.layers.MaxPooling2D(2))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())

model.add(
    keras.layers.Dense(512, activation='relu', kernel_regularizer=kernel_regularizer,
                       kernel_initializer=kernel_initializer))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(class_num, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.build()
# print(model.summary())
numpy.random.seed(seed)
history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
                    epochs=epochs)

model.save('Model')

pd.DataFrame(history.history).plot()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
