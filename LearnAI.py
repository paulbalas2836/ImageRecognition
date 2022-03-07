import numpy
import pandas as pd
from keras.regularizers import l2
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

train_data = numpy.load('data_set.npy', allow_pickle=True)
train_label = numpy.load('data_label.npy', allow_pickle=True)
train_data = train_data.astype('float32')
train_data = train_data / 255.0
train_label = np_utils.to_categorical(train_label)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=4)

class_num = train_label.shape[1]


def create_model(save, summary, kernel_regularizer=l2(0.0005),
                 kernel_initializer="he_normal",
                 img_height=64, img_width=64):
    seed = 21
    epochs = 25
    batch_size = 64

    model = keras.Sequential()
    # model.add(keras.layers.Conv2D(16, 3, input_shape=(img_height, img_width, 3), activation='relu', padding='same',
    #                               kernel_initializer=kernel_initializer,
    #                               kernel_regularizer=kernel_regularizer))
    # model.add(keras.layers.BatchNormalization())
    #
    # model.add(keras.layers.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
    #                               kernel_regularizer=kernel_regularizer))
    # model.add(keras.layers.MaxPooling2D(2))
    # model.add(keras.layers.Dropout(0.2))
    # model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Conv2D(32, 3,input_shape=(img_height, img_width, 3), activation='relu', padding='same',
    #                               kernel_initializer=kernel_initializer,
    #                               kernel_regularizer=kernel_regularizer))
    # model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, 3, input_shape=(img_height, img_width, 3), activation='relu', padding='same',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())

    # model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
    #                               kernel_regularizer=kernel_regularizer))
    # model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(128, 3, input_shape=(img_height, img_width, 3), activation='relu', padding='same',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization(name="BatchNormalization_3"))

    model.add(keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer))
    model.add(keras.layers.MaxPooling2D(2))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())

    # if freeze:
    #     model.trainable = False

    # if use_top:
    model.add(keras.layers.Flatten())

    model.add(
        keras.layers.Dense(512, activation='relu', kernel_regularizer=kernel_regularizer,
                           kernel_initializer=kernel_initializer))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(class_num, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], learning_rate=0.001)
    # numpy.random.seed(seed)
    # history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test),
    #                     epochs=epochs)
    #
    # pd.DataFrame(history.history).plot()
    # plt.show()
    #
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs_range = range(epochs)
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()
    #
    # scores = model.evaluate(x_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1] * 100))

    if save:
        model.build()
        model.summary()
        # serialize model to JSON
        model.build()
        model_json = model.to_json()
        with open("modelNoTop.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("modelWeights.h5")

    # Check the summary of the model
    if summary:
        model.build()
        model.summary()


create_model(save=True, summary=False)
