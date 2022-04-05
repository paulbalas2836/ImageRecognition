import numpy
import pandas as pd
from keras.regularizers import l2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import np_utils


train_data = numpy.load('Train_Dataset/pear_peach_train_dataset.npy', allow_pickle=True)
train_label = numpy.load('Train_Dataset/pear_peach_label_train_dataset.npy', allow_pickle=True)
train_data = train_data.astype('float32')
train_data = train_data / 255.0
train_label = np_utils.to_categorical(train_label)
x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2, random_state=4)

class_num = train_label.shape[1]

# load json and create base model
json_file = open('Model/Peach_Pear_AI_Architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
base_model.load_weights("Model/Peach_Pear_AI_Weights.h5", by_name=True)

model = keras.Sequential()
model.add(base_model)

# set first 2 blocks of Conv2D layers to non-trainable + set all BatchNormalization layers non-trainable
for layer in base_model.layers[:12]:
    layer.trainable = False
base_model.get_layer('sequential').get_layer('batch_normalization_4').trainable = False
base_model.get_layer('sequential').get_layer('batch_normalization_5').trainable = False


model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=64, validation_data=(x_test, y_test),
                    epochs=25)

pd.DataFrame(history.history).plot()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(25)

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

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))

model.build()
model.summary()
