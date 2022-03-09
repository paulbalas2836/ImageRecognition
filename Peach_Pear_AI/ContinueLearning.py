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

apple_kiwi_banana_orange_ai_path = 'D:/ImageRecognitionLicenta/Apple_Kiwi_Banana_Orange_AI/Model'

# load json and create base model
json_file = open(apple_kiwi_banana_orange_ai_path + '/Apple_Kiwi_Banana_Orange_AI_Architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
base_model.load_weights(apple_kiwi_banana_orange_ai_path + "/Apple_Kiwi_Banana_Orange_AI_Weights.h5")

x = base_model.get_layer('batch_normalization_2').output

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(512, activation='relu')(x)
x = keras.layers.Dropout(0.5, name="Dropout_new_model")(x)
x = keras.layers.BatchNormalization(name="BatchNormalization_new_model")(x)
predictions = keras.layers.Dense(class_num, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

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

# serialize model to JSON
model_json = model.to_json()
with open("Model/Peach_Pear_AI_Architecture.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("Model/Peach_Pear_AI_Weights.h5")