import numpy
from keras.regularizers import l2
from tensorflow import keras
from keras.utils import np_utils

model = keras.Sequential()

# load json and create base model
json_file = open('modelNoTop.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
base_model.load_weights("modelWeights.h5")

# inputs = keras.Input(shape=(150, 150, 3))
model.add(base_model.layers.pop())
x = keras.Input(shape=(64,))(base_model)
y = keras.Dense(16, activation='softmax')(x)
model = keras.Model(x, y)
model.build()
model.summary()
