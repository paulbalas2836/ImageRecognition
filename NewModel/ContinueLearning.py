import numpy
from keras.regularizers import l2
from tensorflow import keras
from keras.utils import np_utils

apple_kiwi_banana_orange_ai_path = 'D:/ImageRecognitionLicenta/Apple_Kiwi_Banana_Orange_AI/Model'

# load json and create base model
json_file = open(apple_kiwi_banana_orange_ai_path + '/Apple_Kiwi_Banana_Orange_AI_Architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
base_model = keras.models.model_from_json(loaded_model_json)

# load weights into new model
base_model.load_weights(apple_kiwi_banana_orange_ai_path + "/Apple_Kiwi_Banana_Orange_AI_Weights.h5")

x = base_model.get_layer('batch_normalization_2').output
x = keras.layers.GlobalAveragePooling2D()(x)

x = keras.layers.Dense(256, activation='relu')(x)

predictions = keras.layers.Dense(2, activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.summary()
