import numpy
from tensorflow import keras
from keras.utils import np_utils

test_data = numpy.load('apple_kiwi_banana_orange_test_dataset.npy', allow_pickle=True)
test_label = numpy.load('apple_kiwi_banana_orange_label_test_dataset.npy', allow_pickle=True)
test_data = test_data.astype('float32')
test_data = test_data / 255.0
test_label = np_utils.to_categorical(test_label)
class_num = test_label.shape[1]

# load json and create base model
json_file = open('Model/Apple_Kiwi_Banana_Orange_AI_Architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)

model.load_weights("Model/Apple_Kiwi_Banana_Orange_AI_Weights.h5")

scores = model.evaluate(test_data, test_label, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
