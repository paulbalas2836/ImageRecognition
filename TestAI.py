import numpy
from tensorflow import keras
from keras.utils import np_utils

test_data = numpy.load('test_set.npy', allow_pickle=True)
test_label = numpy.load('test_label.npy', allow_pickle=True)
test_data = test_data.astype('float32')
test_data = test_data / 255.0
test_label = np_utils.to_categorical(test_label)
class_num = test_label.shape[1]

reconstructed_model = keras.models.load_model("Model")

scores = reconstructed_model.evaluate(test_data, test_label, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))
