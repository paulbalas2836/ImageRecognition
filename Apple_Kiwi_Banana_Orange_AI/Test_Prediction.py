import glob
import os

import cv2
import numpy
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.utils import np_utils
from tqdm import tqdm

# path = 'Test_Data/'
# dataset = list()
# for folderName in os.listdir(path):
#     images = glob.glob(path + folderName + "/*")
#     for name in tqdm(images):
#         image = cv2.imread(name)
#         image = cv2.resize(image, (64, 64))
#         dataset.append(image)
#
# numpyDataSet = numpy.array(dataset)
# test_data = numpyDataSet.astype('float32')
# test_data = test_data / 255.0


train_data = numpy.load('apple_kiwi_banana_orange_train_dataset.npy', allow_pickle=True)
train_label = numpy.load('apple_kiwi_banana_orange_label_train_dataset.npy', allow_pickle=True)
train_data = train_data.astype('float32')
test_data = train_data / 255.0
apple_kiwi_banana_orange_ai_path = 'D:/ImageRecognitionLicenta/Apple_Kiwi_Banana_Orange_AI/Model'

# load json and create base model
json_file = open(apple_kiwi_banana_orange_ai_path + '/Apple_Kiwi_Banana_Orange_AI_Architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)

# # load weights into new model
model.load_weights(apple_kiwi_banana_orange_ai_path + "/Apple_Kiwi_Banana_Orange_AI_Weights.h5", by_name=True)

prediction = model.predict(test_data[:20])
print("predictions shape:", prediction, train_label[:20])

plt.imshow(test_data[0].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[1].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[2].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[3].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[4].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[5].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[6].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[7].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[8].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[9].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[10].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[11].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[12].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[13].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[14].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[15].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[16].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[17].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[18].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[19].reshape(64, 64, 3))
plt.show()
plt.imshow(test_data[20].reshape(64, 64, 3))

plt.show()
