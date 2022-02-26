import numpy as np
import cv2
import os
import glob
from tqdm import tqdm


class CreateTestSet:
    def __init__(self):
        self.FruitImage_dict = {"Apple": 0, "Banana": 1, "Kiwi": 2, "Orange": 3}
        self.dataset = list()
        self.labels = list()
        self.height = 64
        self.width = 64

    def create_testSet(self):
        path = 'TestData/'
        for folderName in os.listdir(path):
            images = glob.glob(path + folderName + "/*")
            for name in tqdm(images):
                image = cv2.imread(name)
                image = cv2.resize(image, (self.height, self.width))
                self.dataset.append(image)
                self.labels.append(self.FruitImage_dict[folderName])

        numpyTestSet = np.array(self.dataset)
        numpyLabelSet = np.array(self.labels)
        p = np.random.permutation(len(self.dataset))
        np.save('test_set', numpyTestSet[p])
        np.save('test_label', numpyLabelSet[p])


dataset = CreateTestSet()
dataset.create_testSet()
