import numpy as np
import cv2
import os
import glob
from tqdm import tqdm


class CreateDataset:
    def __init__(self):
        self.FruitImage_dict = {"Peach": 0, "Pear": 1}
        self.dataset = list()
        self.labels = list()
        self.height = 64
        self.width = 64

    def create_dataset(self):
        path = 'Train_Data/'
        for folderName in os.listdir(path):
            images = glob.glob(path + folderName + "/*")
            for name in tqdm(images):
                image = cv2.imread(name)
                image = cv2.resize(image, (self.height, self.width))
                self.dataset.append(image)
                self.labels.append(self.FruitImage_dict[folderName])

        numpyDataSet = np.array(self.dataset)
        numpyLabelSet = np.array(self.labels)
        p = np.random.permutation(len(self.dataset))
        np.random.shuffle(self.dataset)
        np.save('Train_Dataset/peach_pear_train_dataset', numpyDataSet[p])
        np.save('Train_Dataset/peach_pear_label_train_dataset', numpyLabelSet[p])


dataset = CreateDataset()
dataset.create_dataset()
