from random import randint

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm


def image_augmentation():
    pathDataSet = 'Images/'
    j = 0
    for folderName in os.listdir(pathDataSet):
        images = glob.glob(pathDataSet + folderName + "/*")
        i = 0
        for name in tqdm(images):

            image = cv2.imread(name)
            image = cv2.resize(image, (64, 64))
            height, width = image.shape[:2]

            for k in range(10):
                newImage = image
                color = randint(0, 1)
                # Change color
                if color == 0:
                    newImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    newImage = newImage

                if randint(0, 2) == 0:
                    tx = [1, 0, randint(-16, 16)]
                    ty = [0, 1, randint(-16, 16)]
                    M = np.float32([tx, ty])
                    newImage = cv2.warpAffine(newImage, M, (width, height))

                # Flipped image
                if randint(0, 1) == 0:
                    newImage = cv2.flip(newImage, randint(-1, 1))

                if randint(0, 1) == 0:
                    newImage = newImage[randint(0, height - 30):height, randint(0, width - 30):width]

                cv2.imwrite(pathDataSet + folderName + "/image " + str(j) + str(i) + str(k) + ".png", newImage)
            i = i + 1
        j = j + 1


image_augmentation()
