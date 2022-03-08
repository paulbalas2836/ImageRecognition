from random import randint
import tensorflow as tf
import cv2
import os
import glob
import numpy as np
import tensorflow_addons as tfa
from tqdm import tqdm


def image_augmentation():
    path_apple_kiwi_banana_orange_dataset = 'Apple_Kiwi_Banana_Orange_AI/Train_Data/'
    j = 0
    for folder_name in os.listdir(path_apple_kiwi_banana_orange_dataset):
        images = glob.glob(path_apple_kiwi_banana_orange_dataset + folder_name + "/*")
        i = 0
        for name in tqdm(images):

            image = cv2.imread(name)
            height, width = image.shape[:2]

            for k in range(20):
                array = np.array(image)

                # Flipped Image
                newImage = image_flip(array)

                # Rotated Image
                if randint(0, 1) == 0:
                    newImage = image_rotation(array)

                # Traslated Image
                if randint(0, 1) == 0:
                    newImage = image_translate(newImage, height, width)

                # Gaussian Noise
                if randint(0, 2) == 0:
                    newImage = image_noise(newImage)

                cv2.imwrite(path_apple_kiwi_banana_orange_dataset + folder_name + "/image " + str(j) + str(i) + str(k) + ".png", newImage)
            i = i + 1
        j = j + 1


def image_flip(array):
    option = randint(0, 3)
    if option == 0:
        image = np.fliplr(array)
    elif option == 1:
        image = np.flipud(array)
    elif option == 2:
        image = np.flip(array, (1, 0))
    else:
        return array
    return image


def image_rotation(array):
    image = tfa.image.rotate(array, angles=randint(0, 360))
    return image.numpy()


def image_noise(array):
    noise = tf.random.normal(shape=tf.shape(array), mean=0.0, stddev=1.0,
                             dtype=tf.float32).numpy()

    image = tf.add(array, noise)
    return image.numpy()


def image_translate(array, height, width):
    image = tfa.image.translate(images=array,
                                translations=[randint(int(-width / 5), int(width / 5)), randint(int(-height / 5), int(height / 5))])
    return image.numpy()


image_augmentation()
