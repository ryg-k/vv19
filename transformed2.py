import matplotlib
matplotlib.use("Agg")

import os
import keras
from keras.models import model_from_json, Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import LeakyReLU, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from skimage import io, transform
from sklearn.preprocessing import LabelBinarizer
import argparse
import cv2
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Reads the image, normalizes it and scales it down to input it into cnn
def image_transform(image):
    print("transforming image")
    img = io.imread(image)
    img = preprocess_input(img)
    img = img / 255.0
    img_size = 50
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')
    return img

def create_dataset(dirTrain):
    print("creating dataset")
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.0,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    for folder in os.listdir(dirTrain):

        if len([name for name in os.listdir(dirTrain+folder) if os.path.isfile(os.path.join(dirTrain+folder, name))]) >= 15:
            max_i = 25
        else:
            max_i = 40

        for image in os.listdir(dirTrain + folder):

            img = dirTrain + folder + image
            img = load_img(img)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for _ in datagen.flow(x, batch_size=1, save_to_dir=(dirTrain + folder), save_prefix='gen',
                                  save_format='jpg'):
                i += 1
                if i > max_i:
                    break


# Loads the dataset into arrays
def load_data():
    print("loading data")
    train_data = []
    val_data = []
    test_data = []
    train_labels = []
    val_labels = []
    test_labels = []
    labels = []
    classes = 0
    
    for folder in os.listdir(dirTrain):
        img_path = dirTrain + "/" + folder
        labels.append(os.path.basename(folder))
        classes += 1

        for image in os.listdir(img_path):
            img = image_transform(img_path + "/" + image)
            train_labels.append(classes - 1)
            train_data.append(img)
    
    classes = 0
    for folder in os.listdir(dirVal):
        img_path = dirTrain + "/" + folder
        labels.append(os.path.basename(folder))
        classes += 1

        for image in os.listdir(img_path):
            img = image_transform(img_path + "/" + image)
            val_labels.append(classes - 1)
            val_data.append(img) 

    index = 0
    for folder in os.listdir(dirTest):
        img_path = dirTest + "/" + folder
        index += 1

        for image in os.listdir(img_path):
            img = image_transform(img_path + "/" + image)
            test_labels.append(index - 1)
            test_data.append(img)
    
    np.savez('transformed_data.npz',train_d=np.array(train_data),val_d=np.array(val_data), test_d=np.array(test_data), train_l=np.array(train_labels), val_l=np.array(val_labels), test_l=np.array(test_labels))

    with open("label.csv","w") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerows(labels)

if __name__ == "__main__":
    dirTrain = "/home/virtual_net/dataset/train/train"
    dirVal = "/home/virtual_net/dataset/train/val"
    dirTest="/home/virtual_net/dataset/test"
    create_dataset(dirTrain)
    load_data()

