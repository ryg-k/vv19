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
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')
    return img

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

    dirTrain = "/home/virtual_net/dataset/train/train"
    dirVal = "/home/virtual_net/dataset/train/val"
    dirTest="/home/virtual_net/dataset/test"
    
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
        writer.writerows(list)

if __name__ == "__main__":
    load_data()

