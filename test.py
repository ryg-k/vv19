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



# CNN model structure
def cnn(classes):
    print("making model")

    #Load the pretrained model
    pre_trained = VGG19(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    #pre-trained = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_size, img_size, 3), pooling=None, classes=1000)
    #pre_trained = keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_size, img_size, 3), pooling=None, classes=1000)

    for layer in pre_trained.layers:
        layer.trainable = True

    model = Sequential()
    model.add(pre_trained)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy', 'top_k_categorical_accuracy'])

    print("done")
    return model

# Function for predicting images from test folder (saves images with top 5 labels)
def predict(data_test, labels):
    print("predicting")
    i = 0
    index = 0
    imgs = []
    imgnames = []
    value = [0, 0, 0]

    for folder in os.listdir(dirTest):
        img_path = dirTest + "/" + folder
        index += 1

        for image in os.listdir(img_path):
            img = cv2.imread(filename=img_path+"/"+image, flags=cv2.IMREAD_UNCHANGED)
            imgs.append(img)
            imgnames.append(os.path.basename(folder))

    for image, img in zip(data_test, imgs):

        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image)
        #print(predictions)
        predictions = predictions[0]
        top5 = np.array(predictions)
        top = top5.argsort()[-5:][::-1]

        img = cv2.copyMakeBorder(img, top=80, bottom=0, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=value)
        i += 1
        cv2.putText(img, labels[top[0]] + ": " + str(predictions[top[0]]),
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[1]] + ": " + str(predictions[top[1]]),
                    org=(5, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[2]] + ": " + str(predictions[top[2]]),
                    org=(5, 45),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[3]] + ": " + str(predictions[top[3]]),
                    org=(5, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        cv2.putText(img, labels[top[4]] + ": " + str(predictions[top[4]]),
                    org=(5, 75),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4,
                    color=(255, 255, 255),
                    lineType=2)

        if not os.path.exists(prediction + imgnames[i-1]):
            os.mkdir(prediction + imgnames[i-1])
        img_name = prediction + imgnames[i-1] +"/"+ str(i) + ".jpg"

        cv2.imwrite(img_name, img)

<<<<<<< HEAD
=======

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


>>>>>>> e818642efb4111dff6c5443034dde2cee6b19ee9
def print_cmx(y_true,y_pred):
    labels=sorted(list(set(y_true)))
    cmx_data=confusion_matrix(y_true,y_pred,labels=labels)
    #cmx_data=cmx_data.astype('float')/cmx_data.sum(axis=1)[:,np.newaxis]
    df_cmx=pd.DataFrame(cmx_data,index=labels,columns=labels)
    plt.figure(figsize=(30,21))
    sn.heatmap(df_cmx,annot=False)
    plt.savefig('cm.png')


if __name__ == "__main__":
    print("running")
    epochs_n = 100
    img_size = 50
    input_shape = (img_size, img_size, 3)

    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--save", help='path to save model')
    ap.add_argument("-l", "--load", help='path to load model')
    ap.add_argument("-p", "--plot", help='path to output accuracy/loss plot')
    ap.add_argument("-d", "--dataset", help='path to dataset')
    args = vars(ap.parse_args())

    prediction = "predictions"
    dirTrain = "/home/virtual_net/dataset/train/train"
    dirVal = "/home/virtual_net/dataset/train/val"
    dirTest="/home/virtual_net/dataset/test"
    transformed="/home/virtual_net/vv19/npzfile"		
    
    npzfile=np.load("transformed_data.npz")
    data_train=npzfile['train_d']
    data_val=npzfile['val_d']
    data_test=npzfile['test_d']
    labels_train=npzfile['train_l']
    labels_val=npzfile['val_l']
    labels_test=npzfile['test_l']
    num_classes=5
    
    labels=[]
    with open("lable.csv","r") as f:
        reader=csv.reader(f)
        header=next(reader)
    for row in reader:
        labels.append(row)

    encoder = LabelBinarizer()
    transformed_labels_train = encoder.fit_transform(labels_train)
    transformed_labels_val=encoder.fit_transform(labels_val)
    transformed_labels_test = encoder.fit_transform(labels_test)

    if args['save']:
        # Save model
        model = cnn(num_classes)
        #model.summary()

        earlystopping = EarlyStopping(monitor='val_acc',
                                      min_delta=0,
                                      patience=20,
                                      verbose=1,mode='auto' )
        fpath=fpath = 'weightsrm.hdf5'
        checkpointer = ModelCheckpoint(filepath=fpath,
                                       monitor='val_acc', 
                                       verbose=1,
                                       save_best_only=True, 
                                       mode='auto')
        
        history = model.fit(x=data_train,
                      y=transformed_labels_train,
                      epochs=epochs_n,
                      batch_size=128,
                      verbose=1,
                      validation_data=(data_val, transformed_labels_val),
                      #validation_split=0.2,
                      callbacks=[earlystopping, checkpointer])

        Y_pred=model.predict(data_val)
        y_pred=np.argmax(Y_pred,axis=1)
        Z_pred=model.predict(data_test)
        z_pred=np.argmax(Z_pred,axis=1)
        #print_cmx(labels_val,y_pred)
        print_cmx(labels_test,z_pred)
        
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('acc.png')

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig('loss.png')

        model_json = model.to_json()
        with open(args['save'] + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(args['save'] + ".h5")
        print("[INFO] model saved")

        (eval_loss, eval_accuracy, eval_top5_acc) = model.evaluate(x=data_val, y=transformed_labels_val, verbose=1)
        print("[INFO] val accuracy: {: .2f}%".format(eval_accuracy * 100))
        print("[INFO] top5 accuracy: {: .2f}%".format(eval_top5_acc * 100))


        (eval_loss, eval_accuracy, eval_top5_acc) = model.evaluate(x=data_test, y=transformed_labels_test, verbose=1)
        print("[INFO] test accuracy: {: .2f}%".format(eval_accuracy * 100))
        print("[INFO] top5 accuracy: {: .2f}%".format(eval_top5_acc * 100))

    if args['load']:

        # Load model
        json_file = open(args['load'] + ".json", 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(args['load'] + ".h5")
        print("[INFO] model wczytany")

    file_path = 'labels.txt'
    file = open(file_path, 'w')

    for label in labels:
        file.write(label + '\n')
    file.close()

    predict(data_test, labels)

