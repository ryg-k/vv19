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

# Generates images similar to the ones in dataset so there is more training/testing data
def create_dataset(dirTrain, dirVal, dirTest):
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

    for folder in os.listdir(dirVal):

        if len([name for name in os.listdir(dirVal+folder) if os.path.isfile(os.path.join(dirVal+folder, name))]) >= 15:
            max_i = 25
        else:
            max_i = 40

        for image in os.listdir(dirVal + folder):

            img = dirVal + folder + image
            img = load_img(img)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for _ in datagen.flow(x, batch_size=1, save_to_dir=(dirVal + folder), save_prefix='gen',
                                  save_format='jpg'):
                i += 1
                if i > max_i:
                    break

    for folder in os.listdir(dirTest):

        if len([name for name in os.listdir(dirTest+folder) if os.path.isfile(os.path.join(dirTest+folder, name))]) >= 5:
            max_i = 25
        else:
            max_i = 40

        for image in os.listdir(dirTest + folder):

            img = dirTest + folder + image
            img = load_img(img)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0

            for _ in datagen.flow(x, batch_size=1, save_to_dir=(dirTest + folder), save_prefix='gen',
                                  save_format='jpg'):
                i += 1
                if i > max_i:
                    break


# Reads the image, normalizes it and scales it down to input it into cnn
def image_transform(image):
    print("transforming image")
    img = io.imread(image)
    img = preprocess_input(img)
    img = img / 255.0
    img = transform.resize(img, (img_size, img_size, 3), mode='constant')

    return img


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

        #only 4 categories are used in testing
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

    return np.array(train_data), np.array(val_data), np.array(test_data), np.array(train_labels), np.array(val_labels), np.array(test_labels), classes, labels

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
    # create_dataset(dirTrain, dirVal, dirTest)
    data_train, data_val, data_test, labels_train, labels_val, labels_test, num_classes, labels = load_data()

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

