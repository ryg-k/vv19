import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import History
from tensorflow.contrib.tpu.python.tpu import keras_support
import tensorflow.keras.backend as K

import pickle, os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_model():
    model = InceptionV3(include_top=False, weights="imagenet", input_shape=(160,160,3))
    x = GlobalAveragePooling2D()(model.layers[-1].output)
    x = Dense(101, activation="softmax")(x)

    # mixed4(132)から先を訓練する
    for i in range(133):
        model.layers[i].trainable = False

    return Model(model.inputs, x)

def train():
    model = create_model()
    model.compile(tf.train.RMSPropOptimizer(1e-4), "categorical_crossentropy", ["acc"])

    #tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    #tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    #strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    #model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size = 2048
    train_generator = ImageDataGenerator(rescale=1/255.0, horizontal_flip=True).flow_from_directory(
        "food2/train", (160,160), batch_size=batch_size, save_format="jpg")
    test_generator = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
        "food2/test", (160,160), batch_size=batch_size, save_format="jpg")

    hist = History()
    model.fit_generator(train_generator, steps_per_epoch=75750//batch_size,
                        validation_data=test_generator, validation_steps=25250//batch_size,
                        callbacks=[hist], epochs=100)
    history = hist.history

    with open("food_history.dat", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    K.clear_session()
    train()
