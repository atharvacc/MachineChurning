

import tensorflow as tf
import pandas as pd 
import numpy as np
import skimage 
from gray_scale_converter.utils import list_files1, load_from_dir,imread_grayscale
import os
from datetime import datetime 
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPool2D, AveragePooling2D, Conv2DTranspose
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from skimage.io import imread, imsave
import matplotlib.pyplot as plt

class base_gray_scale_model:
    """
    base_gray_scale_model: The basic gray scale model to be used for inference. The learning rate and image_directory is provided by user
                Args: 
                    (String) train_dir: directory where the training images can be found
                    (int) lr          : learning rate
                    (String) test_dir : directory where the testing images can be found
                    (String) extension: extension of the images
                    (int)   batch_size: batch size to be used for training 
                    (int)   epochs    : number of epochs to train for 
            (String) model_save_dir   : directory where the model should be saved after training 
            (String) test_save_dir    : directory where we should save the results after running inference
            (String) model_path       : path to pretrained model in h5 format.

    """
    def __init__(self, lr , batch_size, epochs, model_save_dir,  train_dir = None,  test_dir = None, extension = "png", model_path = None, test_save_dir = "/gray_results" ):
        self.train_dir  = train_dir
        self.lr = lr
        self.test_dir = test_dir
        self.extension = extension
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_save_dir = model_save_dir
        self.test_save_dir = test_save_dir
        if (model_path == None):
            self.model = self.initialize_model()
        else:
            print("Using pretrained model\n")
            self.model = load_model(model_path)

    def initialize_model(self):
        """
        initialzie_model: initialize the model
                Args:
                    None    
                Returns:
                    model: new model
        """
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape = (512,512,1)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same', strides=2))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
        model.add(UpSampling2D((2, 2)))
        opt = Adam(learning_rate = self.lr)
        model.compile(optimizer=opt, loss='mae', metrics = ['mse'])
        return(model)

    def preprocess_data(self, directory, extension):
        """
        preprocess_data: 
                    Args:
                        directory: Directory that contain the images
                        extension: extension of the images
                    Returns:
                        X_HE: numpy array containing the training data
                        Y_HE: numpy array containing the output data (colored images)
        """
        data, data_gray = load_from_dir(directory, extension)
        m,n,o = data_gray.shape
        X_HE = data_gray.reshape(m,n,o,1)
        Y_HE = data/255
        return (X_HE, Y_HE)

    def train(self):
        X_train, y_train = self.preprocess_data(self.train_dir, self.extension)
        print("Shape of training data is : {}".format(X_train.shape))
        print("Shape of testing data is {}".format(y_train.shape))
        validation_split = 0.1
        self.model.fit(x = X_train, y = y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=validation_split, verbose=1)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir + "/model/gray_scale/")
        self.model.save(self.model_save_dir + "/model/gray_scale/model_{}.h5".format(datetime.now().strftime('%Y-%m-%d')))

    def predict(self):
        if not os.path.exists(self.test_save_dir):
            os.makedirs(self.test_save_dir)
        for img_name in os.listdir(self.test_dir):
            if (img_name.endswith(self.extension)):
                print("Working on img {} \n".format(self.test_dir + "/" + img_name))
                img = imread(self.test_dir + "/" + img_name, as_gray=True).reshape(1,512,512,1)
                pred = self.model.predict(img).reshape(512,512,3) * 255
                pred = pred.astype(int)
                plt.figure()
                plt.imshow(pred)
                plt.show()

                imsave("{}/pred_{}".format(self.test_save_dir, img_name), pred)
        return  

