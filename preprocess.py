import pandas as pd
import numpy as np
import keras
import glob
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
from mlxtend.preprocessing import minmax_scaling
from sklearn.metrics import roc_curve, auc
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input, BatchNormalization, Multiply, Activation
from keras.optimizers import RMSprop, SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras import backend as K

import os

BASE_PATH = '../data/food-101/images/'

class data:
    
    def __init__(self):
        pass

    def preview_data(self, classes, n = 5):
        # find image file paths
        files = []
        for c in classes:
            files.append(glob.glob(BASE_PATH + c + '/*'))
            print(type(files[len(files) - 1]))
            print('Class:\t' + c + '\tNum Images:\t' + str(len(files[len(files) - 1])))	

        # preview some images of each class
        fig, axes = plt.subplots(len(classes), n, figsize=(20,10))

        if len(classes) > 1:
            for c in range(len(classes)):
                for i in range(n):
                        axes[c, i].imshow(plt.imread(files[c][i]))
                        axes[c, i].set_title(classes[c])

        

    def load_and_augment_data(self, path):
        self.train_datagen = ImageDataGenerator(featurewise_center=False,
                     samplewise_center=False,
                     featurewise_std_normalization=False,
                     samplewise_std_normalization=False,
                     zca_whitening=False,
                     rotation_range=10,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.1,
                     zoom_range=0.2,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=True,
                     vertical_flip=False,
                     rescale=1/255)
        self.train_generator = train_datagen.flow_from_directory(
            "../data/food-101/train",
            target_size=(224,224),
            batch_size=64)
        self.test_datagen = ImageDataGenerator(rescale=1/255) # just rescale to [0-1] for testing set
        self.test_generator = test_datagen.flow_from_directory(
            "../input/food102/food-101/test",
            target_size=(224,224),
            batch_size=64)
    


