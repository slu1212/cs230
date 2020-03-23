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
import keras.applications.resnet as resnet
from keras import backend as K

import os

class DataCollector:
    
    def __init__(self, preprocessing_function=resnet.preprocess_input):
        self.train_datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    def load_and_augment_data(self, path, class_mode = None, shuffle=False):
        return self.train_datagen.flow_from_directory(path, batch_size=64, shuffle=shuffle, class_mode = class_mode)

        
    


