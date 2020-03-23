import pandas as pd
import numpy as np
import enum
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
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.callbacks.callbacks import ProgbarLogger
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras import backend as K
from preprocess import DataCollector
import keras.applications.resnet as resnet
import keras.applications.densenet as densenet
import os

class ModelType(enum.Enum):
    ResNet50 = 1
    ResNet101 = 2
    DenseNet121 = 3
    DenseNet169 = 4

class ImageEncoder:
    
    def __init__(self, modeltype=ModelType.ResNet50):
        self.model = None
        self.modelType = modeltype
        if modeltype == ModelType.ResNet50:
            self.model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3), 
                                                    pooling='avg', classes=1000)
        elif modeltype == ModelType.ResNet101:
            self.model = keras.applications.resnet.ResNet101(include_top=False, weights='imagenet', input_shape=(256, 256, 3), 
                                                    pooling='avg', classes=1000)
        elif modeltype == ModelType.DenseNet121:
            self.model = keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(256, 256, 3), 
                                                    pooling='avg', classes=1000)
        elif modeltype == ModelType.DenseNet169:
            self.model = keras.applications.densenet.DenseNet169(include_top=False, weights='imagenet', input_shape=(256, 256, 3), 
                                                    pooling='avg', classes=1000)
        
    def encode_img(self, filename):
        if self.model == None:
            return None
        
        img_width, img_height = 256, 256
        img = image.load_img(filename, target_size = (img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        if self.modelType in [ModelType.ResNet50, ModelType.ResNet101]:
            img = resnet.preprocess_input(img)
        if self.modelType in [ModelType.DenseNet121, ModelType.DenseNet169]:
            img = densenet.preprocess_input(img)
        
        return self.model.predict(img)
        
        
        
        
    
    
    
    
    
    
    
    
    
