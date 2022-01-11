# import the keras callback to save the best model based on validation loss
from keras.callbacks import ModelCheckpoint  # , EarlyStopping
from EEGModels import EEGNet, EEGNet2
import h5py
import os
import numpy as np
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


f = open("resultAUC.txt","w+")
a = 10
for i in range(10):
    f.write('{0:d} {1:3d} \n'.format(i, a))

f.close()