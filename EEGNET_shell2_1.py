#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:22:59 2017

@author: vlawhern
"""

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
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, roc_auc_score

# channels, samples. Change to whatever you need


chans, samples = 32, 512

# other variables
batch_size = 4

data_file_name = 'CBW_data_2s512norm_ver2.mat'
DATA_FOLDER_PATH = '/media/eeglab/YG_Storage/cyberwell/CBW_Data/Normdata/'

FILE_PATH = DATA_FOLDER_PATH + '/' + data_file_name

# mat_2 = scipy.io.loadmat(FILE_PATH)
mat_2 = h5py.File(FILE_PATH)
mat_2.keys()

data_x = mat_2['data']
data_y = mat_2['label']
# data_y = data_y[:, 1]
data_y = to_categorical(data_y, 2)
# X_train = np.transpose(batch_images, (0, 3, 1, 2))
data_x = np.transpose(data_x, (0, 2, 1))
# data_x = data_x[:, 0:6, :]
data_x = np.expand_dims(data_x, axis=1)

# X_train = data_x[0:273, :, :, :]
# Y_train = data_y[0:273, :]
# X_validate = data_x[273:364, :, :, :]
# Y_validate = data_y[273:364, :]
# X_test =data_x[364:455, :, :, :]

# X_train = data_x[0:1138, :, :, :]
# Y_train = data_y[0:1138, :]
# X_validate = data_x[1139:1518, :, :, :]
# Y_validate = data_y[1139:1518, :]
# X_test = data_x[1519:1898, :, :, :]
# Y_test = data_y[1519:1898, :]

X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
# X_train, X_test, Y_train, Y_test = KFold(5, shuffle=False, random_state=None)
# construct the model
model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
               kernels=[(2, 32), (8, 4)], dropoutRate=0.25)


# model = EEGNet2(nb_classes = 2, Chans = chans, Samples = samples, regRate = 0.001,
#            dropoutRate = 0.25, kernLength = 64, numFilters = 8)

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=["accuracy"])
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd', metrics=["accuracy"])

# make the model checkpoint
# set checkpointPath to be whatever the file path you want
dataset_name = 'EEGNET_Cyber5'
os.makedirs('models/%s' % dataset_name)
checkpointPath = '.models'
checkpointer = ModelCheckpoint(filepath=checkpointPath, verbose=1, save_best_only=True)

# then fit the model.
# The EEG data (X_train, X_test, X_validate) needs to be in the following format:
#       (trials, kernels, channels, samples).
# The number of kernels when you start the model is 1 so the format is
#       (trials, 1, channels, samples).
#
# Y_train, Y_test, Y_validate needs to be a one-hot encoding of your classes.
# where rows is trials.
#
# for a binary classification problem you'll need two columns, with a 1 indicating
# class membership and 0 otherwise.

# fit the model
fittedModel = model.fit(X_train, Y_train, batch_size=batch_size, epochs=100,
                        verbose=2, validation_data=(X_test, Y_test),
                        callbacks=[checkpointer])

# fig, loss_ax = plt.subplots()
#
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(fittedModel.history['loss'], 'y', label='train loss')
# loss_ax.plot(fittedModel.history['val_loss'], 'r', label='val loss')
#
# acc_ax.plot(fittedModel.history['acc'], 'b', label='train acc')
# acc_ax.plot(fittedModel.history['val_acc'], 'g', label='val acc')
#
# loss_ax.set_xlabel('iteration')
# loss_ax.set_ylabel('loss')
# acc_ax.set_ylabel('accuracy')
#
# loss_ax.legend(loc='upper left')
# acc_ax.legend(loc='lower left')
# plt.show()

from sklearn.metrics import roc_curve

y_pred_keras = model.predict(X_test).ravel()

# Y_test_1 = Y_test[:,1]
# y_pred_keras_1 = y_pred_keras[:,1]
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test.ravel(), y_pred_keras)

from sklearn.metrics import auc

auc_keras = auc(fpr_keras, tpr_keras)

# from sklearn.ensemble import RandomForestClassifier
# # Supervised transformation based on random forests
# rf = RandomForestClassifier(max_depth=3, n_estimators=10)
# rf.fit(X_train, Y_train)
#
# y_pred_rf = rf.predict_proba(X_test)[:, 1]
# fpr_rf, tpr_rf, thresholds_rf = roc_curve(Y_test.ravel(), y_pred_rf)
# auc_rf = auc(fpr_rf, tpr_rf)

np.save("GCNN_fprEEGNETn26.npy", fpr_keras)
np.save("GCNN_tprEEGNETn26.npy", tpr_keras)
np.save("GCNN_aucEEGNETn26.npy", auc_keras)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Testing AUC (area = {:.3f})'.format(auc_keras))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve with Testing Set')
plt.legend(loc='best')
plt.show()
# Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve (zoomed in at top left)')
# plt.legend(loc='best')
# plt.show()

# make some predictions on test set
# predictions = model.predict(X_test)
