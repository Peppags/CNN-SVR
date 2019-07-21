# -*- coding:utf-8 -*-

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# session = tf.Session(config=config)
#
# KTF.set_session(session)

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D
from keras.layers.merge import concatenate
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import scipy.stats as stats


def grna_preprocess(lines):
    length = 23
    data_n = len(lines)
    seq = np.zeros((data_n, length, 4), dtype=int)
    for l in range(data_n):
        data = lines[l]
        seq_temp = data
        for i in range(length):
            if seq_temp[i] in "Aa":
                seq[l, i, 0] = 1
            elif seq_temp[i] in "Cc":
                seq[l, i, 1] = 1
            elif seq_temp[i] in "Gg":
                seq[l, i, 2] = 1
            elif seq_temp[i] in "Tt":
                seq[l, i, 3] = 1
    return seq


def epi_preprocess(lines):
    length = 23
    data_n = len(lines)
    epi = np.zeros((data_n, length), dtype=int)
    for l in range(data_n):
        data = lines[l]
        epi_temp = data
        for i in range(length):
            if epi_temp[i] in "A":
                epi[l, i] = 1
            elif epi_temp[i] in "N":
                epi[l, i] = 0
    return epi


def preprocess(file_path, usecols):
    data = pd.read_csv(file_path, usecols=usecols)
    data = np.array(data)
    ctcf, dnase, h3k4me3, rrbs = epi_preprocess(data[:, 0]), epi_preprocess(data[:, 1]), epi_preprocess(data[:, 2]), epi_preprocess(data[:, 3])
    epi = []
    for i in range(len(data)):
        ctcf_t, dnase_t, h3k4me3_t, rrbs_t = pd.DataFrame(ctcf[i]), pd.DataFrame(dnase[i]), pd.DataFrame(h3k4me3[i]), pd.DataFrame(rrbs[i])
        epi_t = pd.concat([ctcf_t, dnase_t, h3k4me3_t, rrbs_t], axis=1)
        epi_t = np.array(epi_t)
        epi.append(epi_t)
    epi = np.array(epi)
    return epi


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, usecols=[4, 9])
    train_data = np.array(train_data)
    train_seq, train_y = train_data[:, 0], train_data[:, 1]
    train_seq = grna_preprocess(train_seq)
    train_epi = preprocess(train_file, [5, 6, 7, 8])
    train_y = train_y.reshape(len(train_y), -1)

    test_data = pd.read_csv(test_file, usecols=[4, 9])
    test_data = np.array(test_data)
    test_seq, test_y = test_data[:, 0], test_data[:, 1]
    test_seq = grna_preprocess(test_seq)
    test_epi = preprocess(test_file, [5, 6, 7, 8])
    test_y = test_y.reshape(len(test_y), -1)
    return train_seq, test_seq, train_epi, test_epi, train_y, test_y


# Build model
def build_model():
    dropout = 0.3
    seq_input = Input(shape=(23, 4))
    seq_conv1 = Convolution1D(256, 5, kernel_initializer='glorot_uniform', name='seq_conv_1')(seq_input)
    seq_act1 = Activation('relu', name='seq_activation1')(seq_conv1)
    seq_pool1 = AveragePooling1D(2, name='seq_pooling_1')(seq_act1)
    seq_drop1 = Dropout(dropout)(seq_pool1)

    seq_conv2 = Convolution1D(256, 5, kernel_initializer='glorot_uniform', name='seq_conv_2')(seq_drop1)
    seq_act2 = Activation('relu', name='seq_activation_2')(seq_conv2)
    seq_pool2 = AveragePooling1D(2, name='seq_pooling_2')(seq_act2)
    seq_drop2 = Dropout(dropout)(seq_pool2)
    seq_flat = Flatten()(seq_drop2)

    seq_dense1 = Dense(256, activation='relu', name='seq_dense_1')(seq_flat)
    seq_drop3 = Dropout(dropout)(seq_dense1)
    seq_dense2 = Dense(128, activation='relu', name='seq_dense_2')(seq_drop3)
    seq_drop4 = Dropout(dropout)(seq_dense2)
    seq_dense3 = Dense(64, activation='relu', name='seq_dense_3')(seq_drop4)
    seq_drop5 = Dropout(dropout)(seq_dense3)
    seq_out = Dense(40, activation='relu', name='seq_dense_4')(seq_drop5)

    epi_input = Input(shape=(23, 4))
    epi_conv1 = Convolution1D(256, 5, kernel_initializer='glorot_uniform', name='epi_conv_1')(epi_input)
    epi_act1 = Activation('relu', name='epi_activation_1')(epi_conv1)
    epi_pool1 = AveragePooling1D(2, name='epi_pooling_1')(epi_act1)
    epi_drop1 = Dropout(dropout)(epi_pool1)

    epi_conv2 = Convolution1D(256, 5, kernel_initializer='glorot_uniform', name='epi_conv_2')(epi_drop1)
    epi_act2 = Activation('relu', name='epi_activation_2')(epi_conv2)
    epi_pool2 = AveragePooling1D(2, name='epi_pooling_2')(epi_act2)
    epi_drop2 = Dropout(dropout)(epi_pool2)
    epi_flat = Flatten()(epi_drop2)

    epi_dense1 = Dense(256, activation='relu', name='epi_dense_1')(epi_flat)
    epi_drop3 = Dropout(dropout)(epi_dense1)
    epi_dense2 = Dense(128, activation='relu', name='epi_dense_2')(epi_drop3)
    epi_drop4 = Dropout(dropout)(epi_dense2)
    epi_dense3 = Dense(64, activation='relu', name='epi_dense_3')(epi_drop4)
    epi_drop5 = Dropout(dropout)(epi_dense3)
    epi_out = Dense(40, activation='relu', name='epi_dense_4')(epi_drop5)

    merged = concatenate([seq_out, epi_out], axis=-1)

    pretrain_model = Model(inputs=[seq_input, epi_input], outputs=[merged])

    # Load weights for the model
    pretrain_model.load_weights("weights/weights.h5", by_name=True)

    prediction = Dense(1, activation='linear', name='prediction')(merged)
    model = Model([seq_input, epi_input], prediction)
    return merged, model


if __name__ == '__main__':

    train_path = "data/training_example.csv"
    test_path = "data/testing_example.csv"

    # Load data
    seq_train, seq_test, epi_train, epi_test, y_train, y_test = load_data(train_path, test_path)

    merged, model = build_model()

    new_model = Model(model.inputs, outputs=[merged])
    x_train = new_model.predict([seq_train, epi_train])
    x_test = new_model.predict([seq_test, epi_test])

    x_train, x_test = np.array(x_train), np.array(x_test)

    # Select important features from initial CNN features
    selected_cnn_fea_cols = [17, 26, 9, 19, 30, 6, 12, 39, 36, 21, 22, 3, 25]
    x_train = x_train[:, selected_cnn_fea_cols]
    x_test = x_test[:, selected_cnn_fea_cols]

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    clf = SVR(kernel="rbf", gamma=0.12, C=1.7, epsilon=0.11, verbose=1)

    # Fit the SVR model according to the given training data
    clf.fit(x_train, y_train)

    # Perform regression on samples in x_test
    y_pred = clf.predict(x_test)
    print(y_pred)

    # Calculate Spearman correlation coefficient
    # Spearman_correlation, _ = stats.stats.spearmanr(y_test, y_pred)

    # Print Spearman correlation result
    # print("Spearman correlation=%.3f" % (Spearman_correlation))






























