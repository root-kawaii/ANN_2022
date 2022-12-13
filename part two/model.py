import logging
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import unison_shuffled_copies

tfk = tf.keras
tfkl = tf.keras.layers


def build_GRU_classifier(input_shape, classes, seed):
    model = tfk.Sequential(name="GRU-Model")  # Model
    # Input Layer - need to speicfy the shape of inputs
    input_layer = tfkl.Input(shape=(input_shape), name='Input-Layer')
    x = tfkl.GRU(units=8, activation='tanh',
                 recurrent_activation='sigmoid', stateful=False, return_sequences=True)(input_layer)  # Encoder Layer
    # Repeat Vector
    x = tfkl.GRU(units=8, activation='tanh',
                 recurrent_activation='sigmoid', stateful=False)(x)  # Decoder Layer
    x = tfkl.Dropout(.5, seed=seed)(x)
    # Classifier
    x = tfkl.Dense(16, activation='relu')(x)
    output_layer = tfkl.Dense(classes, activation='softmax')(x)
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])
    return model


def build_1DCNN_classifier(input_shape, classes, seed):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor
    cnn = tfkl.Conv1D(40, 3, padding='same', activation='relu')(input_layer)
    cnn = tfkl.MaxPooling1D()(cnn)
    cnn = tfkl.Conv1D(40, 3, padding='same', activation='relu')(cnn)
    gap = tfkl.GlobalAveragePooling1D()(cnn)
    dropout = tfkl.Dropout(.5, seed=seed)(gap)

    # Classifier
    classifier = tfkl.Dense(128, activation='relu')(dropout)
    output_layer = tfkl.Dense(classes, activation='softmax')(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model


def build_BiLSTM_classifier(input_shape, classes, seed):
    # Build the neural network layer by layer
    input_layer = tfkl.Input(shape=input_shape, name='Input')

    # Feature extractor
    bilstm = tfkl.Bidirectional(
        tfkl.LSTM(128, return_sequences=True))(input_layer)
    bilstm = tfkl.Bidirectional(tfkl.LSTM(128))(bilstm)
    dropout = tfkl.Dropout(.5, seed=seed)(bilstm)

    # Classifier
    classifier = tfkl.Dense(16, activation='relu')(dropout)
    output_layer = tfkl.Dense(classes, activation='softmax')(classifier)

    # Connect input and output through the Model class
    model = tfk.Model(inputs=input_layer, outputs=output_layer, name='model')

    # Compile the model
    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    # Return the model
    return model


def build_model_RESNET(input_shape, nb_classes):
    n_feature_maps = 32

    input_layer = tfk.layers.Input(input_shape)

    # BLOCK 1

    conv_x = tfk.layers.Conv1D(
        filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = tfk.layers.BatchNormalization()(conv_x)
    conv_x = tfk.layers.Activation('relu')(conv_x)

    conv_y = tfk.layers.Conv1D(
        filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = tfk.layers.BatchNormalization()(conv_y)
    conv_y = tfk.layers.Activation('relu')(conv_y)

    conv_z = tfk.layers.Conv1D(
        filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = tfk.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = tfk.layers.Conv1D(
        filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = tfk.layers.BatchNormalization()(shortcut_y)

    output_block_1 = tfk.layers.add([shortcut_y, conv_z])
    output_block_1 = tfk.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = tfk.layers.BatchNormalization()(conv_x)
    conv_x = tfk.layers.Activation('relu')(conv_x)

    conv_y = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = tfk.layers.BatchNormalization()(conv_y)
    conv_y = tfk.layers.Activation('relu')(conv_y)

    conv_z = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = tfk.layers.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = tfk.layers.BatchNormalization()(shortcut_y)

    output_block_2 = tfk.layers.add([shortcut_y, conv_z])
    output_block_2 = tfk.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = tfk.layers.BatchNormalization()(conv_x)
    conv_x = tfk.layers.Activation('relu')(conv_x)

    conv_y = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = tfk.layers.BatchNormalization()(conv_y)
    conv_y = tfk.layers.Activation('relu')(conv_y)

    conv_z = tfk.layers.Conv1D(
        filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = tfk.layers.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = tfk.layers.BatchNormalization()(output_block_2)

    output_block_3 = tfk.layers.add([shortcut_y, conv_z])
    output_block_3 = tfk.layers.Activation('relu')(output_block_3)

    # FINAL

    gap_layer = tfk.layers.GlobalAveragePooling1D()(output_block_3)

    output_layer = tfk.layers.Dense(
        nb_classes, activation='softmax')(gap_layer)

    model = tfk.models.Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                  optimizer=tfk.optimizers.Adam(), metrics='accuracy')

    return model
