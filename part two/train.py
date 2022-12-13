import logging
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from model import build_1DCNN_classifier, build_BiLSTM_classifier, build_model_RESNET, build_GRU_classifier


plt.rc('font', size=16)

tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Import datset
data = np.load('x_train.npy')
label = np.load('y_train.npy')
# Load labels
label = tfk.utils.to_categorical(label)

# Split the dataset
train = []
val = []
test = []
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

x_train, x_test, y_train, y_test = train_test_split(
    data, label, test_size=validation_ratio + test_ratio, random_state=seed, stratify=label)

x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=test_ratio/(
    validation_ratio + test_ratio), random_state=seed, stratify=y_test)

# Standardization Train, we only fit the scaler on the train set
values = x_train.reshape(-1, x_train.shape[-1])
scaler = StandardScaler()
scaler = scaler.fit(values)
x_train_norm = scaler.transform(values).reshape(x_train.shape)
# Standardization Validation
values_val = x_val.reshape(-1, x_val.shape[-1])
x_val_norm = scaler.transform(values_val).reshape(x_val.shape)
# Standardization Test
values_test = x_test.reshape(-1, x_test.shape[-1])
x_test_norm = scaler.transform(values_test).reshape(x_test.shape)

'''
# Plot normalization
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.hist(x_train.reshape(-1, x_train.shape[-1])[:, 0], range=[-5000, 5000])
plt.title("train original")
plt.subplot(2, 2, 2)
plt.hist(x_train_norm.reshape(-1,
         x_train_norm.shape[-1])[:, 0], range=[-20, 20])
plt.title("train normalized")
plt.subplot(2, 2, 3)
plt.hist(x_val.reshape(-1, x_val.shape[-1])[:, 0])
plt.title("val orig")
plt.subplot(2, 2, 4)
plt.hist(x_val_norm.reshape(-1, x_val_norm.shape[-1])[:, 0])
plt.title("val normalized")
plt.show()
'''

# Choosing the model

# model = build_1DCNN_classifier(
#    (36, 6), 12, seed)

model = build_BiLSTM_classifier(
    (36, 6), 12, seed)

# model = build_model_RESNET(
#    (36, 6), 12)

# model = build_GRU_classifier(
#    (36, 6), 12, seed)

model.summary()

history = model.fit(
    x=x_train_norm,
    y=y_train,
    batch_size=128,
    epochs=500,
    validation_data=(x_val_norm, y_val),
    callbacks=[
        tfk.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=25, restore_best_weights=True),
        tfk.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', mode='max', patience=25, factor=0.5, min_lr=1e-5)
    ]
).history

# Plotting
best_epoch = np.argmax(history['val_accuracy'])
plt.figure(figsize=(17, 4))
plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
plt.plot(history['val_loss'], label='Validation loss',
         alpha=.9, color='#5a9aa5')
plt.axvline(x=best_epoch, label='Best epoch',
            alpha=.3, ls='--', color='#5a9aa5')
plt.title('Categorical Crossentropy')
plt.legend()
plt.grid(alpha=.3)
plt.show()

plt.figure(figsize=(17, 4))
plt.plot(history['accuracy'], label='Training accuracy',
         alpha=.8, color='#ff7f0e')
plt.plot(history['val_accuracy'], label='Validation accuracy',
         alpha=.9, color='#5a9aa5')
plt.axvline(x=best_epoch, label='Best epoch',
            alpha=.3, ls='--', color='#5a9aa5')
plt.title('Accuracy')
plt.legend()
plt.grid(alpha=.3)
plt.show()

plt.figure(figsize=(18, 3))
plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
plt.axvline(x=best_epoch, label='Best epoch',
            alpha=.3, ls='--', color='#5a9aa5')
plt.legend()
plt.grid(alpha=.3)
plt.show()

model.save('on_a_gang_model')

predictions = model.predict(x_test_norm)
predictions.shape


cm = confusion_matrix(np.argmax(y_test, axis=-1),
                      np.argmax(predictions, axis=-1))

# Compute the classification metrics
accuracy = accuracy_score(np.argmax(y_test, axis=-1),
                          np.argmax(predictions, axis=-1))
precision = precision_score(
    np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
recall = recall_score(np.argmax(y_test, axis=-1),
                      np.argmax(predictions, axis=-1), average='macro')
f1 = f1_score(np.argmax(y_test, axis=-1),
              np.argmax(predictions, axis=-1), average='macro')
print('Accuracy:', accuracy.round(4))
print('Precision:', precision.round(4))
print('Recall:', recall.round(4))
print('F1:', f1.round(4))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm.T, cmap='Blues')
plt.xlabel('True labels')
plt.ylabel('Predicted labels')
plt.show()
