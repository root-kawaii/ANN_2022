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
from model import build_1DCNN_classifier, build_BiLSTM_classifier, build_model_RESNET


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


data = np.load('x_train.npy')
label = np.load('y_train.npy')


data, label = unison_shuffled_copies(data, label)
label = tfk.utils.to_categorical(label)

training, test = data[:1943, :], data[1943:, :]
training_label, test_label = label[:
                                   1943], label[1943:]

# model = build_1DCNN_classifier(
#    (36, 6), 12, seed)

# model = build_BiLSTM_classifier(
#    (36, 6), 12, seed)

model = build_model_RESNET(
    (36, 6), 12)

model.summary()

history = model.fit(
    x=training,
    y=training_label,
    batch_size=64,
    epochs=100,
    validation_split=.2,
    callbacks=[
        tfk.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=250, restore_best_weights=True),
        tfk.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', mode='max', patience=250, factor=0.5, min_lr=1e-5)
    ]
).history

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

predictions = model.predict(test)
predictions.shape


cm = confusion_matrix(np.argmax(test_label, axis=-1),
                      np.argmax(predictions, axis=-1))

# Compute the classification metrics
accuracy = accuracy_score(np.argmax(test_label, axis=-1),
                          np.argmax(predictions, axis=-1))
precision = precision_score(
    np.argmax(test_label, axis=-1), np.argmax(predictions, axis=-1), average='macro')
recall = recall_score(np.argmax(test_label, axis=-1),
                      np.argmax(predictions, axis=-1), average='macro')
f1 = f1_score(np.argmax(test_label, axis=-1),
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
