import os
import random
import warnings


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



# Hide GPU from visible devices
#tf.config.set_visible_devices([], 'GPU')
#tf.debugging.set_log_device_placement(True)


image_set = []
label_set = []

warnings.filterwarnings("ignore")
tfk = tf.keras
tfkl = tf.keras.layers
print(tf.__version__)

# Random seed for reproducibility
seed = 777

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)


# Recreate folders division every time
'''
input_folder = "training_data_final"
output = "training_data_final"
splitfolders.ratio(input_folder, output=output, seed=seed, ratio=(.8, .1, .1)) 
'''

train_dataset = tf.keras.utils.image_dataset_from_directory('training_data_final/train',
                                                            shuffle=True,
                                                            batch_size=8,
                                                            label_mode='categorical',
                                                            image_size=(96,96))

validation_dataset = tf.keras.utils.image_dataset_from_directory('training_data_final/val',
                                                                 shuffle=True,
                                                                 batch_size=8,
                                                                 label_mode='categorical',
                                                                 image_size=(96,96))

test_dataset = tf.keras.utils.image_dataset_from_directory('training_data_final/test',
                                                                 shuffle=True,
                                                                 label_mode='categorical',
                                                                 batch_size=8,
                                                                 image_size=(96,96))


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tfk.Sequential([
  tfkl.RandomFlip('horizontal'),
  tfkl.RandomFlip('vertical'),
  tfkl.RandomRotation(0.15),
  tfkl.RandomHeight(0.2),
  tfkl.RandomWidth(0.2),
  tfkl.RandomZoom(0.15),
  tfkl.RandomContrast(factor=0.1),
  tfkl.GaussianNoise(0.1)
])

IMG_SHAPE = (96,96) + (3,)

supernet = tf.keras.applications.convnext.ConvNeXtBase(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SHAPE
)


supernet.trainable = False

class1 = 2829/(8*148)
class2 = 2829/(8*425)
class3 = 2829/(8*412)
class4 = 2829/(8*408)
class5 = 2829/(8*424)
class6 = 2829/(8*177)
class7 = 2829/(8*429)
class8 = 2829/(8*406)

weights = {0: class1, 1: class2, 2: class3, 3: class4, 4: class5, 5: class6, 6: class7, 7: class8}

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

inputs = tf.keras.Input(shape=(96, 96, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.convnext.preprocess_input(x)
x = supernet(x,training = False)
x = global_average_layer(x)
x = tfkl.Dropout(0.3, seed=seed)(x)
x = tfkl.Dense(
    256, 
    activation='relu',
    kernel_initializer = tfk.initializers.HeUniform(seed),)(x)
x = tfkl.Dropout(0.3, seed=seed)(x)
outputs = tfkl.Dense(
    8, 
    activation='softmax',
    kernel_initializer = tfk.initializers.GlorotUniform(seed))(x)


model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.003
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tfk.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])


initial_epochs = 20

model.summary()

loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,
                    class_weight=weights)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



## FINE TUNING

supernet.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(supernet.layers))

# Fine-tune from this layer onwards
fine_tune_at = 0

# Freeze all the layers before the `fine_tune_at` layer
for layer in supernet.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tfk.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

fine_tune_epochs = 70
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset,
                         class_weight=weights)

acc = history_fine.history['accuracy']
val_acc = history_fine.history['val_accuracy']

loss = history_fine.history['loss']
val_loss = history_fine.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.6, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 1.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


model.save('b2')

#model.save("convNext")