import os
import random
import time
import warnings


import splitfolders
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

'''

for dir in os.listdir('training_data_final'):
    for file in os.listdir('training_data_final/'+dir):
        image = Image.open('training_data_final/'+dir+'/'+file).convert('L')
        print("Original image shape: ", image.size)
        print("Resized image shape: ", image.size)
        fig = plt.figure(figsize=(8, 8))
        print(file)
        ##plt.imshow(image)
        ##plt.show()
        print(dir)
        label_set.append(dir)
        image_set.append(np.array(image, dtype=np.float32))

image_set , label_set = shuffle(image_set,label_set)
image_set = image_set/255
'''
# Recreate folders division every time

##input_folder = "training_data_final"
##output = "training_data_final"
##splitfolders.ratio(input_folder, output=output, seed=seed, ratio=(.8, .1, .1)) 


data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
train_gen = data_gen.flow_from_directory(directory='training_data_final/train',
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=True,
                                               seed=seed)

val_gen = data_gen.flow_from_directory(directory='training_data_final/val',
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=True,
                                               seed=seed)

test_gen = data_gen.flow_from_directory(directory='training_data_final/test',
                                               target_size=(256,256),
                                               color_mode='rgb',
                                               classes=None, # can be set to labels
                                               class_mode='categorical',
                                               batch_size=8,
                                               shuffle=True,
                                               seed=seed)


# print(train_gen.class_indices)
# from dic to array of labels
labels = []
labels_dic = train_gen.class_indices

for i in labels_dic.items():
    labels.append(i[0])


def get_next_batch(generator):
  batch = next(generator)

  image = batch[0]
  target = batch[1]
  print()
  print("(Input) image shape:", image.shape)
  print("Target shape:",target.shape)

  # Visualize only the first sample
  image = image[0]
  target = target[0]
  target_idx = np.argmax(target)
  print()
  print("Categorical label:", target)
  print("Label:", target_idx)
  print("Class name:", labels[target_idx])
  fig = plt.figure(figsize=(6, 4))
  plt.imshow(np.uint8(image))

  return batch


# Create some augmentation examples
# Get sample image
image = next(train_gen)[0][4]

# Create an instance of ImageDataGenerator for each transformation
rot_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=30)
shift_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=50)
zoom_gen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.3)
flip_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)

# Get random transformations
rot_t = rot_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Rotation:', rot_t, '\n')
shift_t = shift_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Shift:', shift_t, '\n')
zoom_t = zoom_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Zoom:', zoom_t, '\n')
flip_t = flip_gen.get_random_transform(img_shape=(256, 256), seed=seed)
print('Flip:', flip_t, '\n')

# Apply the transformation
gen = tf.keras.preprocessing.image.ImageDataGenerator(fill_mode='constant', cval=0.)
rotated = gen.apply_transform(image, rot_t)
shifted = gen.apply_transform(image, shift_t) 
zoomed = gen.apply_transform(image, zoom_t) 
flipped = gen.apply_transform(image, flip_t)  

# Plot original and augmented images
fig, ax = plt.subplots(1, 5, figsize=(15, 45))
ax[0].imshow(np.uint8(image))
ax[0].set_title('Original')
ax[1].imshow(np.uint8(rotated))
ax[1].set_title('Rotated')
ax[2].imshow(np.uint8(shifted))
ax[2].set_title('Shifted')
ax[3].imshow(np.uint8(zoomed))
ax[3].set_title('Zoomed')
ax[4].imshow(np.uint8(flipped))
ax[4].set_title('Flipped')


