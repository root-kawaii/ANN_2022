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
bro = []
labels = train_gen.class_indices
for i in labels.items:
    bro.append(labels.items)
print(bro)

'''

def get_next_batch(generator):
  batch = next(generator)

  image = batch[0]
  target = batch[1]

  print("(Input) image shape:", image.shape)
  print("Target shape:",target.shape)

  # Visualize only the first sample
  image = image[0]
  target = target[0]
  target_idx = np.argmax(target)
  print()
  print("Categorical label:", target)
  print("Label:", target_idx)
  print("Class name:", bro[target_idx])
  fig = plt.figure(figsize=(6, 4))
  plt.imshow(np.uint8(image))

  return batch


_ = get_next_batch(train_gen)

'''