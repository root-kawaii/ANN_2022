import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import warnings
import time 
from PIL import Image
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


