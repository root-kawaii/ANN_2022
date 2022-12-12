import tensorflow as tf
import os
import random
import warnings
import numpy as np
from sklearn.metrics import classification_report


tfk = tf.keras
tfkl = tf.keras.layers

test_dataset = tf.keras.utils.image_dataset_from_directory('training_data_final',
                                                                 shuffle=True,
                                                                 label_mode='categorical',
                                                                 batch_size=8,
                                                                 image_size=(96,96))

convnext = tf.keras.models.load_model('b1')
b7 = tf.keras.models.load_model('b2')
third = tf.keras.models.load_model('b1')

convnext._name = 'bro'
third._name = 'bro2'

inputs = tf.keras.Input(shape=(96, 96, 3))
x1 = convnext(inputs)
x2 = b7(inputs)
x3 = third(inputs)
outputs = tf.keras.layers.Multiply()([x1,x2,x3])


model1 = tf.keras.Model(inputs, outputs)
model1.compile(loss=tfk.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

loss, accuracy = model1.evaluate(test_dataset)
print('Test accuracy :', accuracy)  

model1.save('combined3')