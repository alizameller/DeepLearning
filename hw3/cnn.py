#!/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sklearn.inspection
from sklearn.inspection import DecisionBoundaryDisplay
import math
from mnist import MNIST

mndata = MNIST('./')
train_images, train_labels = mndata.load_training()
train_images = tf.reshape(train_images, (60000, 28, 28))
print(tf.shape(train_images))
print(tf.shape(train_labels))

test_images, test_labels = mndata.load_testing()
test_images = tf.reshape(test_images, (60000, 28, 28))
print(tf.shape(test_images))
print(tf.shape(test_labels))

for i in range(9):  
    plt.subplot(330 + 1 + i)
    plt.imshow(train_images[i], cmap=plt.get_cmap('gray'))
plt.show()