from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import pylab as plt
import numpy as np
import tensorflow as tf

layers = tf.keras.layers

class SimpleNet(tf.keras.Model):
  '''Simple netwark instantiation to compare 2 jigsaw puzzle patches

  Args:
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.

  Data input:
    Blob input dimension: (n-examples, 28, 28, 6).
    Note that n inpput channels is 6 due to concatenation of the two patches.
    Output: 4 neurons - for each direction (can be 0 for all if there is no match)

  Raises:

  '''
  def __init__(self,
               classes=4,
               name=None,
               trainable=True,
               model_fn=None):
    super(SimpleNet, self).__init__(name='')

    data_format = 'channels_last'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.conv1 = layers.Conv2D(8, (3, 3), data_format=data_format, padding='valid', name='conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.max_pool1 = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    #self.conv2 = layers.Conv2D(16, (3, 3), data_format=data_format, padding='valid', name='conv2')
    #self.bn_conv2 = layers.BatchNormalization(axis=bn_axis, name='bn_conv2')
    #self.max_pool2 = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.flatten = layers.Flatten()
    self.fc_1 = layers.Dense(256, name='fc_1')
    self.fc_2 = layers.Dense(256, name='fc_2')
    self.fc_3 = layers.Dense(256, name='fc_3')
    self.fc_last = layers.Dense(classes, name='fc_last')

    if not model_fn is None:
      if os.path.exists(model_fn):
        images = tf.constant(np.zeros((1, 16, 16, 6)).astype(np.float32))
        self(images, training=False)
        self.load_weights(model_fn)
      else:
        print('Keras model was not found: ' + model_fn)

  def call(self, input_tensor, training, visualize=False):
    if len(self.conv1.weights) == 0:
      x = self.conv1(tf.constant(input_tensor))
    else:
      x = self.conv1(input_tensor)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)
    #x = self.max_pool(x)

    #x = self.conv2(x)
    #x = self.bn_conv2(x, training=training)
    #x = tf.nn.relu(x)
    #x = self.max_pool(x)

    x = self.flatten(x)

    x = self.fc_1(x)
    x = tf.nn.relu(x)
    if 1:
      x = self.fc_2(x)
      x = tf.nn.relu(x)
      x = self.fc_3(x)
      x = tf.nn.relu(x)

    x = self.fc_last(x)

    if visualize:
      plt.figure()
      plt.subplot(1, 2, 1)
      plt.imshow(input_tensor.numpy()[0, :, :, :3])
      plt.subplot(1, 2, 2)
      plt.imshow(input_tensor.numpy()[0, :, :, 3:])
      plt.suptitle(x.numpy())
      plt.show()

    return x