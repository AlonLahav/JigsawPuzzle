from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import os
import pylab as plt
import numpy as np
import tensorflow as tf

from generate_features import GenerateFeatures
from params import *

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
               params,
               classes=4,
               name=None,
               trainable=True,
               model_fn=None):
    super(SimpleNet, self).__init__(name='')

    self._first_time = True

    data_format = 'channels_last'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.conv1 = layers.Conv2D(8, (3, 3), data_format=data_format, padding='valid', name='conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.max_pool1 = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.conv2 = layers.Conv2D(16, (3, 3), data_format=data_format, padding='valid', name='conv2')
    self.bn_conv2 = layers.BatchNormalization(axis=bn_axis, name='bn_conv2')
    self.max_pool2 = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.flatten = layers.Flatten()
    self.fc_1 = layers.Dense(256, name='fc_1')
    self.fc_2 = layers.Dense(256, name='fc_2')
    self.fc_3 = layers.Dense(256, name='fc_3')
    self.fc_last = layers.Dense(classes, name='fc_last')

    if not model_fn is None:
      if os.path.exists(model_fn):
        images = tf.constant(np.zeros((1, params.patch_size, params.patch_size, 6)).astype(np.float32))
        self(images, training=False)
        self.load_weights(model_fn)
      else:
        print('Keras model was not found: ' + model_fn)

  def call(self, input_tensor, training, visualize=False):
    if self._first_time:
      input_tensor = tf.constant(input_tensor)
      self._first_time = False

    if params.net.only_fc:
      x = input_tensor
    else:
      x = self.conv1(input_tensor)
      x = self.bn_conv1(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.max_pool(x)
      x = self.conv2(x)
      x = self.bn_conv2(x, training=training)
      x = tf.nn.leaky_relu(x)
      x = self.max_pool(x)

    x = self.flatten(x)

    x = self.fc_1(x)
    x = tf.nn.leaky_relu(x)
    x = self.fc_2(x)
    x = tf.nn.leaky_relu(x)
    x = self.fc_3(x)
    x = tf.nn.leaky_relu(x)

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


class NetOnNet(tf.keras.Model):
  '''A netwark instantiation to compare 2 jigsaw puzzle patches

  Args:
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.

  Data input:
    Blob input dimension: (n-examples, h, w, 6).
    Note that n input channels is 6 due to concatenation of the two patches.
    Output: relative position

  Raises:

  '''

  def __init__(self,
               params,
               classes=4,
               name=None,
               trainable=True,
               model_fn=None):
    super(NetOnNet, self).__init__(name='')

    self._first_time = True

    MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28_NoResizer'
    self._generate_features = GenerateFeatures(MODEL_NAME, params.net.features_layer)

    data_format = 'channels_last'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), data_format=data_format)

    self.conv1 = layers.Conv2D(256, (1, 1), data_format=data_format, padding='valid', name='conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')

    self.conv2 = layers.Conv2D(256, (1, 1), data_format=data_format, padding='valid', name='conv2')
    self.bn_conv2 = layers.BatchNormalization(axis=bn_axis, name='bn_conv2')

    self.flatten = layers.Flatten()
    self.fc_1 = layers.Dense(256, name='fc_1')
    self.fc_2 = layers.Dense(256, name='fc_2')
    self.fc_3 = layers.Dense(256, name='fc_3')
    self.fc_last = layers.Dense(classes, name='fc_last')

    if not model_fn is None:
      if os.path.exists(model_fn):
        images = tf.constant(np.zeros((1, params.patch_size, params.patch_size, 6)).astype(np.float32))
        self(images, training=False)
        self.load_weights(model_fn)
      else:
        print('Keras model was not found: ' + model_fn)

  def call(self, input_image, training, visualize=False):
    t1 = np.array(input_image)[:, :, :, :3]
    t2 = np.array(input_image)[:, :, :, 3:]

    if 1:
      ftrs1 = self._generate_features.get_features(t1)
      ftrs2 = self._generate_features.get_features(t2)
      input_data = np.concatenate((ftrs1, ftrs2), axis=3)
    else:
      input_data = np.concatenate((t1, t2), axis=3)

    if self._first_time:
      input_data = tf.constant(input_data)
      self._first_time = False

    if params.net.only_fc:
      x = input_data
    else:
      x = self.conv1(input_data)
      x = self.bn_conv1(x, training=training)
      x = tf.nn.leaky_relu(x)

    if params.net.only_fc:
      pass
    else:
      x = self.conv2(x)
      x = self.bn_conv2(x, training=training)
      x = tf.nn.leaky_relu(x)
    x = self.flatten(x)

    x = self.fc_1(x)
    x = tf.nn.leaky_relu(x)
    x = self.fc_2(x)
    x = tf.nn.leaky_relu(x)
    x = self.fc_3(x)
    x = tf.nn.leaky_relu(x)

    x = self.fc_last(x)

    if visualize:
      plt.figure()
      plt.subplot(2, 2, 1)
      plt.imshow(t1[0]/t1[0].max())
      plt.subplot(2, 2, 2)
      plt.imshow(t2[0]/t2[0].max())
      plt.subplot(2, 2, 3)
      plt.imshow(ftrs1[0, :, :, 0])
      plt.subplot(2, 2, 4)
      plt.imshow(ftrs2[0, :, :, 0])
      plt.show()

    return x

class Genady():
  def __init__(self):
    pass

  def __call__(self, input_image, training):
    pi = input_image[0][:, :, :3]
    pj = input_image[0][:, :, 3:]

    # change to LAB colorspace

    # Dissimilarity calculation (Genady's paper (2))
    assert(pi.shape[0] == pi.shape[1]) # just make sure it is square
    K = pi.shape[1]
    Dij_right = np.sum(np.abs(2 * pi[:, K - 1, :] - pi[:, K - 2, :] - pj[:, 0, :]))
    Dij_left  = np.sum(np.abs(2 * pi[:, 0, :] - pi[:, 1, :] - pj[:, K - 1, :]))
    Dij_down  = np.sum(np.abs(2 * pi[K - 1, :, :] - pi[K - 2, :, :] - pj[0, :, :]))
    Dij_up    = np.sum(np.abs(2 * pi[0, :, :] - pi[1, :, :] - pj[K - 1, :, :]))

    maxval = max(Dij_right, Dij_left, Dij_up, Dij_down)
    #maxval = K * 3 * 4 * (np.max(input_image) - np.min(input_image))# 3 channels, 4 components in sum, K pixels in line
    #maxval = K * (np.max(input_image) - np.min(input_image))# 3 channels, 4 components in sum, K pixels in line

    d = np.array(((maxval,    Dij_up,   maxval),
                  (Dij_left,  maxval,   Dij_right),
                  (maxval,    Dij_down, maxval)))

    # In order to have the similar results to a Deep Network (after sigmoid)
    dummy_logit = 1 - d / maxval * 2
    dummy_logit *= 3

    return dummy_logit