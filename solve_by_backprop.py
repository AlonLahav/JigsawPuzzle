import os
import random
import time
from time import gmtime, strftime

import numpy as np
import pylab as plt
import imageio
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from params import *
import pair_wise
import run_train_val
import visual_eval

back_prop_mode = True
params.batch_size = 1
LR = 0.75
lr_step = 500
n_iters = 50
video_output = 1

if not tf.executing_eagerly():
  tfe.enable_eager_execution()


def figure_2_np_array(fig):
  fig.add_subplot(111)
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


train_images = run_train_val.get_images_from_folder(params.train_images_path)
model = pair_wise.SimpleNet(model_fn=params.model_2_load)

im_idx = 0
all_patches, split_order = visual_eval.split_image(train_images[im_idx], shuffle=False)
#visual_eval.visualize(train_images[idx], split_order, None)
#pi = [[all_patches[i], np.array((0., 0.))] for i in range(len(all_patches))]
#pi = [[all_patches[i], np.array((0., 0.))] for i in [0, 1, 2, 5, 10, 6, 7, 11, 12]]
pi = [[all_patches[i], np.array((0., 0.))] for i in [0, 1, 2]]

if video_output:
  timestr = strftime("%Y-%m-%d_%H:%M", gmtime())
  video = imageio.get_writer('output_tmp/pzl_' + timestr + '.mp4', fps=60)

for n in range(n_iters):
  print (n, n_iters)
  if n % lr_step == 0 and n > 0 and LR > 0.2:
    LR *= 0.5
  idx1 = np.random.randint(len(pi))
  idx2 = np.random.randint(len(pi))
  images = np.concatenate([pi[idx1][0], pi[idx2][0]], axis=2)[np.newaxis, :]
  logits = model(images, training=False).numpy()
  sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
  exp_shift = np.array([sigmoid_res[3] - sigmoid_res[2], sigmoid_res[0] - sigmoid_res[1]])  # patch 2 is on the left - on the right side of patch 1
  exp_shift = np.maximum(exp_shift, np.array([-1., -1.]))
  exp_shift = np.minimum(exp_shift, np.array([ 1.,  1.]))
  #exp_shift = np.round(exp_shift)
  conf_shift = np.max(exp_shift ** 2)

  fig = plt.figure(1)
  fig.clf()
  visual_eval.visualize(None, None, pi, show=False)
  current_shift = pi[idx2][1] - pi[idx1][1]
  pi[idx2][1] += LR * conf_shift * (exp_shift - current_shift)

  if video_output:
    plt.title(n)
    img = figure_2_np_array(fig)
    video.append_data(img)

if video_output:
  plt.close()
  video.close()
