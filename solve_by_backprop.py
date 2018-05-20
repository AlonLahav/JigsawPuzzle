import os
import random
import time
from time import gmtime, strftime
import copy

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

params.batch_size = 1
LR = 1.0
lr_step = 750
n_iters = 100
inc_n_rel_patches = 100
video_output = 1

if not tf.executing_eagerly():
  tfe.enable_eager_execution()


def figure_2_np_array(fig):
  fig.add_subplot(111)
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


def get_current_shift_matrix(coords1, coords2):
  # TODO: float relative coord -> interpulate it to the "current shift" matrix
  current_shift = np.zeros((params.pred_radius * 2 + 1, params.pred_radius * 2 + 1))
  current_shift[2, 2] = 0.5 * (1 - (coords2[0] - coords1[0]) ** 2) + \
                        0.5 * (1 - (coords2[1] - coords1[1]) ** 2)
  #idx_y = int(round(coords2[0] - coords1[0] + params.pred_radius))
  #idx_x = int(round(coords2[1] - coords1[1] + params.pred_radius))
  #if idx_x >= 0 and idx_x < current_shift.shape[1] and idx_y >= 0 and idx_y < current_shift.shape[0]:
  #  current_shift[idx_y, idx_x] = 1
  return current_shift

images_to_test = run_train_val.get_images_from_folder(params.test_images_path)
if params.method == 'est_dist_ths':
  classes = 4 + 1
elif params.method == 'pred_matrix':
  classes = (params.pred_radius * 2 + 1) ** 2
else:
  classes = 4
model = pair_wise.SimpleNet(params, model_fn=params.model_2_load, classes=classes)

im_idx = 5
all_patches, split_order = visual_eval.split_image(images_to_test[im_idx], shuffle=False)
#visual_eval.visualize(images_to_test[im_idx], split_order, None)
pi = [[all_patches[i], np.array((0., 0.))] for i in range(len(all_patches))]
#pi = [[all_patches[i], np.array((0., 0.))] for i in [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13]]
pi = [[all_patches[i], np.array((0., 0.))] for i in [0, 1, 2, 5, 10, 6, 7, 11, 12]]
#pi = [[all_patches[i], np.array((0., 0.))] for i in [5, 6, 7, 12, 11]]
pi = [[all_patches[i], np.array((0., 0.))] for i in [5, 6, 7]]
pi = [[all_patches[i], np.array((0., 0.))] for i in [64, 65, 66, 79, 80, 81]]

optimizer = tf.train.GradientDescentOptimizer(LR / 2 / 1)
variables = []
for p in pi:
  p[1] = tfe.Variable((0., 0.))
  variables.append(p[1])

if video_output:
  timestr = strftime("%Y-%m-%d_%H:%M", gmtime())
  video = imageio.get_writer('output_tmp/pzl_' + timestr + '.mp4', fps=60)

  imageio.imwrite('output_tmp/im.jpg', images_to_test[im_idx])

n_rel_patches = 1
for n in range(n_iters):
  if n % inc_n_rel_patches == 0 and n != 0 and n_rel_patches < 8:
    n_rel_patches *= 2
  print (n, n_iters)
  if n % lr_step == 0 and n > 0 and LR > 0.2:
    LR *= 0.5
  idx1 = np.random.randint(len(pi))
  for _ in range(n_rel_patches):
    loss = None
    while 1:
      idx2 = np.random.randint(len(pi))
      if idx1 != idx2:
        break
    images = np.concatenate([pi[idx1][0], pi[idx2][0]], axis=2)[np.newaxis, :]
    logits = model(images, training=False).numpy()
    with tfe.GradientTape() as tape:
      sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
      sigmoid_res.shape = (params.pred_radius * 2 + 1, params.pred_radius * 2 + 1)
      exp_shift_matrix = sigmoid_res
      if 0: # TODO: TF cannot derivate it - there must be a way..
        current_shift = get_current_shift_matrix(pi[idx2][1], pi[idx1][1])
      else:
        est_arg_max = np.unravel_index(np.argmax(sigmoid_res), sigmoid_res.shape)
        conf = sigmoid_res[est_arg_max]
        d = pi[idx2][1] - pi[idx1][1]
        loss = conf * tf.losses.absolute_difference(d + params.pred_radius,  est_arg_max) / n_rel_patches
    if not loss is None:
      grads = tape.gradient(loss, variables)
      optimizer.apply_gradients(zip(grads, variables), tf.train.get_or_create_global_step())

  fig = plt.figure(1)
  fig.clf()
  pi_ = []
  for p in pi:
    pi_.append([p[0], p[1].numpy()])

  visual_eval.visualize(None, None, pi_, show=False)
  if video_output:
    plt.title(n)
    img = figure_2_np_array(fig)
    video.append_data(img)

if video_output:
  plt.close()
  video.close()
