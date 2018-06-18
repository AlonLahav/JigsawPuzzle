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

params.max_images_per_folder = 20
params.puzzle_n_parts = (4, 4) # x - y
move_only_candidate = 1
params.batch_size = 1
LR = 0.25
lr_step = 750000
n_iters = 4000000
inc_n_rel_patches = 1000
video_output = 1
clear_diag_rel = 0
use_overlap_loss = 0

im_idx = 7
simulate_cnn = 'no' # no / perfect / simulate_errors

params.logdir = '/home/alonlahav/git-projects/JigsawPuzzle/models/17'

if not tf.executing_eagerly():
  tfe.enable_eager_execution()


def figure_2_np_array(fig):
  fig.add_subplot(111)
  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return data


def get_simulated_logits(pi, idx1, idx2, simulate_cnn, params):
  coord1 = pi[idx1][2]
  coord2 = pi[idx2][2]
  dy = coord2[0] - coord1[0]
  dx = coord2[1] - coord1[1]
  idx_y = dy + params.pred_radius
  idx_x = dx + params.pred_radius
  logits = -np.ones((params.pred_radius * 2 + 1, params.pred_radius * 2 + 1)) * 100.0
  if idx_y >= 0 and idx_y < params.pred_radius * 2 + 1 and idx_x >= 0 and idx_x < params.pred_radius * 2 + 1:
    logits[idx_y, idx_x] = 100.0

  if simulate_cnn == 'simulate_errors':
    err_matrix = np.random.uniform(-10, 10, logits.shape)
    logits += err_matrix

  return logits


def calc_total_loss(pi):
  loss = 0
  overlap = 0
  for idx1 in range(len(pi)):
    for idx2 in range(idx1 + 1, (len(pi))):
      if 0:#  simulate_cnn == 'no':
        logits = model(images, training=False).numpy()
      else:
        logits = get_simulated_logits(pi, idx1, idx2, simulate_cnn, params)
      sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
      sigmoid_res.shape = (params.pred_radius * 2 + 1, params.pred_radius * 2 + 1)
      est_arg_max = np.unravel_index(np.argmax(sigmoid_res), sigmoid_res.shape)
      conf = sigmoid_res[est_arg_max]
      d = pi[idx2][1] - pi[idx1][1]
      loss += conf * tf.losses.absolute_difference(d + params.pred_radius, est_arg_max) / n_rel_patches
      overlap += np.all((pi[idx2][1] - pi[idx1][1]).numpy() < 1)
  return loss.numpy(), overlap / (len(pi) * (len(pi) - 1) / 2)


def get_left_upper_point(pi):
    return np.argmin(np.array([p[1].numpy() for p in pi]).sum(axis=1))


params.puzzle_n_parts = (20, 20)
images_to_test = run_train_val.get_images_from_folder(params.test_images_path)
if simulate_cnn == 'no':
  if params.method == 'est_dist_ths':
    classes = 4 + 1
  elif params.method == 'pred_matrix':
    classes = (params.pred_radius * 2 + 1) ** 2
  else:
    classes = 4
  model = pair_wise.SimpleNet(params, model_fn=params.model_2_load, classes=classes)

timestr = strftime("%Y-%m-%d_%H:%M", gmtime())
iters_log_fn = 'output_tmp/pzl_' + timestr + '-n-iters-log.txt'
for n_run, n_parts_sq in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50][0:]):
  n_parts = (n_parts_sq, n_parts_sq)
  t_beg = time.time()
  params.puzzle_n_parts = n_parts
  im2use = images_to_test[im_idx]
  im2use = cv2.resize(im2use, (params.puzzle_n_parts[0] * params.patch_size + 1, params.puzzle_n_parts[1] * params.patch_size + 1))
  all_patches, split_order = visual_eval.split_image(im2use, shuffle=False)

  pi = [[all_patches[i], np.array((0., 0.)), split_order[i]] for i in range(len(all_patches))]

  optimizer = tf.train.GradientDescentOptimizer(LR / 2 / 1)
  variables = []
  for p in pi:
    p[1] = tfe.Variable((0., 0.))
    variables.append(p[1])

  if video_output:
    if not os.path.isdir('output_tmp'):
      os.makedirs('output_tmp')
    video = imageio.get_writer('output_tmp/pzl_' + timestr + '--' + str(n_run) + '.mp4', fps=60)

    imageio.imwrite('output_tmp/im.jpg', im2use)

  n_total_patches = params.puzzle_n_parts[0] * params.puzzle_n_parts[1]
  n_rel_patches = 1
  for n in range(n_iters):
    if n % (int(n_total_patches ** 2 / 4)) == 0:
      total_loss, total_overlap = calc_total_loss(pi)
      print (n, n_iters, total_loss, round(total_overlap, 2), n_rel_patches)
    if total_overlap < 0.75 / n_rel_patches and n_rel_patches < 8:
      n_rel_patches *= 2
      print ('Overlap is small -> increasing n_rel_patches')
    if n % lr_step == 0 and n > 0 and LR > 0.2:
      LR *= 0.5
    idx1 = np.random.randint(len(pi))
    for _ in range(n_rel_patches):
      loss = None
      ii = 0
      while 1:
        ii += 1
        idx2 = np.random.randint(len(pi))
        if idx2 == idx1:
          continue
        if ii > 100:
          break
        if total_overlap < -0.2:
          left_upper_point = get_left_upper_point(pi)
          if idx1 == left_upper_point:
            break
          d = pi[idx2][1].numpy() - pi[idx1][1].numpy()
          if d[0] < 0 or d[1] < 0:
            break
        else:
          if np.linalg.norm(pi[idx1][1].numpy() - pi[idx2][1].numpy()) < params.pred_radius:
            break
      images = np.concatenate([pi[idx1][0], pi[idx2][0]], axis=2)[np.newaxis, :]
      if simulate_cnn == 'no':
        logits = model(images, training=False).numpy()
      else:
        logits = get_simulated_logits(pi, idx1, idx2, simulate_cnn, params)
      with tfe.GradientTape() as tape:
        sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
        sigmoid_res.shape = (params.pred_radius * 2 + 1, params.pred_radius * 2 + 1)
        if clear_diag_rel:
          sigmoid_res *= np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        exp_shift_matrix = sigmoid_res
        if 1: # using w.sum
          loss = 0
          for idx, conf in enumerate(sigmoid_res.flatten()):
            if conf > 0.1:
              est_arg_max = np.unravel_index(np.argmax(sigmoid_res), sigmoid_res.shape)
              d = pi[idx2][1] - pi[idx1][1]
              loss += conf * tf.losses.absolute_difference(d + params.pred_radius, est_arg_max) / n_rel_patches
        else: # using argmax
          est_arg_max = np.unravel_index(np.argmax(sigmoid_res), sigmoid_res.shape)
          conf = sigmoid_res[est_arg_max]
          d = pi[idx2][1] - pi[idx1][1]
          if use_overlap_loss:
            ovrlp_loss = 0.3 * tf.reduce_min(tf.nn.relu(0.9 - tf.abs(d)))
          else:
            ovrlp_loss = 0
          loss = ovrlp_loss + conf * tf.losses.absolute_difference(d + params.pred_radius,  est_arg_max) / n_rel_patches
      if not loss is None and loss != 0:
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads[idx1:idx1+1], variables[idx1:idx1+1]), tf.train.get_or_create_global_step())

    if video_output and (n % (int(n_total_patches ** 2 / 20)) == 0):
      fig = plt.figure(1)
      fig.clf()
      pi_ = []
      for p in pi:
        pi_.append([p[0], p[1].numpy()])
      visual_eval.visualize(None, None, pi_, show=False)
      plt.title(str((n, round(total_loss, 2), round(total_overlap, 2))))
      img = figure_2_np_array(fig)
      video.append_data(img)

    if simulate_cnn == 'no':
      if total_overlap < 0.1:
        break
    else:
      if total_loss == 0 or total_overlap < 0.05:
        break

  if video_output:
    plt.close()
    video.close()
  with open('output_tmp/log.txt', 'at') as f:
    f.write('# patches: ' + str(params.puzzle_n_parts) + ' , # iters: ' + str(n) + ' , final loss : ' + str(total_loss) + ' , run time: ' + str(round(time.time() - t_beg, 1)) + '\n')
  with open(iters_log_fn, 'at') as f:
    f.write(str(n) + ', ')
