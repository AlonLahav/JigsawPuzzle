from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os, sys
import imageio
import cv2
import pylab as plt
import numpy as np
import shutil

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from params import *
import pair_wise
import data_input
from pprint import pprint
from ttictoc import TicToc

# Next to do:
# - Statistics over many images
# - Speed up - using GPU & batch size > 1
# - Check other networks
# - Implement best bodies -> solve puzzle using genady's alg.
# - Including indications when no piece is at the borders

tfe.enable_eager_execution()

params.net.net_type = 'genady'
test_images = data_input.get_images_from_folder(params.test_images_path)


def solve_greedy(noise=0, show=0):
  # Init net
  assert(params.method == 'pred_matrix')
  classes = (params.pred_radius * 2 + 1) ** 2
  if params.net.net_type == 'simple':
    assert(params.preprocess == 'mean-0')
    model = pair_wise.SimpleNet(params, model_fn=params.model_2_load, classes=classes)
  elif params.net.net_type == 'genady':
    model = pair_wise.Genady()
  else:
    assert(params.preprocess is None)
    model = pair_wise.NetOnNet(params, model_fn=params.model_2_load, classes=classes)

  n_labels = (params.pred_radius * 2 + 1) ** 2

  # Take & prepare one image
  idx = 0
  im = imageio.imread(test_images[idx]).astype('float32')
  im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0],
                       params.patch_size * params.puzzle_n_parts[1]))
  if params.preprocess == 'mean-0':
    im = im / im.max()
    im = im - im.mean()

  im += np.random.normal(size=im.shape) * noise

  # "Cut" the image
  def im_cut(im, x, y):
    return im[y * params.patch_size:(y + 1) * params.patch_size,
              x * params.patch_size:(x + 1) * params.patch_size]

  # Build compainion matrix
  all_r = []
  pp = params.puzzle_n_parts[0]
  r_mat = np.zeros((pp, pp))
  comp_matrix = []
  for _ in range(pp):
    comp_matrix.append([np.nan] * pp)
  with TicToc('Calc companion matrix'):
    for x1 in range(params.puzzle_n_parts[0]):
      for y1 in range(params.puzzle_n_parts[1]):
        comp_matrix[y1][x1] = EasyDict()
        im1 = im_cut(im, x1, y1)
        im1_rel = comp_matrix[y1][x1]
        im1_rel.left = []
        im1_rel.right = []
        im1_rel.up = []
        im1_rel.down = []
        s_max = -1     * np.ones((3, 3))
        a_max = []
        for x2 in range(params.puzzle_n_parts[0]):
          for y2 in range(params.puzzle_n_parts[1]):
            if x1 == x2 and y1 == y2:
              continue
            # Estimate 2 pieces relationship
            im2 = im_cut(im, x2, y2)
            concat_im = np.concatenate([im1, im2], axis=2)
            images = [concat_im]
            logits = model(images, training=False)
            if params.net.net_type != 'genady':
              logits = logits.numpy()
            logits = logits.reshape((3, 3))
            sigmoid_res = 1/(1+np.exp(-logits))

            # Update best piece
            im1_rel.left.append ([sigmoid_res[1, 0], (y2, x2)])
            im1_rel.right.append([sigmoid_res[1, 2], (y2, x2)])
            im1_rel.up.append   ([sigmoid_res[0, 1], (y2, x2)])
            im1_rel.down.append ([sigmoid_res[2, 1], (y2, x2)])
        sort_and_keep_best(im1_rel, 5)
        r = check_results(im1_rel, y1, x1)
        all_r.append(r)
        r_mat[y1, x1] = r
        if 0:
          print('y1, x1 : ', y1, x1)
          print('  % correct: ', r)
  print('  % correct total: ', np.mean(all_r))

  if show:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im - im.min())
    plt.subplot(1, 2, 2)
    plt.imshow(r_mat)
    plt.show()


  return np.mean(all_r)

def check_results(im1_rel, y1, x1):
  if params.net.net_type == 'genady':
    th = 0.55
  else:
    th = 0.15
  th = -1.0
  r = 0
  for dircn, dy, dx in zip(['down', 'up', 'left', 'right'],
                           [+1,     -1,    0,      0],
                           [ 0,      0,   -1,     +1]):
    if im1_rel[dircn][0][0] > th:
      if im1_rel[dircn][0][1][0] == y1 + dy and im1_rel[dircn][0][1][1] == x1 + dx:
        r += 1
    else:
      if dircn == 'left' and x1 == 0 or \
         dircn == 'right' and x1 == params.puzzle_n_parts[0] - 1 or \
         dircn == 'up' and y1 == 0 or \
         dircn == 'down' and y1 == params.puzzle_n_parts[1] - 1:
        r += 1

  return float(r) / 4

def sort_and_keep_best(im1_rel, n_best_to_keep):
  def _sort_and_keep_one(one_rel):
    sorted_idxs = np.argsort([a[0] for a in one_rel])[-1::-1]
    r = [one_rel[i] for i in sorted_idxs[:n_best_to_keep]]
    return r

  im1_rel.down  = _sort_and_keep_one(im1_rel.down)
  im1_rel.up    = _sort_and_keep_one(im1_rel.up)
  im1_rel.right = _sort_and_keep_one(im1_rel.right)
  im1_rel.left  = _sort_and_keep_one(im1_rel.left)

def pd_vs_noise():
  pp = params.puzzle_n_parts[0]
  n_parts_at_border = pp + (pp - 1) * 2 + (pp - 2)
  n_parts_at_corners = 4
  upper_limit = (pp ** 2 - n_parts_at_border +
                 (n_parts_at_border - n_parts_at_corners) * 0.75 +
                 n_parts_at_corners * 0.5) / (pp ** 2)

  random_guess = 1 / (pp ** 2 - 1)


  all_noise = {'genady': np.arange(0, 0.50001, 0.005),
               'simple': np.arange(0, 0.50001, 0.02)}
  all_pds = {}
  net_types = ['genady', 'simple']
  legend = []
  plt.figure()
  for net_type in net_types:
    params.net.net_type = net_type
    all_pds[net_type] = []
    for noise in all_noise[net_type]:
      print(' -- > ', net_type, noise)
      pd = solve_greedy(noise=noise)
      all_pds[net_type].append(pd)
    plt.plot(all_noise[net_type], all_pds[net_type])
    legend.append(net_type)
  plt.plot(all_noise[net_type],
           np.ones_like(all_noise[net_type]) * upper_limit)
  legend.append('upper_limit')
  plt.plot(all_noise[net_type],
           np.ones_like(all_noise[net_type]) * random_guess)
  legend.append('random_guess')
  plt.xlabel('Noise')
  plt.ylabel('%')
  plt.legend(legend)
  plt.title('Pairwise analisys')
  plt.show()

if __name__ == '__main__':
  #pd_vs_noise()
  solve_greedy(show=1)