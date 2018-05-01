import os
import random
import time

import numpy as np
import pylab as plt
import imageio
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from params import *
import pair_wise
import run_train_val

if not tf.executing_eagerly():
  tfe.enable_eager_execution()

def split_image(image):
  ps = []
  order = []
  xs = range(params.puzzle_n_parts[0])
  ys = range(params.puzzle_n_parts[1])
  if 1:
    random.shuffle(xs)
    random.shuffle(ys)
  for p1idx_x in xs:
    for p1idx_y in ys:
      im = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
                 p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size,:]
      order.append((p1idx_y, p1idx_x))
      ps.append(im.copy())
  return ps, order


def visualize(org_image, split_order, pi):
  if 1:
    plt.figure()
    plt.imshow(org_image - np.min(org_image))
    for idx, yx in enumerate(split_order):
      plt.text((yx[1] + .5) * params.patch_size, (yx[0] + 0.5) * params.patch_size, idx, bbox=dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8),))
    print ' -- '

  all_pos = np.array([(p[1][0], p[1][1]) for p in pi])
  minx = int(np.floor(np.min(all_pos[:, 1])))
  maxx = int(np.ceil(np.max(all_pos[:, 1])))
  miny = int(np.floor(np.min(all_pos[:, 0])))
  maxy = int(np.ceil(np.max(all_pos[:, 0])))
  im2show = np.zeros(((maxy - miny + 1) * params.patch_size,
                      (maxx - minx + 1) * params.patch_size,
                      3))

  for p in pi:
    print p[1]
    xb = (p[1][1] - minx) * params.patch_size
    yb = (p[1][0] - miny) * params.patch_size
    im2show[yb:yb+params.patch_size, xb:xb+params.patch_size] = p[0]

  plt.figure()
  plt.imshow(im2show - np.min(im2show))
  plt.show()


eval_images = run_train_val.get_images_from_folder(params.eval_images_path)

model = pair_wise.SimpleNet(model_fn=params.model_2_load)

#if 1:
while 1:
  images, labels = run_train_val.get_next_batch(eval_images)
  logits = model(images, training=False).numpy()
  sigmoid_res = 1/(1+np.exp(-logits))
  true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
  print('Accuracy On Test: ' + str(true_pred))
  if 0:#true_pred != 100:
    print(np.round(sigmoid_res))
    print(labels)

  #exit(0)

im_idx = 3
Pi = []
im2use = eval_images[im_idx]
left_patches, split_order = split_image(im2use)
Pi.append((left_patches.pop(0), (0,0)))
while len(left_patches) > 0:
  im2 = left_patches.pop(0)
  no_match_was_found = True
  for idx, patch_to_check in enumerate(Pi):
    im1 = patch_to_check[0]
    images = np.concatenate([im1, im2], axis=2)[np.newaxis, :]
    logits = model(tf.constant(images), training=False).numpy()
    sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
    #print(sigmoid_res)
    matches = np.argwhere(sigmoid_res > 0.5).squeeze()
    if np.size(matches) == 1:
      no_match_was_found = False
      if matches == 0: # patch 2 is on the left
        new_coords = (patch_to_check[1][0], patch_to_check[1][1] - 1, idx)
        Pi.append((im2, (new_coords)))
      if matches == 1: # patch 2 is on the right
        new_coords = (patch_to_check[1][0], patch_to_check[1][1] + 1, idx)
        Pi.append((im2, (new_coords)))
      if matches == 2: # patch 2 is on the top
        new_coords = (patch_to_check[1][0] - 1, patch_to_check[1][1], idx)
        Pi.append((im2, (new_coords)))
      if matches == 3: # patch 2 is on the bottom
        new_coords = (patch_to_check[1][0] + 1, patch_to_check[1][1], idx)
        Pi.append((im2, (new_coords)))
    if not no_match_was_found:
      #visualize(eval_images[im_idx], split_order, Pi)
      break

  if no_match_was_found:
    left_patches.append(im2)

visualize(im2use, split_order, Pi)