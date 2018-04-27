import os

import numpy as np
import imageio
import cv2
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from params import *
import pair_wise

tfe.enable_eager_execution()

def split_image(image):
  ps = []
  for p1idx_x in range(params.puzzle_n_parts[0]):
    for p1idx_y in range(params.puzzle_n_parts[1]):
      im = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
                 p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size,:]
      ps.append(im.copy())
  return ps


def visualize(pi):
  print ' -- '
  for p in pi:
    print p[1]


eval_images = []
for fn in os.listdir(params.eval_images_path):
  if fn.endswith('jpeg'):
    im = imageio.imread(params.train_images_path + '/' + fn).astype('float32')
    im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0], params.patch_size * params.puzzle_n_parts[1]))
    im = im / im.max()
    im = im - im.mean()
    eval_images.append(im)

model = pair_wise.SimpleNet()
images = tf.constant(np.zeros((1, params.patch_size, params.patch_size, 6)).astype('float32'))
model(images, training=False)
model.load_weights('/home/alon/git-projects/puzzles/models/last_model.keras')

im_idx = 0
Pi = []
left_patches = split_image(eval_images[im_idx])
Pi.append((left_patches.pop(), (0,0)))
while len(left_patches) > 0:
  visualize(Pi)
  im2 = left_patches.pop(0)
  no_match_was_found = True
  print(len(Pi), len(left_patches))
  for patch_to_check in Pi:
    im1 = patch_to_check[0]
    images = np.concatenate([im1, im2], axis=2)[np.newaxis, :]
    logits = model(tf.constant(images), training=False).numpy()
    sigmoid_res = 1 / (1 + np.exp(-logits)).squeeze()
    #print(sigmoid_res)
    matches = np.argwhere(sigmoid_res > 0.5).squeeze()
    if np.size(matches) == 1:
      no_match_was_found = False
      if matches == 0: # patch 2 is on the left
        new_coords = (patch_to_check[1][0], patch_to_check[1][1] - 1)
        Pi.append((im2, (new_coords)))
      if matches == 1: # patch 2 is on the right
        new_coords = (patch_to_check[1][0], patch_to_check[1][1] + 1)
        Pi.append((im2, (new_coords)))
      if matches == 2: # patch 2 is on the top
        new_coords = (patch_to_check[1][0] - 1, patch_to_check[1][1])
        Pi.append((im2, (new_coords)))
      if matches == 3: # patch 2 is on the bottom
        new_coords = (patch_to_check[1][0] + 1, patch_to_check[1][1])
        Pi.append((im2, (new_coords)))
    if not no_match_was_found:
      break

  if no_match_was_found:
    left_patches.append(im1)

  if len(left_patches) + len(Pi) != 9:
    a = 1
