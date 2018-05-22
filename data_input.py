import time
import os, sys
import imageio
import cv2
import pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from params import *
import pair_wise


def crop_and_put_matrix_label(image, params):
  p1idx_y = np.random.randint(params.pred_radius, params.puzzle_n_parts[0] - params.pred_radius)
  p1idx_x = np.random.randint(params.pred_radius, params.puzzle_n_parts[1] - params.pred_radius)

  # patch1 is the reference, patch2 is the candidate
  rnd_idx = np.random.randint(0, (params.pred_radius * 2 + 1) ** 2)
  labels = np.zeros((params.pred_radius * 2 + 1, params.pred_radius * 2 + 1))
  dy, dx = np.unravel_index(rnd_idx, labels.shape)
  if dx != params.pred_radius or dy != params.pred_radius:
    labels[dy, dx] = 1
  else:
    dx = np.random.randint(params.pred_radius, params.puzzle_n_parts[1] - params.pred_radius * 2)
    dy = np.random.randint(params.pred_radius, params.puzzle_n_parts[0] - params.pred_radius * 2)
  dy -= params.pred_radius
  dx -= params.pred_radius
  p2idx_y = (p1idx_y + dy) % params.puzzle_n_parts[0]
  p2idx_x = (p1idx_x + dx) % params.puzzle_n_parts[1]


  im1 = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
              p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size, :]
  im2 = image[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
              p2idx_x * params.patch_size:(p2idx_x + 1) * params.patch_size, :]

  concat_im = np.concatenate([im1, im2], axis=2)

  return concat_im, labels

def crop_and_put_label_2(image):
  p1idx_x = np.random.randint(0, params.puzzle_n_parts[0])
  p1idx_y = np.random.randint(0, params.puzzle_n_parts[1])
  p2idx_x = np.random.randint(0, params.puzzle_n_parts[0])
  p2idx_y = np.random.randint(0, params.puzzle_n_parts[1])
  if 0:
    p1idx_x = 0
    p1idx_y = 0
    p2idx_x = 0
    p2idx_y = 1
  if p1idx_x == p2idx_x + 1 and p1idx_y == p2idx_y:
    label = [1, 0, 0, 0]  # patch2 is on the left
  elif p1idx_x == p2idx_x - 1 and p1idx_y == p2idx_y:
    label = [0, 1, 0, 0]  # patch2 is on the right
  elif p1idx_y == p2idx_y + 1 and p1idx_x == p2idx_x:
    label = [0, 0, 1, 0]  # patch2 is on the top
  elif p1idx_y == p2idx_y - 1 and p1idx_x == p2idx_x:
    label = [0, 0, 0, 1]  # patch2 is on the bottom
  else:
    label = [0, 0, 0, 0]  # far patches

  im1 = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
              p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size, :]
  im2 = image[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
              p2idx_x * params.patch_size:(p2idx_x + 1) * params.patch_size, :]

  concat_im = np.concatenate([im1, im2], axis=2)

  return concat_im, np.array(label)


def crop_and_put_label(image, est_dist_ths):
  m = 0.1
  p1idx_x = np.random.randint(0, params.puzzle_n_parts[0])
  p1idx_y = np.random.randint(0, params.puzzle_n_parts[1])
  p2idx_x = p1idx_x
  p2idx_y = p1idx_y
  rel = np.random.randint(5)
  if 0:
    p1idx_x = 0
    p1idx_y = 2
    rel = 2
  if rel == 1 and p1idx_x > 0:
    label = [1, 0, 0, 0]        # patch2 is on the left
    p2idx_x = p1idx_x - 1
    dist_est = [1 - m, 1 + m]
  elif rel == 2 and p1idx_x < params.puzzle_n_parts[0] - 1:
    label = [0, 1, 0, 0]        # patch2 is on the right
    p2idx_x = p1idx_x + 1
    dist_est = [1 - m, 1 + m]
  elif rel == 3 and p1idx_y > 0:
    label = [0, 0, 1, 0]        # patch2 is on the top
    p2idx_y = p1idx_y - 1
    dist_est = [1 - m, 1 + m]
  elif rel == 4 and p1idx_y < params.puzzle_n_parts[1] - 1:
    label = [0, 0, 0, 1]        # patch2 is on the bottom
    p2idx_y = p1idx_y + 1
    dist_est = [1 - m, 1 + m]
  else:
    label = [0, 0, 0, 0]        # far patches
    dist_est = [1.5, 10]
    while 1:
      p2idx_x = np.random.randint(1, params.puzzle_n_parts[0])
      p2idx_y = np.random.randint(1, params.puzzle_n_parts[1])
      if np.abs(p2idx_x - p1idx_x) > 0 or np.abs(p2idx_y - p1idx_y) > 0:
        break

  im1 = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
              p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size, :]
  im2 = image[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
              p2idx_x * params.patch_size:(p2idx_x + 1) * params.patch_size, :]
  concat_im = np.concatenate([im1, im2], axis=2)

  if 0:
    plt.subplot(1, 3, 1)
    plt.imshow(image+0.5)
    plt.subplot(1, 3, 2)
    plt.imshow(im1+0.5)
    plt.subplot(1, 3, 3)
    plt.imshow(im2+0.5)
    plt.suptitle(rel)
    plt.show()

  all_labels = [np.array(label)]
  if est_dist_ths:
    all_labels.append(np.array(dist_est))
  return concat_im, all_labels

def data_augmentation(im, params):
  im_res = im.copy()

  # Crop augmentation
  yb = np.random.randint(params.margin_size)
  ye = yb + params.puzzle_n_parts[0] * params.patch_size
  xb = np.random.randint(params.margin_size)
  xe = xb + params.puzzle_n_parts[1] * params.patch_size
  im_res = im_res[yb:ye, xb:xe]

  return im_res


def get_next_batch(images_list, params, est_dist_ths=False):
  im_batch = []
  all_labels_batch = []
  for _ in range(params.batch_size):
    idx = np.random.randint(len(images_list))
    after_aug = data_augmentation(images_list[idx], params)
    if params.method == 'est_dist_ths' or params.method == 'one_hot':
      images_np, all_labels = crop_and_put_label(after_aug, params)
    elif params.method == 'pred_matrix':
      images_np, all_labels = crop_and_put_matrix_label(after_aug, params)
    im_batch.append(images_np)
    all_labels_batch.append(all_labels)
  return im_batch, all_labels_batch


if __name__ == '__main__':
  pass
