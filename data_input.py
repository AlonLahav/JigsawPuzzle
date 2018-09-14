import time
import os, sys

from scipy.spatial import Voronoi, voronoi_plot_2d
import imageio
import cv2
import pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import triangle
import triangle.plot as tr_plot

from params import *
import pair_wise


def crop_and_put_matrix_label(image1, prev_image, params):
  p1idx_y = np.random.randint(params.pred_radius, params.puzzle_n_parts[0] - params.pred_radius + 1)
  p1idx_x = np.random.randint(params.pred_radius, params.puzzle_n_parts[1] - params.pred_radius + 1)

  # patch1 is the reference, patch2 is the candidate
  rnd_idx = np.random.randint(0, (params.pred_radius * 2 + 1) ** 2)
  labels = np.zeros((params.pred_radius * 2 + 1, params.pred_radius * 2 + 1))
  dy, dx = np.unravel_index(rnd_idx, labels.shape)
  if dx != params.pred_radius or dy != params.pred_radius:
    labels[dy, dx] = 1
    image2 = image1
  else:
    # Get different image and crop randomaly
    p1idx_y = np.random.randint(params.pred_radius, params.puzzle_n_parts[0] - params.pred_radius + 1)
    p1idx_x = np.random.randint(params.pred_radius, params.puzzle_n_parts[1] - params.pred_radius + 1)
    image2 = prev_image

  dy -= params.pred_radius
  dx -= params.pred_radius
  p2idx_y = (p1idx_y + dy) % params.puzzle_n_parts[0]
  p2idx_x = (p1idx_x + dx) % params.puzzle_n_parts[1]


  if image1 is None:
    concat_im = None
  else:
    im1 = image1[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
                p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size, :]
    im2 = image2[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
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

prev_image = None
def get_next_batch(images_list, params, est_dist_ths=False):
  global prev_image
  im_batch = []
  all_labels_batch = []
  time_log = [0, 0, 0]
  if len(images_list) == 0:
    prev_image = after_aug = None
  else:
    idx = np.random.randint(len(images_list))
    if params.load_images_to_memory:
      im = images_list[idx]
    else:
      read_beg = time.time()
      im = imageio.imread(images_list[idx]).astype('float32')
      time_log[0] += time.time() - read_beg
      if 1:
        tb = time.time()
        im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0] + params.margin_size,
                             params.patch_size * params.puzzle_n_parts[1] + params.margin_size))
        time_log[1] += time.time() - tb
      else:
        im = im[:params.patch_size * params.puzzle_n_parts[0] + params.margin_size,
             :params.patch_size * params.puzzle_n_parts[1] + params.margin_size, :]
      if params.preprocess == 'mean-0':
        im = im / im.max()
        im = im - im.mean()
    after_aug = data_augmentation(im, params)
    if prev_image is None:
      prev_image = np.zeros_like(im)
  for _ in range(params.batch_size + (prev_image is None)):
    if params.method == 'est_dist_ths' or params.method == 'one_hot':
      images_np, all_labels = crop_and_put_label(after_aug, params)
    elif params.method == 'pred_matrix':
      images_np, all_labels = crop_and_put_matrix_label(after_aug, prev_image, params)
    im_batch.append(images_np)
    all_labels_batch.append(all_labels)
    prev_image = after_aug
  return im_batch, all_labels_batch, time_log


def get_images_from_folder(folder):
  print ('Getting images from folder: ' + folder)
  images = []
  for fn in os.listdir(folder):
    if len(images) % 100 == 0:
      print('.', end='', flush=True)
    if fn.endswith('jpeg') or fn.endswith('jpg'):
      im = imageio.imread(folder + '/' + fn).astype('float32')
      if im.ndim != 3:  # Use only RGB images
        continue
      if im.shape[0] < params.patch_size * params.puzzle_n_parts[0] + params.margin_size:
        continue
      if im.shape[1] < params.patch_size * params.puzzle_n_parts[1] + params.margin_size:
        continue
      if params.load_images_to_memory:
        im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0] + params.margin_size, params.patch_size * params.puzzle_n_parts[1] + params.margin_size))
        if params.preprocess == 'mean-0':
          im = im / im.max()
          im = im - im.mean()
      else:
        im = folder + '/' + fn
      images.append(im)
      if len(images) >= params.max_images_per_folder:
        break
  print (' -- Done.\n')
  return images


def unstructured_cut(im):
  voronoi = 1
  n = 40
  ptx = np.random.randint(0, im.shape[1], size=(n, 1))
  pty = np.random.randint(0, im.shape[0], size=(n, 1))
  pts = np.hstack((ptx, pty))
  if voronoi:
    vor = Voronoi(pts)
  else:
    t = triangle.triangulate({'vertices': pts})
  plt.figure()
  ax = plt.subplot(111, aspect='equal')
  voronoi_plot_2d(vor, ax)
  plt.imshow(im[::-1,:])
  plt.show()


if __name__ == '__main__1':
  params.max_images_per_folder = 5
  train_images = get_images_from_folder(params.train_images_path)
  im = train_images[1] - train_images[1].min()
  im /= im.max() / 255
  im = im.astype('uint8')
  unstructured_cut(im)

if __name__ == '__main__':
  acc = np.zeros((3, 3))
  for _ in range(1000):
    _, all_labels_batch = get_next_batch([], params)
    acc += np.sum(np.array(all_labels_batch), axis=0)
    print(acc)