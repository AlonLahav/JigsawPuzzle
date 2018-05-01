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

'''
TO DO:
- Check results visually
- Check filters visually
- GitHub
- Run on cloud
- Compare eager VS graph

More low priority TODOs:
- Tensorboard -> Show Graph
'''

tfe.enable_eager_execution()


def train_one_step(model, images, labels, optimizer):
  with tfe.GradientTape() as tape:
    logits = model(images, training=True)
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), tf.train.get_or_create_global_step())

  return loss

def crop_and_put_label_2(image):
  p1idx_x = np.random.randint(1, params.puzzle_n_parts[0])
  p1idx_y = np.random.randint(1, params.puzzle_n_parts[1])
  p2idx_x = np.random.randint(1, params.puzzle_n_parts[0])
  p2idx_y = np.random.randint(1, params.puzzle_n_parts[1])
  if p1idx_x == p2idx_x - 1 and p1idx_y == p2idx_y:
    label = [1, 0, 0, 0]  # patch2 is on the left
  elif p1idx_x == p2idx_x + 1 and p1idx_y == p2idx_y:
    label = [0, 1, 0, 0]  # patch2 is on the right
  elif p1idx_y == p2idx_y - 1 and p1idx_x == p2idx_x:
    label = [0, 0, 1, 0]  # patch2 is on the top
  elif p1idx_y == p2idx_y + 1 and p1idx_x == p2idx_x:
    label = [0, 0, 0, 1]  # patch2 is on the bottom
  else:
    label = [0, 0, 0, 0]  # far patches

  im1 = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
              p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size,:]
  im2 = image[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
              p2idx_x * params.patch_size:(p2idx_x + 1) * params.patch_size,:]

  concat_im = np.concatenate([im1, im2], axis=2)

  return concat_im, np.array(label)


def crop_and_put_label(image):
  p1idx_x = np.random.randint(1, params.puzzle_n_parts[0])
  p1idx_y = np.random.randint(1, params.puzzle_n_parts[1])
  p2idx_x = p1idx_x
  p2idx_y = p1idx_y
  rel = np.random.randint(5)
  if 0:
    p1idx_x = 1
    p1idx_y = 0
    rel = 2
  if   rel == 1 and p1idx_x > 0:
    label = [1, 0, 0, 0]        # patch2 is on the left
    p2idx_x = p1idx_x - 1
  elif rel == 2 and p1idx_x < params.puzzle_n_parts[0] - 1:
    label = [0, 1, 0, 0]        # patch2 is on the right
    p2idx_x = p1idx_x + 1
  elif rel == 3 and p1idx_y > 0:
    label = [0, 0, 1, 0]        # patch2 is on the top
    p2idx_y = p1idx_y - 1
  elif rel == 4 and p1idx_y < params.puzzle_n_parts[1] - 1:
    label = [0, 0, 0, 1]        # patch2 is on the bottom
    p2idx_y = p1idx_y + 1
  else:
    label = [0, 0, 0, 0]        # far patches
    while 1:
      p2idx_x = np.random.randint(1, params.puzzle_n_parts[0])
      p2idx_y = np.random.randint(1, params.puzzle_n_parts[1])
      if np.abs(p2idx_x - p1idx_x) > 1 or np.abs(p2idx_y - p1idx_y) > 1:
        break

  im1 = image[p1idx_y * params.patch_size:(p1idx_y + 1) * params.patch_size,
              p1idx_x * params.patch_size:(p1idx_x + 1) * params.patch_size,:]
  im2 = image[p2idx_y * params.patch_size:(p2idx_y + 1) * params.patch_size,
              p2idx_x * params.patch_size:(p2idx_x + 1) * params.patch_size,:]
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

  return concat_im, np.array(label)

def get_next_batch(images_list):
  im_batch = []
  lb_batch = []
  for _ in range(params.batch_size):
    idx = np.random.randint(len(images_list))
    images_np, labels_np = crop_and_put_label_2(images_list[idx])
    im_batch.append(images_np)
    lb_batch.append(labels_np)
  return tf.constant(im_batch), tf.constant(lb_batch)


# Get images to work on
def get_images_from_folder(folder):
  images = []
  for fn in os.listdir(params.train_images_path):
    if fn.endswith('jpeg'):
      im = imageio.imread(params.train_images_path + '/' + fn).astype('float32')
      im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0], params.patch_size * params.puzzle_n_parts[1]))
      im = im / im.max()
      im = im - im.mean()
      images.append(im)
  return images

def train_val():
  train_images = get_images_from_folder(params.train_images_path)
  test_images  = get_images_from_folder(params.test_images_path)

  # Init net
  model = pair_wise.SimpleNet(model_fn='/home/alon/git-projects/JigsawPuzzle/models/1/last_model.keras')

  # Learn
  if params.action == 'train':
    all_loss = []
    all_true_pred = []
    optimizer = tf.train.GradientDescentOptimizer(0.1 / params.batch_size)

    summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      n_iters_to_train = params.num_epocs * len(train_images) / params.batch_size
      tb = time.time()
      for iter in range(n_iters_to_train):
        images, labels = get_next_batch(train_images)
        loss = train_one_step(model, images, labels, optimizer)
        tf.contrib.summary.scalar('loss', loss)
        tf.contrib.summary.all_summary_ops()
        all_loss.append(loss.numpy())

        if iter % 200 == 0:
          print (round(time.time() - tb, 1), n_iters_to_train, iter, loss.numpy())
          tb = time.time()
        if iter % 100 == 0:
          images, labels = get_next_batch(test_images)
          logits = model(images, training=False).numpy()
          sigmoid_res = 1/(1+np.exp(-logits))
          true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
          all_true_pred.append(true_pred)
          tf.contrib.summary.scalar('accuracy', true_pred)
          print('Accuracy On Test: ' + str(true_pred))
        if iter % 1000 == 0:
          model.save_weights('/home/alon/git-projects/JigsawPuzzle/models/1/last_model.keras')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(all_loss)
    plt.subplot(1, 2, 2)
    plt.plot(all_true_pred)
    plt.show()

  if params.action == 'eval':
    images, labels = get_next_batch(test_images)
    logits = model(images, training=False).numpy()
    sigmoid_res = 1/(1+np.exp(-logits))
    true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
    print('Accuracy On Test: ' + str(true_pred))


if __name__ == '__main__':
  params.action = 'eval'  # 'train' / 'eval'/ 'eval-visually'
  train_val()