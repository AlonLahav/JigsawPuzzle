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
import data_input

'''
TO DO:
- Fix memory(?) problem
- Check results visually
- Check filters visually
- GitHub
- Run on cloud
- Compare eager VS graph

More low priority TODOs:
- Tensorboard -> Show Graph
'''

tfe.enable_eager_execution()


def train_one_step(model, images, all_labels, optimizer):
  with tfe.GradientTape() as tape:
    logits = model(np.array(images), training=True)
    if params.method == 'pred_matrix':
      labels = np.array(all_labels)
      labels.shape = (len(all_labels), labels.shape[1] * labels.shape[2])
      loss_nbr = tf.losses.sigmoid_cross_entropy(labels, logits)
      loss_dst = 0
    elif params.method == 'est_dist_ths':
      labels = np.array([l[0] for l in all_labels])
      loss_nbr = tf.losses.sigmoid_cross_entropy(np.array(labels), logits[:, :4])
      dist_est_lbls = np.array([l[1] for l in all_labels])
      dist_est_prd = logits[:, 4:]
      loss_dst_upper = tf.reduce_mean(tf.nn.relu(-(dist_est_lbls[:, 1:]-dist_est_prd)))
      loss_dst_lower = tf.reduce_mean(tf.nn.relu( (dist_est_lbls[:, :1]-dist_est_prd)))
      loss_dst =  loss_dst_upper + loss_dst_lower
      tf.contrib.summary.scalar('loss_dst', loss_dst)
    else:
      loss_nbr = tf.losses.sigmoid_cross_entropy(np.array(labels), logits[:, :4])
    loss = loss_dst + loss_nbr
  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables), tf.train.get_or_create_global_step())

  return loss


# Get images to work on
def get_images_from_folder(folder):
  images = []
  for fn in os.listdir(folder):
    if fn.endswith('jpeg') or fn.endswith('jpg'):
      im = imageio.imread(folder + '/' + fn).astype('float32')
      if im.ndim != 3: # Use only RGB images
        continue
      im = cv2.resize(im, (params.patch_size * params.puzzle_n_parts[0] + params.margin_size, params.patch_size * params.puzzle_n_parts[1] + params.margin_size))
      im = im / im.max()
      im = im - im.mean()
      images.append(im)
      if len(images) >= params.max_images_per_folder:
        break
  return images


def train_val(params):
  train_images = get_images_from_folder(params.train_images_path)
  test_images = get_images_from_folder(params.test_images_path)

  # Init net
  if params.method == 'est_dist_ths':
    classes = 4 + 1
  elif params.method == 'pred_matrix':
    classes = (params.pred_radius * 2 + 1) ** 2
  else:
    classes = 4
  model = pair_wise.SimpleNet(params, model_fn=params.model_2_load, classes=classes)

  # Learn
  if params.action == 'train':
    optimizer = tf.train.GradientDescentOptimizer(params.learning_rate)

    summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10)
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      n_iters_to_train = int(params.num_epocs * len(train_images) / params.batch_size)
      tb = time.time()
      for itr in range(n_iters_to_train):
        # Train model
        images, all_labels = data_input.get_next_batch(train_images, params)
        loss = train_one_step(model, images, all_labels, optimizer)
        tf.contrib.summary.scalar('loss', loss)
        tf.contrib.summary.all_summary_ops()

        # Print process time
        if itr % 200 == 0:
          print (round(time.time() - tb, 1), n_iters_to_train, itr, loss.numpy())
          tb = time.time()

        # Test on train & test set
        if itr % 100 == 0:
          images, labels = data_input.get_next_batch(test_images, params)
          labels = np.array(labels)
          labels.shape = (len(all_labels), labels.shape[1] * labels.shape[2])
          logits = model(images, training=False).numpy()
          sigmoid_res = 1/(1+np.exp(-logits))
          true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
          tf.contrib.summary.scalar('accuracy', true_pred)
          print('Accuracy On Test: ' + str(true_pred))

          images, labels = data_input.get_next_batch(train_images, params)
          labels = np.array(labels)
          labels.shape = (len(all_labels), labels.shape[1] * labels.shape[2])
          logits = model(images, training=False).numpy()
          sigmoid_res = 1/(1+np.exp(-logits))
          true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
          tf.contrib.summary.scalar('train_accuracy', true_pred)
          print('Accuracy On Train: ' + str(true_pred))

        # Save model
        if itr % 1000 == 0:
          model.save_weights(params.model_2_save)
        if 0:
          tf.keras.utils.plot_model(model, to_file=params.logdir + '/model.png', show_shapes=True)


if __name__ == '__main__':
  params.action = 'train'  # 'train' / 'eval'/ 'eval-visually'
  #os.environ['CUDA_VISIBLE_DEVICES'] = ''
  train_val(params)
