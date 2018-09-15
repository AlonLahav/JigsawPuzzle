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

'''
TO DO:
- SGD -> momentum / ADAM
- increase radius size  

- Check filters visually
- Compare eager VS graph

More low priority TODOs:
- Tensorboard -> Show Graph
'''

tfe.enable_eager_execution()

def visualize_one_minibatch(images, pred_labels, pred_score, gt_labels=None):
  f = 0
  if gt_labels is None:
    gt_labels = [None] * len(pred_labels)
  for im_idx in range(8):
    img = images[im_idx]
    score = pred_score[im_idx]
    gt = gt_labels[im_idx]
    f += 1
    plt.figure(f)
    plt.subplot(3,3,5)
    plt.imshow(img[:, :, :3].astype('int16'))
    g = np.argwhere(gt.flatten()).squeeze() + 1
    for i in range(9):
      plt.subplot(3, 3, i + 1)
      if g == i + 1:
        gt_str = '+'
      else:
        gt_str = '-'
      plt.title(gt_str + str(np.round(score[i] * 100)))
      if score[i] > 0.5:
        plt.imshow(img[:, :, 3:].astype('int16'))
      plt.axis('off')
    if 0:
      r = np.argwhere(lbl.flatten()).squeeze()+1
      g = np.argwhere(gt.flatten()).squeeze()+1
      #if r == g:
      #  continue
      #if r.size == 0:
      #  continue
      if isinstance(r, np.int64):
        plt.subplot(3,3,r)
        plt.imshow(img[:, :, 3:].astype('int16'))
      ttl_str = 'r = ' + str(r) + ' , g = ' + str(g)
      if r.size > 1 or g != r or not isinstance(g, np.int64):
        if isinstance(g, np.int64):
          plt.subplot(3, 3, g)
          plt.imshow(img[:, :, 3:].astype('int16'), alpha=0.5)
      plt.suptitle(ttl_str)
    plt.show()

def train_one_step(model, images, all_labels, optimizer):
  with tfe.GradientTape() as tape:
    logits = model(np.array(images), training=True, visualize=0)
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


def calc_pred_accurances(labels, sigmoid_res, n_occurances, n_arg_max, sum_predictions):
  if n_occurances is None:
    n_occurances = np.zeros_like(labels[0])
    n_arg_max = np.zeros_like(labels[0])
    sum_predictions = np.zeros_like(labels[0])
  for n in range(labels.shape[0]):
    if labels[n].any():
      lbl = np.argmax(labels[n])
      n_occurances[lbl] += 1
      prd = np.argmax(sigmoid_res[n])
      if lbl == prd:
        n_arg_max [lbl] += 1
      sum_predictions[lbl] += sigmoid_res[n][lbl]

  return n_occurances, n_arg_max, sum_predictions


def train_val(params):
  train_images = data_input.get_images_from_folder(params.train_images_path)
  test_images = data_input.get_images_from_folder(params.test_images_path)

  global_step = tf.train.get_or_create_global_step()
  try:
    with open(params.logdir + '/solver.txt') as f:
      n = f.read()
    global_step.assign(int(n))
  except:
    shutil.copyfile(os.path.split(__file__)[0] + '/params.py', params.logdir + '/params.py')

  # Init net
  if params.method == 'est_dist_ths':
    classes = 4 + 1
  elif params.method == 'pred_matrix':
    classes = (params.pred_radius * 2 + 1) ** 2
  else:
    classes = 4
  if params.net.net_type == 'simple':
    model = pair_wise.SimpleNet(params, model_fn=params.model_2_load, classes=classes)
  else:
    model = pair_wise.NetOnNet(params, model_fn=params.model_2_load, classes=classes)

  n_labels = (params.pred_radius * 2 + 1) ** 2
  n_occurances_trn = n_arg_max_trn = sum_predictions_trn =  n_occurances_tst = n_arg_max_tst = sum_predictions_tst = None

  # Learn
  optimizer = tf.train.AdamOptimizer(params.learning_rate)

  summary_writer = tf.contrib.summary.create_file_writer(params.logdir, flush_millis=10)
  save_summary_each = 1
  with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
    n_iters_to_train = int(params.num_epocs * len(train_images) / params.batch_size)
    tb = time.time()
    for itr in range(n_iters_to_train):
      # Train model
      if 'train' in params.action:
        if 1:
          get_input_beg = time.time()
          images, all_labels, time_log = data_input.get_next_batch(train_images, params)
          get_input_time = time.time() - get_input_beg
          itr_beg = time.time()
          loss = train_one_step(model, images, all_labels, optimizer)
          train_step_time = time.time() - itr_beg
        if itr % save_summary_each == 0:
          tf.contrib.summary.scalar('loss', loss)
          tf.contrib.summary.scalar('timing/train_step_time', train_step_time)
          tf.contrib.summary.scalar('timing/total_get_input_time', get_input_time)
          tf.contrib.summary.scalar('timing/read-image', time_log[0])
          tf.contrib.summary.scalar('timing/resize', time_log[1])
          tf.contrib.summary.all_summary_ops()
          if float(itr) / save_summary_each > 100 and save_summary_each < 1e3:
            save_summary_each *= 10

        # Print process time
        if itr % 200 == 0:
          print (round(time.time() - tb, 1), n_iters_to_train, itr, loss.numpy())
          tb = time.time()

        # Save model
        if itr % 1000 == 0:
          model.save_weights(params.model_2_save)
          with open(params.logdir + '/solver.txt', 'wt') as f:
            f.write(str(global_step.numpy()))

      # Test on train & test set
      if itr % 100 == 0 or 'test' in params.action:
        images, labels, _ = data_input.get_next_batch(test_images, params)
        labels = np.array(labels)
        labels.shape = (len(labels), labels.shape[1] * labels.shape[2])
        logits = model(images, training=False).numpy()
        sigmoid_res = 1/(1+np.exp(-logits))
        true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
        tf.contrib.summary.scalar('accuracy/test', true_pred)
        print('Accuracy On Test: ' + str(true_pred))
        n_occurances_tst, n_arg_max_tst, sum_predictions_tst = calc_pred_accurances(labels, sigmoid_res, n_occurances_tst, n_arg_max_tst, sum_predictions_tst)
        #visualize_one_minibatch(images, sigmoid_res>0.5, sigmoid_res, labels)

        images, labels, _ = data_input.get_next_batch(train_images, params)
        labels = np.array(labels)
        labels.shape = (len(labels), labels.shape[1] * labels.shape[2])
        logits = model(images, training=False).numpy()
        sigmoid_res = 1/(1+np.exp(-logits))
        true_pred = 100.0 * np.sum(1 - np.any((np.round(sigmoid_res) == 1) - labels, axis=1)) / params.batch_size
        tf.contrib.summary.scalar('accuracy/train', true_pred)
        print('Accuracy On Train: ' + str(true_pred))
        n_occurances_trn, n_arg_max_trn, sum_predictions_trn = calc_pred_accurances(labels, sigmoid_res, n_occurances_trn, n_arg_max_trn, sum_predictions_trn)

        if itr % (100 * 10) == 0:
          print('n_occurances_trn')
          print (n_occurances_trn.reshape((params.pred_radius * 2 + 1,params.pred_radius * 2 + 1)))
          print('n_occurances_tst')
          print (n_occurances_tst.reshape((params.pred_radius * 2 + 1,params.pred_radius * 2 + 1)))
          print('Accuracy arg-max tst')
          print ((n_arg_max_tst / n_occurances_tst.astype('float32')).reshape((params.pred_radius * 2 + 1,params.pred_radius * 2 + 1)))
          print('sum_predictions_tst')
          print ((sum_predictions_tst / n_occurances_tst.astype('float32')).reshape((params.pred_radius * 2 + 1,params.pred_radius * 2 + 1)))

if __name__ == '__main__':
  params.action = ['train'] # 'train'  / 'test' # 'train' / 'eval'/ 'eval-visually'
  #os.environ['CUDA_VISIBLE_DEVICES'] = ''
  train_val(params)
