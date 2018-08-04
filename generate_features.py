import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import platform
from PIL import Image
from matplotlib import pyplot as plt

sys.path.append("/home/alonlahav/git-projects/models/research")
sys.path.append('/home/alonlahav/git-projects/models/research/object_detection')


class GenerateFeatures():
  def __init__(self, MODEL_NAME):
    ftr_layer = 'block2'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    if not os.path.isfile(MODEL_FILE):
      opener = urllib.request.URLopener()
      opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
      tar_file = tarfile.open(MODEL_FILE)
      for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
          tar_file.extract(file, os.getcwd())

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
      self._sess = tf.Session()
      ops = tf.get_default_graph().get_operations()
      self._ftr_tensor = [op for op in ops if op.name.lower().find(ftr_layer) != -1 and op.type.lower() == 'add'][-1].outputs[0]
      self._image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

  def load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

  def get_features(self, image):
    # Run inference
    output_dict = self._sess.run(self._ftr_tensor,
                           feed_dict={self._image_tensor: np.expand_dims(image, 0)})
    return output_dict


if __name__ == '__main__':
  MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'  # ssd_mobilenet_v1_coco_2017_11_17 / ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03
  generate_features = GenerateFeatures(MODEL_NAME)
  PATH_TO_TEST_IMAGES_DIR = 'test_images'
  TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  for image_path in TEST_IMAGE_PATHS:
    image = Image.open(image_path)
    image_np = generate_features.load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    ftrs = generate_features.get_features(image_np)

    n = 4
    plt.figure()
    plt.subplot(n, n, 1)
    plt.imshow(image)
    for i in range(1, n * n):
      plt.subplot(n, n, i + 1)
      plt.imshow(ftrs[0, :, :, i])
    plt.show()
