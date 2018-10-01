import numpy as np
from easydict import EasyDict
import getpass

np.random.seed(2)

params = EasyDict()

params.patch_size = 28
params.puzzle_n_parts = (10, 10) # x - y
params.margin_size = 15
params.num_epocs = 10000000
params.batch_size = 8
if getpass.getuser() == 'alon':
  params.train_images_path = '/home/alon/datasets/mscoco/val2017'
  params.test_images_path = '/home/alon/datasets/mscoco/val2017'
  params.eval_images_path = '/home/alon/datasets/mscoco/val2017'
  params.logdir = '/home/alon/git-projects/JigsawPuzzle/models/28-ok-'
else:
  params.train_images_path = '/home/alonlahav/datasets/mscoco/train2017'
  params.test_images_path = '/home/alonlahav/datasets/mscoco/test2017'
  params.eval_images_path = '/home/alonlahav/datasets/mscoco/val2017'
  params.logdir = '/home/alonlahav/git-projects/JigsawPuzzle/models/24-simple-net'
params.max_images_per_folder = 5000 # np.inf
params.load_images_to_memory = 0
params.model_2_save = params.logdir + '/last_model.keras'
params.model_2_load = params.model_2_save
params.pred_radius = 1
params.method = 'pred_matrix' #  pred_matrix / est_dist_ths / one_hot
params.learning_rate = .01 / params.batch_size
params.preprocess = 'mean-0' # None , 'mean-0'
params.net = EasyDict()
params.net.net_type = 'simple' # simple / net-on-net
params.net.only_fc = True
params.net.num_fc_layers = 4
params.net.features_layer = 'block2/unit_3'