import numpy as np
from easydict import EasyDict

np.random.seed(2)

params = EasyDict()

params.patch_size = 64 #28
params.puzzle_n_parts = (5, 5) # x - y
params.margin_size = 15
params.num_epocs = 10000000
params.batch_size = 32
params.train_images_path = '/home/alonlahav/datasets/mscoco/train2017'
params.test_images_path = '/home/alonlahav/datasets/mscoco/test2017'
params.eval_images_path = '/home/alonlahav/datasets/mscoco/val2017'
params.max_images_per_folder = 5000 # np.inf
params.load_images_to_memory = 0
params.logdir = '/home/alonlahav/git-projects/JigsawPuzzle/models/24-simple-net'
params.model_2_save = params.logdir + '/last_model.keras'
params.model_2_load = params.model_2_save
params.pred_radius = 1
params.method = 'pred_matrix' #  pred_matrix / est_dist_ths / one_hot
params.learning_rate = .01 / params.batch_size
params.preprocess = None # None , 'mean-0'
params.net = EasyDict()
params.net.net_type = 'simple' # simple / net-on-net
params.net.only_fc = False
params.net.features_layer = 'block2/unit_3'