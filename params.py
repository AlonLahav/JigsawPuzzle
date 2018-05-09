import numpy as np

np.random.seed(2)

class Struct:
  def __init__(self):
    pass

  pass


params = Struct()

params.patch_size = 16 #28
params.puzzle_n_parts = (5, 5) # x - y
params.num_epocs = 10000000
params.batch_size = 8
params.train_images_path = '/home/alon/datasets/puzzles/train'
params.test_images_path = '/home/alon/datasets/puzzles/test'
params.eval_images_path = '/home/alon/datasets/puzzles/eval'
params.max_images_per_folder = np.inf
params.logdir = '/home/alon/git-projects/JigsawPuzzle/models/7'
params.model_2_save = params.logdir + '/last_model.keras'
params.model_2_load = params.model_2_save

