import numpy as np

np.random.seed(0)

class Struct:
  def __init__(self):
    pass

  pass


params = Struct()

params.patch_size = 16 #28
params.puzzle_n_parts = (5, 5) # x - y
params.num_epocs = 100000
params.batch_size = 8
params.train_images_path = '/home/alon/datasets/puzzles/train'
params.test_images_path = '/home/alon/datasets/puzzles/test'
params.eval_images_path = '/home/alon/datasets/puzzles/eval'
params.logdir = '/home/alon/git-projects/JigsawPuzzle/models/1'

