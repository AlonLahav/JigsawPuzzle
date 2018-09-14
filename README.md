# JigsawPuzzle
## Install TensorFlow
(Followed by https://yangcha.github.io/CUDA90/)
```
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev
```

- Reboot the system to load the NVIDIA drivers.

- Set up the development environment by modifying the PATH and LD_LIBRARY_PATH variables:
```
gedit ~/.bashrc
```

Now add them to the end of .bashrc file:
```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## GoogleAPI
### Bypass resize
At `image_resizer_builder.py` change `image_resizer_config.keep_aspect_ratio_resizer` to `preprocessor.resize_bypass`
At `preprocessor.py` add the following:
```
def resize_bypass(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
   if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      new_size = _compute_new_static_size(image, min_dimension, max_dimension)
    else:
      new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
    new_image = image

    result = [new_image]
    result.append(new_size)
    return result
```
### Export new .pb file
```
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=/home/alonlahav/git-projects/faster_rcnn_resnet50_coco_2018_01_28/pipeline.config
TRAINED_CKPT_PREFIX=/home/alonlahav/git-projects/faster_rcnn_resnet50_coco_2018_01_28/model.ckpt
EXPORT_DIR=/home/alonlahav/git-projects/faster_rcnn_resnet50_coco_2018_01_28/export 
cd git-projects/models/research/
export PYTHONPATH=/home/alonlahav/git-projects/models/slim:/home/alonlahav/git-projects/models:/home/alonlahav/git-projects/models/research
python3 object_detection/export_inference_graph.py     --input_type=${INPUT_TYPE}     --pipeline_config_path=${PIPELINE_CONFIG_PATH}     --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX}     --output_directory=${EXPORT_DIR}

```



























