# ddtn (Deep Diffeomorphic Transformer Networks)

This repo is a **Tensorflow** implementation of so called continues piecewise
affine based (CPAB) transformations by [Freifeld et al.](https://www.cs.bgu.ac.il/~orenfr/papers/freifeld_etal_PAMI_2017). We use these
transformations to create a more flexible spatial transformer layer, than the
original layer by [Jadenberg et al.](https://arxiv.org/abs/1506.02025). The code is
written using both Tensorflows c++ and python API. Thus the main contribution
of this repo is a so implementation of a ST-layer with diffeomorphic transformations. 
However, the repo also contains code for other kinds of ST-layers. The following
transformation models are included:

* Affine transformations
* Diffiomorphic affine transformations
* Homography transformations
* CPAB transformations
* Thin-Plate-Spline (TPS) transformations

The code is based upon the original implementation of the CPAB transformations by
Oren Freifeld (Github repo: [cpabDiffeo](https://github.com/freifeld/cpabDiffeo)).
Additionally, some of the code for doing interpolation is based upon the Tensorflow
implementation of the Spatial Transformer Networks (Github repo: 
[spatial-transformer-network](https://github.com/kevinzakka/spatial-transformer-network)).

## Author of this software

Nicki Skafte Detlefsen (email: nsde@dtu.dk)

## License
This software is released under the MIT License (included with the software). 
Note, however, that using this code (and/or the results of running it) to support
any form of publication (e.g., a book, a journal paper, a conference papar, a patent
application ect.) we request you to cite the following papers:

```
[1] @article{detlefsen2018transformations,
  title = {Deep Diffeomorphic Transformer Networks},
  author = {Nicki Skafte Detlefsen and Oren Freifeld and S{\o}ren Hauberg},
  journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018},
  publisher={IEEE}
}

[2] @article{freifeld2017transformations,
  title={Transformations Based on Continuous Piecewise-Affine Velocity Fields},
  author={Freifeld, Oren and Hauberg, Soren and Batmanghelich, Kayhan and Fisher, John W},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2017},
  publisher={IEEE}
}
```

## Requirements

* Generic python packages: numpy, scipy, matplotlib
* Tensorflow
* To use the GPU implementation, you need a nvidia GPU and CUDA + cuDNN installed. 
  See [Tensorflows GPU installation instructions](https://www.tensorflow.org/install/) 
  for more details
  
The code was testes running python 3.6, tensorflow 1.8.0 and CUDA 9.0. However, 
the code should be compatible with tensorflow from version 1.4.0.

The code should run on all operating system with or without an GPU. If you are on
Linux or MAC and tensorflow is able to detect your GPU then the fast GPU version
of the CPAB transformations will be uses. If you are on windows or do have an GPU,
an slower (and memory consuming) implementation will be used that is written in
pure tensorflow.

## Installation

1. Clone this reposatory to a directory of your choice
```
git clone https://github.com/SkafteNicki/ddtn
```
2. Add this directory to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/ddtn
```
3. (optional) If you want to use the fast GPU version their is a good chance that
you have to recompile the dynamic libraries where the operation is defined. Go to
the folder 'ddtn/ddtn/cuda/' and type the following in a command prompt
```
make clean
make
```

## Using the code

Try opening a python command promt and type in the follow command
```
import ddtn
```
This should give you one of the following outputs
```
----------------------------------------------------------------------
Operating system: linux
Using the fast cuda implementation for CPAB
----------------------------------------------------------------------
or
----------------------------------------------------------------------
Operating system: linux
Using the slow pure tensorflow implementation for CPAB
----------------------------------------------------------------------
```
The reposatory comes with two scripts you can try to run
``` 
python play_with_transformers.py <- show what the different transformers can do
python mnist_classifier.py <- trains a classifier on a distorted mnist dataset
```
for both script you can type `-h` after to get the commandline options.

To use the different transformers in your own settings, there are primarily 3
files that are important

1. `ddtn.transformers.transformers` 
   
   Defines methods called tf_'name'_transformer eg. tf_Affine_transformer, tf_CPAB_transformer ect. that as input takes a grid of points and a parametrization and outputs the transformed grid.

2. `ddtn.transformers.transformer_layers`

   Defines methods called ST_'name'_transformer eg. ST_Affine_transformer, ST_CPAB_transformer ect. that as input takes an image, a parametrization and output the transformed image.

3. `ddtn.transformers.keras_layers` 

   Defined keras layers called Spatial'name'Layer eg. SpatialAffineLayer, SpatialCPABLayer ect. that can be incorporated into keras models (main used high-level-api for tensorflow).

## Known bugs
1. *"Executor failed to create kernel. Not found: Op type not registered 
'tf_CPAB_transformer' in binary running on HedonismeBot. Make sure the Op and 
Kernel are registered in the binary running in this process"*.

   `tf.Defun` seems to be working but still give out this error message. Should
    probably try to find a replacement for `tf.Defun`. Has something to do with
    the fact that tensorflow sessions freeze the current graph. Look into
    `tf.RegisterGradient` at some point.

