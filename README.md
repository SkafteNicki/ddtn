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
* Homography transformations
* CPAB transformations
* Thin-Plate-Spline (TPS) transformations

The code is based upon the original implementation of the CPAB transformations by
Oren Freifeld (Github repo: [cpabDiffeo](https://github.com/freifeld/cpabDiffeo)).
Additionall, some of the code for doing interpolation is based upon the Tensorflow
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

The code are only tested on Linux, but should also work on MAC. The code cannot 
run on Windows at the moments, since the compiled dynamic libraries (.so files) 
are only combiled for UNIX. If comes up with a way to compile these for windows, 
please let us know.

## Installation

Clone this reposatory to a directory of your choice
```
git clone https://github.com/SkafteNicki/ddtn
```
Add this directory to your PYTHONPATH
```
export PYTHONPATH=$PYTHONPATH:$YOUR_FOLDER_PATH/ddtn
```

## Running code


## Known bugs
1 "Executor failed to create kernel. Not found: Op type not registered 
'tf_CPAB_transformer' in binary running on HedonismeBot. Make sure the Op and 
Kernel are registered in the binary running in this process"
   `tf.Defun` seems to be working but still give out this error message. Should
    probably try to find a replacement for `tf.Defun`
   - This error has something to do with `tf.Defun`, but the code seems to run.
   It is probably due to the fact that tensorflow sessions freeze the current
   graph. Will try to find a replacement for `tf.Defun`. Look into `tf.RegisterGradient`
   at some point.


