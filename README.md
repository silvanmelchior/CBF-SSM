# Conditional Backward/Forward SSM

This repository contains the official implementation of the CBF-SSM model presented in
[Structured Variational Inference in Unstable Gaussian Process State Space Models](https://arxiv.org/abs/1907.07035)
by Silvan Melchior, Felix Berkenkamp, Sebastian Curi, Andreas Krause.

Please cite the above paper when using this code in any way.

## Datasets

The datasets PR-SSM was already benchmarked on (Actuator, Ballbeam, Drive, Dryer,
Furnace, Sarcos) can be downloaded as described in the
[readme](https://github.com/boschresearch/PR-SSM/tree/master/datasets/real_world_tasks)
in their repo.

The remaining datasets (RoboMove, Voliro, SpringNonLinear) can be downloaded
[here](https://drive.google.com/open?id=1fBT0xdyvtnW066_FKW_fGp3NvKGPAyyt).

All datasets need to be placed in [cbfssm/datasets/data](cbfssm/datasets/data).

## Installation

To install CBF-SSM, run:

```
$ cd <path-of-repo>
$ pip3 install -e .
```

## Reproduce Paper Results

The folder [run](run) contains a script to reproduce the results for every
dataset we use to compare CBF-SSM to previous work. The results will be in a new folder
called *run_output*.

## Run Your Own Experiments

Follow these instructions to run your own experiments using CBF-SSM

### Dataset Class

At first, write a new dataset class which derives from the
[base class](cbfssm/datasets/base_ds.py). The code needs to overload `dim_u`, `dim_y` 
and the method `prepare_data` (see [example](cbfssm/datasets/dsmanager_ds.py)) s.t. it

* loads the data
* normalizes the data
* saves the data as train- and test-arrays with shape
  `[experiments, time-samples, data-dimension]`
* calls `create_batches()`

Loading of the data depends on the source of your new dataset. For normalizing the data,
there are helper functions if you have one experiment only (i.e. one long sequence),
again see [example](cbfssm/datasets/dsmanager_ds.py).

### Run File

Then, write a new run-file. You can use the [template](run/template.py) as a
starting point, which also contains a lot of comments on how to choose your parameters.
