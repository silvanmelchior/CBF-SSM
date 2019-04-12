# Conditional Backward/Forward SSM

This repository contains the official implementation of the CBF-SSM model presented in
[Non-Factorized Variational Inference in Unstable Gaussian Process State Space Models](https://arxiv.org/)
by Silvan Melchior et al., 2019. (TODO: adjust link, add conference)

Please cite the above paper when using this code in any way.

## Dependencies

The code depends on (tested version): 
* tensorflow (tensorflow-gpu 1.5.0)
* matplotlib (3.0.0)
* scipy (1.1.0)

## Datasets

The datasets PR-SSM was already benchmarked on (Actuator, Ballbeam, Drive, Dryer,
Furnace, Sarcos) can be downloaded as described in the
[readme](https://github.com/boschresearch/PR-SSM/tree/master/datasets/real_world_tasks)
in their repo. TODO: robomove, springnonlinear, voliro.

All datasets need to be placed in [datasets/data](datasets/data).

## Python Path

The script [set_path.sh](set_path.sh) sets $PYTHONPATH in a linux bash console s.t.
the import statements of all scripts are correct. It needs to be sourced:

```
$ cd <path-of-repo>
$ . set_path.sh
$ python3 /cbfssm/run/<script-name>.py
```

## Reproduce Paper Results

The folder [cbfssm/run](cbfssm/run) contains a script to reproduce the results for every
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

Then, write a new run-file. You can use the [template](cbfssm/run/template.py) as a
starting point, which also contains a lot of comments on how to choose your parameters.
