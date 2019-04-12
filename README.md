# Conditional Backward/Forward SSM

This repository contains the official implementation of the CBF-SSM model presented in
"Non-Factorized Variational Inference in Unstable Gaussian Process State Space Models"
by Silvan Melchior et al., 2019. The paper can be found here: TODO.

Please cite the above paper when using this code in any way.

## Dependencies

The code depends on (brackets: version code was tested with) 
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
the import statements of all scripts are correct:

```
$ cd <path-of-repo>
$ . set_path.sh
$ python3 /cbfssm/run/<script-name>.py
```

## Reproduce Paper Results

The folder [cbfssm/run](cbfssm/run) contains a script to reproduce the results of every
dataset we use to compare CBF-SSM to previous work.

## Run Your Own Experiments

The following explanations should help to run your own experiments using CBF-SSM

### Dataset Class

At first, a new dataset class needs to be written which derives from the
[base class](cbfssm/datasets/base_ds.py). The code needs to overload `dim_u`, `dim_y` 
and the method `prepare_data` (see [example](cbfssm/datasets/dsmanager_ds.py)) s.t. it

* loads the data
* normalizes the data
* saves the data as train- and test-arrays with shape
  `[experiments, time-samples, data-dimension]`
* calls `create_batches()`

Loading of the data depends on the source of your new dataset. For normalizing the data,
there are helper functions if you have one experiment only (i.e. one long sequence)

### Run File

Then, a new run-file needs to be written which can later be exectuted. Use the
[template run-file](cbfssm/run/template.py) as a starting point. It also contains
a lot of hints how to choose good initial estimates for your parameters.
