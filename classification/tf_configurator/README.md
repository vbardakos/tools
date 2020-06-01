# TensorFlow Configurator

The purpose of this module is to automate and deal
with repetitive data preprocessing operations. Also,
it provides a framework for efficient use of TensoFlow Datasets
during the training and testing phases.

## Description

The module takes every parameter needed (batch size, training steps etc)
to run a Tensorflow estimator or a Keras model and writes a `json` file
in the desired path.  

Also, it allows to write the generated Datasets from `input_fn` and
to write all the essential information, such as the shapes,
the size and the number of classes.

## Benefits

By creating and running a `config` with every information needed
we ensure that we:

 1. Do not need to navigate through different scripts to make minor changes
 2. Speed up the process by decreasing the computations during import and preprocess
 3. Achieve a greater level of abstraction in `model_fn`
 4. Have a place for everything, which decreases the debugging time needed
 5. Ensure there is proper flow control

By creating and running `*.tfrecord` files we achieve to:

 1. Run `input_fn` once and for all
 2. Serialise `input_fn` output and write data in an efficient format
 3. Remove the initial files that might be take precious space
 4. Write and read data at any stage of preprocessing
