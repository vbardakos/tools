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

## Example

```python
from conf import Configurator
from reader import IO

> Configurator.reset() # deletes previously written configurations & data

""" SET PATH, PARAMS AND CONFIG """
> Configurator.add_path('sample_path/') # first, declare path for config & data
> Configurator().add_batch(64).add_steps(0).add_train(True).set_config() # we add batch/steps/train, and set the config
>
> IO().config_path()
'sample_path/.conf.json'

""" SET DATA, DATA PARAMS AND WRITE THEM """
import tensorflow_datasets as tfds

> dataset = tfds.load('mnist', split='train',as_supervised=True,download=False,shuffle_files=False)
> print(dataset)
<_OptionsDataset shapes: ((28, 28, 1), ()), types: (tf.uint8, tf.int64)>
>
> Configurator.add_data(dataset).set_data() # add our dataset and set params in the config
> Configurator().write_data(one_hot_labels=True) # write our data & applies one-hot encoding to labels

""" IMPORT OUR DATA AND CONFIG """
> my_config = IO().config() # exports config as a dictionary
> x_train,y_train  = IO().data() # exports our data as a list
>
> [print(label) for label in y_train.take(2)] # y_train got one-hot encoded before written
tf.Tensor([0. 0. 0. 1. 0. 0. 0. 0. 0. 0.], shape=(10,), dtype=float32)
tf.Tensor([0. 0. 0. 0. 1. 0. 0. 0. 0. 0.], shape=(10,), dtype=float32)
>
> my_config
{
    'train': {
        'name': '_train',
        'batch': 64,
        'steps': 937,
        'size': 60000,
        'x': {'shape': [28, 28, 1], 'dtype': tf.uint8, 'class': 28},
        'y': {'shape': [10], 'dtype': tf.float32, 'class': 10}
        },
    'test': {
        'name': '_test',
        'size': None,
        'x': {'shape': None, 'dtype': None, 'class': None},
        'y': {'shape': None, 'dtype': None, 'class': None}
        },
    'meta': {'steps': 'Auto', 'train': True}
}
```

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
