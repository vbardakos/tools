"""
Confugurator in progress
@vbar

TO DO
    - Improve Utilities
    - add flow control
    - add Method add_data() likewise _Conf
"""
import tensorflow as tf
from _base import _ConfBase, _ConfData

class Configurator(_ConfBase, _ConfData):

    # def __init__(self, inputs, train=True, unbatch=False):
    #     if isinstance(inputs, tuple) and len(inputs) is 2:
    #         assert all(isinstance(o,tf.data.Dataset) for o in inputs)
    #         i = inputs[0]
    #         o = inputs[1]
    #     elif isinstance(inputs.element_spec, tuple) and len(inputs.element_spec):
    #         i = inputs.map(lambda x,_ : x)
    #         o = inputs.map(lambda _,y : y)
    #     else:
    #         raise ValueError('Invalid Dataset/s')

    #     self.path = _set_path()
    #     if train:
    #         self.name = lambda x : f'{self.path}{x}_train.tfrecord'
    #     else:
    #         self.name = lambda x : f'{self.path}{x}_test.tfrecord'

    #     super().__init__(i,o,unbatch)

    def one_hot(self,label=False):
        if not hasattr(self,'clses'):
            self._data_class(label)
        if label:
            self.o = self.o.map(lambda x : tf.one_hot(x, self._clses))
        else:
            self.i = self.o.map(lambda x : tf.one_hot(x, self._clses))
        return self

    def write_input(self):
        n = self.name('x')
        s = self.i.map(tf.io.serialize_tensor)
        w = tf.data.experimental.TFRecordWriter(n)
        w.write(s)

    def write_output(self):
        n = self.name('y')
        s = self.o.map(tf.io.serialize_tensor)
        w = tf.data.experimental.TFRecordWriter(n)
        w.write(s)