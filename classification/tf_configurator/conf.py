"""
Confugurator Class
@vbar

"""
import tensorflow as tf
from _base import _ConfBase, _ConfData


class Configurator(_ConfBase, _ConfData):

    def set_config(self):
        """ sets initial configuration """
        self._conf_param()

    def set_data(self,ignore=False):
        """
        Sets data attributes in config
        Param ignore (bool):
            computes again every attribute
            ignoring current values
        """
        self._set_data(ignore)

    def get_path(self):
        """ returns the path written """
        return self.pdir

    def get_data(self):
        """ returns input & output saved in the class """
        return self.i, self.o

    def write_data(self, one_hot_labels=False, one_hot_features=False):
        """
        writes the given data.
        Params `one_hot_labels`/`one_hot_features (bool)
            applies one hot encoding before writing them
        """
        if one_hot_labels:
            self.one_hot(True)
        elif one_hot_features:
            self.one_hot(False)
        suffix = self.conf[self.name]['name'] + '.tfrecord'
        names = [self.pdir+var+suffix for var in ('x','y')]
        for var,name in zip([self.i,self.o],names):
            serialised = var.map(tf.io.serialize_tensor)
            w = tf.data.experimental.TFRecordWriter(name)
            w.write(serialised)