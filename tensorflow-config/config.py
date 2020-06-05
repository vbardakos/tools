"""
Confugurator Class
@vbar

"""
import tensorflow as tf
from _base_ import _ConfBase, _ConfData, _InOut


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


class IO(_InOut):
    """
    config(self)
    Returns `dict` Config.json

    path(self)
    Returns `str` path used
    
    data(self,train=None)
    Returns a list of Datasets
    """

    def __init__(self,*args):
        super().__init__(*args)
    
    def config(self):
        """ reads config.json from c_path """
        c = self.conf
        train = c['meta']['train']
        tlist = ['train','test']
        tlist = tlist if train else tlist.reverse()
        try:
            for t in tlist:
                for v in ('x','y'):
                    c[t][v]['dtype'] = tf.dtypes.as_dtype(c[t][v]['dtype'])
        except:
            pass
        return c

    def config_path(self):
        """ returns Config path / c_path """
        return self.c_path

    def data(self, train=None):
        """
        Read/Import datasets and returns
        them as a list.

        Params
        train (bool)
        if not specified gets val from 'meta/data'
        """
        mapper = lambda x : tf.ensure_shape(tf.io.parse_tensor(x,out_type=d),s)
        ds_lst = list()
        for f in ('x','y'):
            s,d,n = self._val_extractor(train,f)
            d  = tf.dtypes.as_dtype(d)
            ds = tf.data.TFRecordDataset(self.pdir+f+n)
            ds = ds.map(mapper)
            ds_lst.append(ds)
        return ds_lst
        
    def _val_extractor(self,train,v):
        """ Extracts shape, dtype, name """
        if isinstance(train,type(None)):
            train = self.conf['meta']['train']
        s = 'train' if train else 'test'
        name = self.conf[s]['name'] + '.tfrecord'
        vals = [self.conf[s][v][k] for k in ('shape','dtype')]
        assert all(v != None for v in vals)
        vals.append(name)
        return vals