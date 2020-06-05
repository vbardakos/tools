"""
Import Utility Class for Config & data
@vbar

"""
import tensorflow as tf
from _base import _InOut

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