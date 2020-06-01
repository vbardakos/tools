import tensorflow as tf
from _base import _Funcs

class IO(_Funcs):
    """
    Methods()
        config(self)
            Returns `dict` Config.json
        path(self)
            Returns `str` path used
        data(self,train=None)
            Returns a list of Datasets
    """

    def __init__(self,*args):
        super().__init__(*args)
        if not self.pdir:
            self.pdir = self._reader(self.name)
        self.c_path = self.pdir + self.conf
    
    def config(self):
        """ config.json from c_path """
        return self._reader(self.c_path)

    def path(self):
        """ c_path """
        return self.c_path

    def data(self, train=None):
        """
        params
            train (bool) : if not specified gets val from 'meta/data'
        """
        mapper = lambda x : tf.ensure_shape(tf.io.parse_tensor(x,out_type=d),s)
        ds_lst = list()
        for f in ('x','y'):
            s,d,n = self._val_extractor(train,f)
            d  = tf.dtypes.as_dtype(d)
            ds = tf.data.TFRecordDataset(f+n)
            ds = ds.map(mapper)
            ds_lst.append(ds)
        return ds_lst
        
    def _val_extractor(self,train,v):
        """ Extracts shape, dtype, name """
        c = self.config()
        if isinstance(train,type(None)):
            train = c['meta']['train']
        s = ['train' if train else 'test']
        name = c[s]['name'] + '.tfrecord'
        vals = [c[s][v][k] for k in ('shape','dtype')]
        assert all(v != None for v in vals)
        vals.append(name)
        return vals