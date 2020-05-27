class Reader(object):

    def __init__(self,path=_set_path()):
        self.path, self.dir = path, None
        if self.path:
            self.dir = self.path
        if 'config.json' not in os.listdir('data/'):
            raise FileExistsError("'config.json' Does Not Exist")

    def config(self):
        if not hasattr(self,'conf'):
            with open(self.path+'config.json') as f:
                self.conf = json.load(f)
        return self.conf

    def data(self,train=True):
        if train:
            name = '_train.tfrecord'
        else:
            name = '_test.tfrecord'
        if not hasattr(self,'conf'):
            self.conf = self.config()
        
        lst = list()
        for f in ('x','y'):
            shape = self.conf[f]['shape']
            dtype = self.conf[f]['dtype']
            dtype = tf.dtypes.as_dtype(dtype)
            try:
                n  = self.path + f + name
                ds = tf.data.TFRecordDataset(n)
                ds = ds.map(lambda x: tf.ensure_shape(tf.io.parse_tensor(x, dtype), shape))
                lst.append(ds)
            except:
                lst.append(None)
        return lst