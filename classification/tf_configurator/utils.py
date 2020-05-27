class Utilities(object):

    def __init__(self, features=None, labels=None, unbatch=False):
        if unbatch:
            self.i = features.unbatch()
            self.o = labels.unbatch()
        else:
            self.i = features
            self.o = labels

    def data(self):
        return self.i, self.o

    def batch_size(self):
        return _set_batch()

    def shape(self,labels=False):
        if not labels:
            shape = self.i.element_spec.shape
        else:
            shape = self.o.element_spec.shape
        return shape

    def has_attr(self):
        if hasattr(self,'sample'):
            print(True)
        else:
            print(False)

    def dtype(self,labels=False):
        if not labels:
            dtype = self.i.element_spec.dtype
        else:
            dtype = self.o.element_spec.dtype
        
        return str(dtype).split("'")[1]
        
    def classes(self):
        """ class number """
        if hasattr(self,'cls'):
            pass
        elif self.shape(True).as_list():
            self.cls = self.shape(True)[0]
        else:
            dtype = tf.dtypes.as_dtype(self.dtype(True))
            self.cls = self.o.reduce(tf.constant(0,dtype), lambda _,y : tf.maximum(_,y)).numpy() + 1
        return self.cls

    def sample_size(self):
        """ data occurences """
        if hasattr(self,'sample'):
            pass
        elif hasattr(self,'o'):
            self.sample = self.o.reduce(tf.constant(0), lambda x,_ : x+1).numpy()
        else:
            self.sample = self.i.reduce(tf.constant(0), lambda x,_ : x+1).numpy()
        return self.sample

    def steps(self):
        """ training steps with rounded observations/batch_size """
        if _set_steps():
            return _set_steps()
        elif hasattr(self,'sample'):
            return self.sample//self.batch_size()
        else:
            self.sample = self.sample_size()
            return self.sample//self.batch_size()

    def output_one_hot(self):
        if not hasattr(self,'cls'):
            self.cls = self.classes()
        self.o = self.o.map(lambda x : tf.one_hot(x, self.cls))
        return self.o