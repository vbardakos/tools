class Configurator(Utilities):

    def __init__(self, inputs, train=True, unbatch=False):
        if isinstance(inputs, tuple) and len(inputs) is 2:
            assert all(isinstance(o,tf.data.Dataset) for o in inputs)
            i = inputs[0]
            o = inputs[1]
        elif isinstance(inputs.element_spec, tuple) and len(inputs.element_spec):
            i = inputs.map(lambda x,_ : x)
            o = inputs.map(lambda _,y : y)
        else:
            raise ValueError('Invalid Dataset/s')

        self.path = _set_path()
        if train:
            self.name = lambda x : f'{self.path}{x}_train.tfrecord'
        else:
            self.name = lambda x : f'{self.path}{x}_test.tfrecord'

        super().__init__(i,o,unbatch)

    def one_hot(self):
        if not hasattr(self,'cls'):
            self.cls = self.classes()
        self.o = self.o.map(lambda x : tf.one_hot(x, self.cls))
        return self.o

    def write_config(self):
        conf = {
            'train' :  {
                'batch' : int(self.batch_size()),
                'steps' : int(self.steps()),
                'size'  : int(self.sample_size()),
                },
            'x' : {
                'shape' : list(self.shape()),
                'dtype' : str(self.dtype()),
                },
            'y' : {
                'shape' : list(self.shape(True)),
                'dtype' : str(self.dtype(True)),
                'class' : int(self.classes()),
                }   
            }

        with open(self.path+'config.json', 'w') as f:
            json.dump(conf, f)

    def name(self):
        return name('x')

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

    def write_all(self):
        self.write_config()
        self.write_input()
        self.write_output()

    def reset_config(self):
        try:
            os.remove(self.path+'config.json')
        except:
            pass

    def reset_data(self):
        for f in os.listdir(self.path):
            if f.endswith('.tfrecord'):
                os.remove(self.path+f)