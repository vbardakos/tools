"""
Base classes for Configurator & IO
@vbar

TO DO
    - Fix Utils and move it
    - reassign methods
"""
import os
import json
import tensorflow as tf

class _InOut(object):
    """
    Parent Utility class for IO & _Conf
    """

    FNAME = '.path.json'
    CNAME = '.conf.json'

    def __init__(self, path_name=FNAME, conf_name=CNAME):
        try:
            self.pdir = self._reader(path_name)
            self.c_path = self.pdir + conf_name
            self.conf = self._reader(self.c_path)
        except Exception as e:
            raise FileNotFoundError(e,' `config.json` path is not assigned')

    @staticmethod
    def _reader(name):
        """ json reader """
        with open(name) as f:
            file = json.load(f)
        return file
    
    @staticmethod
    def _writer(var,name):
        """ json writer """
        with open(name,'w') as f:
            json.dump(var,f)


class _ConfUtils(object):

    def _load_conf(self):
        """ load config """
        if not os.path.exists(self.c_path):
            self._writer(self._empty_conf(),self.c_path)
            self.conf = self._reader(self.c_path)
        return self.conf

    def _fname(self,train,name=None):
        """ data file name suffix """
        if name:
            self._xyname = name
        elif train:
            self._xyname = '_train'
        else:
            self._xyname = '_test'

    @staticmethod
    def _change(name,split,rename=True):
        """ name changer to old """
        new = name.split(split)
        new[-2] += '_old'
        new = split.join(new)
        if rename:
            os.rename(name,new)
        else:
            return new

    @staticmethod
    def _empty_conf():
        conf = {
            'train': {
                'name' : None, 'batch': None, 'steps': None, 'size' : None,
                'x': {'shape': None, 'dtype': None, 'class': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
            },
            'test' : {
                'name': None, 'size': None,
                'x': {'shape': None, 'dtype': None, 'class': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
            },
            'meta': {'steps': None, 'train': None},
        }
        return conf


class _ConfBase(_ConfUtils, _InOut):

    def __init__(self,*args):
        super().__init__(*args)

    @classmethod
    def add_path(cls,path,ignore=False):
        if path and not path.endswith('/'):
            path += '/'
        if ignore:
            try:
                c_dir = cls._reader(cls.FNAME)
                # change file/path func
                cls._change(cls.FNAME,'.')
                cls._change(c_dir,'.')
            except FileNotFoundError:
                pass # Captures Syntax/Permission/Exists
        elif os.path.exists(cls.FNAME):
            raise FileExistsError(f"'{cls.FNAME}' already exists. Try to reset()")
        cls._writer(path,cls.FNAME)
        if path:
            os.mkdir(path)
        cls._writer(cls._empty_conf(),path+cls.CNAME)
        return cls()

    @classmethod
    def reset(cls,old_path=False):
        if old_path:
            path = cls._change(cls.FNAME,'.',False)
        else:
            path = cls.FNAME
        try:
            c_dir = cls._reader(path)
            os.remove(path)
            if c_dir:
                [os.remove(c_dir+f) for f in os.listdir(c_dir)]
                os.rmdir(c_dir)
            else:
                [os.remove(f) for f in os.listdir() if f.endswith(('json','tfrecord'))]
        except FileNotFoundError:
            print('Reset: File/Directory Not Found')

    def add_batch(self,batch):
        assert isinstance(batch,int)
        if batch < 1:
            batch = 1
        self.batch = batch
        return self

    def add_steps(self,steps):
        assert isinstance(steps,int)
        if steps < 0:
            steps = 0
        self.steps = steps
        return self

    def add_train(self, train):
        assert isinstance(train,bool)
        self.train = train
        return self

    def set_config_params(self):
        conf = self._load_conf()

        self._conf_batch(conf)
        self._conf_steps(conf)
        self._conf_train(conf)

        self._writer(conf,self.c_path)

    def _conf_batch(self,conf):
        if hasattr(self,'batch'):
            conf['train']['batch'] = self.batch

    def _conf_steps(self,conf):
        """
        sets steps in config
            if steps -> steps
            else:    -> if has been set: meta = 0
                        else:            meta = None
        """
        auto = lambda : conf['train']['size']//conf['train']['batch']
        if hasattr(self,'steps'):
            conf['meta']['steps'] = self.steps
            if self.steps:
                conf['train']['steps'] = self.steps
            else:
                try:
                    conf['train']['steps'] = auto()
                    conf['meta']['steps']  = 'Auto'
                except:
                    pass
        elif hasattr(self,'batch') and isinstance(conf['meta']['steps'],str):
            conf['train']['steps'] = auto()
    
    def _conf_train(self,conf):
        if hasattr(self,'train'):
            conf['meta']['train'] = self.train
            if not hasattr(self,'_xyname'):
                self._fname(self.train)
            if self.train:
                conf['train']['name'] = self._xyname
            else:
                conf['test']['name']  = self._xyname
        elif not isinstance(conf['meta']['train'],bool):
            raise ValueError("'train' param undefined")


class _ConfData(_InOut):

    # def __init__(self, *args):
    #     super().__init__()

    @classmethod
    def add_data(cls,data,train=None,*args):
        super().__init__(cls)
        if isinstance(data,(tuple,list)) and len(data) is 2:
            assert all(isinstance(f,tf.data.Dataset) for f in data)
            cls.i = data[0]
            cls.o = data[1]
        elif isinstance(data,tf.data.Dataset):
            if isinstance(data.element_spec,tuple) and len(data.element_spec):
                cls.i = data.map(lambda x,_ : x)
                cls.o = data.map(lambda _,y : y)
        else:
            raise ValueError("add_data : Could not get the data")

        if isinstance(train,type(None)):
            train = cls.conf['meta']['train']
        assert isinstance(train,bool)
        cls.name = 'train' if train else 'test'
        return cls()

    def _data_shape(self,label=False,ignore=False):
        """ gets the shape of features/labels """
        dname  = self._dname(label)
        shape = tf.TensorShape(self.conf[self.name][dname]['shape'])
        if label and (not shape or ignore):
            shape = self.o.element_spec.shape
        elif not shape or ignore:
            shape = self.i.element_spec.shape
        self._shape = tuple(shape)
        self.conf[self.name][dname]['shape'] = self._shape

    def _data_dtype(self,label=False,ignore=False):
        dname = self._dname(label)
        dtype = self.conf[self.name][dname]['dtype']
        if label and (not dtype or ignore):
            self._dtype = self.o.element_spec.dtype
        elif not dtype or ignore:
            self._dtype = self.i.element_spec.dtype
        else:
            self._dtype = tf.dtypes.as_dtype(dtype)
        self.conf[self.name][dname]['dtype'] = str(self._dtype).split("'")[1]
        
    def _data_class(self,label=False,ignore=False):
        """ class number """
        dname = self._dname(label)
        clses = self.conf[self.name][dname]['class']
        if not clses or ignore:
            self._data_dtype(label)
            self._data_shape(label)
            mapper = lambda _,y : tf.maximum(_,y)
            if self._shape:
                self._clses = self._shape[0]
            elif label:
                self._clses = self.o.reduce(tf.constant(0,self._dtype),mapper).numpy() + 1
            else:
                self._clses = self.i.reduce(tf.constant(0,self._dtype),mapper).numpy() + 1
    
            self.conf[self.name][dname]['class'] = int(self._clses)

    def _data_steps(self):
        """ training steps with rounded observations/batch_size """
        meta = self.conf['meta']['steps']
        if isinstance(meta,str) or meta is 0:
            batch = self.conf['train']['batch']
            if not hasattr(self,'sample'):
                self._data_size()
            self.conf['meta']['steps']  = "Auto"
            self.conf['train']['steps'] = int(self.sample//batch)

    def _data_size(self, ignore=False):
        """ data occurences """
        self.sample = self.conf[self.name]['size']
        if not self.sample or ignore: # None or ignore
            self.sample = self.o.reduce(tf.constant(0), lambda x,_ : x+1).numpy()
            self.conf[self.name]['size'] = int(self.sample)
    
    @staticmethod
    def _dname(label):
        return 'x' if not label else 'y'

    def printer(self):
        for l in (True,False):
            self._data_class(l)
            self._data_shape(l)
            self._data_dtype(l)
        self._data_size()
        self._data_steps()
        self._writer(self.conf,self.c_path)
        return self.conf