import os
import json

class _Funcs(object):

    FNAME = '.path.json'
    CNAME = '.conf.json'

    def __init__(self,path=None, path_name=FNAME, conf_name=CNAME):
        self.name = path_name
        self.pdir = path
        self.conf = conf_name

    @staticmethod
    def _reader(name):
        with open(name) as f:
            file = json.load(f)
        return file
    
    @staticmethod
    def _writer(var,name):
        with open(name,'w') as f:
            json.dump(var,f)

    @staticmethod
    def _change(name,split,rename=True):
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
                'x': {'shape': None, 'dtype': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
            },
            'test' : {
                'name': None, 'size': None,
                'x': {'shape': None, 'dtype': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
            },
            'meta': {'steps': None, 'train': None},
        }
        return conf


class _Conf(_Funcs):

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
        os.mkdir(path)
        return cls(path)

    @classmethod
    def reset(cls,old_path=False):
        if old_path:
            path = cls._change(cls.FNAME,'.',False)
        else:
            path = cls.FNAME
        try:
            c_dir = cls._reader(path)
            os.remove(path)
            [os.remove(c_dir+f) for f in os.listdir(c_dir)]
            os.rmdir(c_dir)
        except FileNotFoundError:
            print('File/Directory Not Found')

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
        if hasattr(self,'batch'):
            conf['train']['batch'] = self.batch
        if hasattr(self,'steps'):
            conf['meta']['steps'] = self.steps
            if self.steps:
                conf['train']['steps'] = self.steps
            else:
                try:
                    conf['train']['steps'] = conf['train']['batch']//conf['train']['size']
                    conf['meta']['steps'] = 'Auto'
                except:
                    pass
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
        if not self.pdir:
            self.pdir = self._reader(self.name)
        self._writer(conf,self.pdir+self.conf)

    def _load_conf(self):
        """ load config """
        if not self.pdir:
            self.pdir = self._reader(self.name)
        if os.path.exists(self.pdir) and not os.path.exists(self.pdir+self.conf):
            self._writer(self._empty_conf(),self.pdir+self.conf)
        return self._reader(self.pdir+self.conf)

    def _fname(self,train,name=None):
        if name:
            self._xyname = name
        elif train:
            self._xyname = '_train'
        else:
            self._xyname = '_test'