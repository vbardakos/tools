import os
import json


class _Conf(object):

    FNAME = '.path.json'
    CNAME = '.conf.json'

    def __init__(self,path=None, path_name=FNAME, conf_name=CNAME):
        self.name = path_name
        self.pdir = path
        self.conf = conf_name

    @classmethod
    def add(cls,path,ignore=False):
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

    def set_batch(self,batch):
        if batch < 1:
            batch = 1
        else:
            assert isinstance(batch,int)
        self.batch = batch

    def set_steps(self,steps):
        if steps < 0:
            steps = 0
        else:
            assert isinstance(steps,int)
        self.steps = steps

    def set_train(self, train):
        assert isinstance(train,bool)
        self.train = train

    def set_conf(self):
        conf = self._load_conf()
        if hasattr(self,'batch'):
            conf['train']['batch'] = self.batch
        if hasattr(self,'steps'):
            conf['meta']['steps'] = self.steps
            if self.steps:
                conf['train']['steps'] = self.steps
            else:
                try:
                    conf['train']['steps'] = conf['param']['batch']//conf['train']['size']
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
        elif not isinstance(conf['param']['train'],bool):
            raise ValueError("train param undefined")
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

    def _fname(self,train,name=None):
        if name:
            self._xyname = name
        elif train:
            self._xyname = '_train'
        else:
            self._xyname = '_test'

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