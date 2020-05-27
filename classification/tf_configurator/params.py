import os
import json

### TO DO ###
# Config Class
# __init__:
#   1 add(ignore) -> ignore : os.rename old '.path.json', '.conf.json'
#   2 reset() -> rm '.conf.json', '.path.json'
#   3 load() -> try json.load()
# set:
#   1 batch - Default 1
#   2 steps - Default 0
#   3 train - cannot run Configurator

# Configurator Class - Config child
# __init__:
#   1 param : input_fn
# write
# 


class Params(object):

    def __init__(self, path=None):
        self.conf = False

        if isinstance(path,str):
            if not path.endswith('/'):
                path = path + '/'
            if os.path.exists('.path.json'):
                with open('.path.json') as f:
                    init = json.load(f)

                if init != path:
                    for f in os.listdir(init):
                        os.remove(f)
                    os.rename(init,path)
                else:
                    self.conf = True
            else:
                os.mkdir(path)

            with open('.path.json','w') as f:
                json.dump(path,f)
        else:
            try:
                # tries to get current path
                with open('.path.json') as f:
                    path = json.load(f)
                    self.conf = True
            except FileNotFoundError:
                raise ValueError("param : PATH not specified")
            except:
                raise BrokenPipeError("Unexpected Error")

        self.path = path + '_conf.json'

    def set(self, batch=None,steps=None,train=None):
        params = (batch,steps,train)
        if conf:
            if any(not isinstance(p,type(None)) for p in params):
                with open(self.path,'w') as f:
                    c = json.load(f)
                if isinstance(batch,int):
                    c['param']['batch'] = self._batch(batch)
                if isinstance(steps,int):
                    if steps:
                        c['param']['steps'] = self._steps(steps)
                    elif c['train']['size']:
                        c['param']['steps'] = self._steps(steps)//c['train']['size']
                    else:
                        c['param']['steps'] = 0
                if isinstance(train,bool):
                    c['param']['train'] = train
        elif all(isinstance(p,(int,bool)) for p in params):
            c = self._conf(self._batch(batch),self._steps(steps),train)
        else:
            raise ValueError("Params not Specified")

        with open(self.path,'w') as f:
            json.dump(c,f)

    def _batch(self, batch_size):
        if batch_size < 1:
            batch_size = 1
        return batch_size

    def _steps(self, steps):
        if steps < 0:
            steps = 0
        return steps

    def _conf(self, batch, steps, train):
        conf = {
            'param': {'batch': batch, 'steps': steps, 'train': train},
            'train': {
                'size': None,
                'x': {'shape': None, 'dtype': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
                },
            'test' : {
                'size': None,
                'x': {'shape': None, 'dtype': None},
                'y': {'shape': None, 'dtype': None, 'class': None},
                },
            }
        return conf
