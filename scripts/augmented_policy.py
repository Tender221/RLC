"""Matched, deterministic training views independent of replay and model RNG."""
import hashlib
from pathlib import Path
import time
import weakref

import numpy as np
import torch
from torch.nn import functional as F

from wnet.policy import UpdatePolicy,PolicyConfig
from wnet.pilot import _sync


def view_parameters(cfg,step,size):
    rng=np.random.default_rng(np.random.SeedSequence([cfg.seed,cfg.order_seed,step,20261031,503]))
    coordinates=rng.integers(0,9,size=(size,2))
    flip=rng.integers(0,2,size=(size,1))
    return np.concatenate([coordinates,flip],axis=1).tolist()


@torch.no_grad()
def apply_views(x,parameters):
    if x.ndim!=4 or x.shape[-2:]!=(32,32) or len(parameters)!=len(x):
        raise ValueError('Fixed CIFAR view protocol expects one crop/flip per 32x32 image')
    p=torch.as_tensor(parameters,dtype=torch.long,device=x.device)
    padded=F.pad(x,(4,4,4,4),mode='reflect')
    windows=padded.unfold(2,32,1).unfold(3,32,1)
    crops=windows[torch.arange(len(x),device=x.device),:,p[:,0],p[:,1]]
    return torch.where(p[:,2,None,None,None].bool(),crops.flip(-1),crops)


class AugmentedPolicy(UpdatePolicy):
    def __init__(self,inner=None):
        super().__init__(PolicyConfig(method='observe',decision_every=1,max_fraction=0.))
        self.inner=inner;self.parameters=None;self._model_ref=None;self._handle=None
        self.device='cpu'

    def configuration(self):
        return {**super().configuration(),'training_view':'reflect_pad4_crop32_horizontal_flip',
                'view_rng_salt':[20261031,503],'inner':None if self.inner is None else self.inner.configuration(),
                'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def initialize(self,stream,cfg):
        super().initialize(stream,cfg)
        if cfg.max_diagnostics:
            raise ValueError('Augmented common-setting protocol uses no diagnostics')
        self.device=cfg.device
        if self.inner is not None:
            self.inner.initialize(stream,cfg)
            self.records=self.inner.records

    def _augment_training_input(self,module,args):
        if not module.training or self.parameters is None:
            return args
        _sync(self.device); started=time.perf_counter()
        x=apply_views(args[0],self.parameters)
        _sync(self.device);self.feature_seconds+=time.perf_counter()-started
        return (x,)+args[1:]

    def decide(self,model,opt,candidate,bundle,indices,memory,order,task,step,cfg,loss_fn):
        _sync(cfg.device);started=time.perf_counter()
        if self._model_ref is None or self._model_ref() is not model:
            if self._handle is not None:self._handle.remove()
            self._handle=model.register_forward_pre_hook(self._augment_training_input)
            self._model_ref=weakref.ref(model)
        self.parameters=view_parameters(cfg,step,len(candidate[1]))
        self.feature_seconds+=time.perf_counter()-started
        if self.inner is not None:
            factor=self.inner.decide(model,opt,candidate,bundle,indices,memory,order,task,step,cfg,loss_fn)
            self.records=self.inner.records
        else:
            factor=1.
            counts=torch.bincount(candidate[1].detach().cpu(),minlength=cfg.num_classes).tolist()
            self.records.append({'step':step,'task':task,'factor':1.,'score':None,'features':{},'input_counts':counts})
        self.records[-1]['view_parameters']=self.parameters
        return factor

    def state(self):
        return {**super().state(),'inner':None if self.inner is None else self.inner.state()}

    def load(self,state):
        super().load(state)
        if self.inner is not None:
            self.inner.load(state['inner'])
            self.records=self.inner.records

    def summary(self):
        name='er' if self.inner is None else self.inner.summary()['policy_method']
        return {**super().summary(),'policy_method':'augmented_'+name,
                'augmentation_seconds':self.feature_seconds,
                'policy_feature_seconds':self.feature_seconds+(0. if self.inner is None else self.inner.feature_seconds)}
