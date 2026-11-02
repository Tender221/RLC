"""Replay likelihood correction at task boundaries with optional contrast velocity reset."""
import hashlib
from pathlib import Path
import time
import weakref

import torch

from scripts.augmented_policy import AugmentedPolicy
from scripts.replay_bias import fit_bias_shift
from wnet.paired import capture_state,restore_state
from wnet.pilot import _batch,_sync


def bias_direction(order,task,classes_per_task,num_classes):
    seen=order[:(task+1)*classes_per_task]
    new=order[task*classes_per_task:(task+1)*classes_per_task]
    d=torch.zeros(num_classes,dtype=torch.float64)
    d[seen]=-len(new)/len(seen);d[new]+=1.
    return d


@torch.no_grad()
def apply_bias_calibration(model,opt,logits,labels,direction,state_control):
    layers=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    if not layers or layers[-1].bias is None:raise ValueError('Last classifier needs a linear bias')
    bias=layers[-1].bias
    if len(bias)!=len(direction):raise ValueError('Direction must match full output head')
    fit=fit_bias_shift(logits,labels,direction)
    d=direction.to(bias);bias.add_(fit['shift']*d)
    velocity=opt.state.get(bias,{}).get('momentum_buffer')
    removed=0.;residual=0.
    if velocity is not None and float(d@d)>0:
        if state_control:
            projection=(velocity@d)/(d@d)*d
            removed=float(projection.norm());velocity.sub_(projection)
        residual=abs(float(velocity@d))
    return {**fit,'velocity_removed_norm':removed,'velocity_residual':residual}


class BoundaryBiasPolicy(AugmentedPolicy):
    def __init__(self,state_control=False):
        super().__init__();self.state_control=bool(state_control)
        self._opt_ref=None;self._step_handle=None;self._pending=None

    def configuration(self):
        return {**super().configuration(),'boundary_bias':True,'contrast_velocity_reset':self.state_control,
                'calibration_memory':'post_sgd_before_reservoir_add','calibration_batch_size':128,
                'boundary_driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fit_sha256':hashlib.sha256(Path('scripts/replay_bias.py').read_bytes()).hexdigest()}

    def initialize(self,stream,cfg):
        super().initialize(stream,cfg)
        if cfg.optimizer!='sgd':raise ValueError('Contrast state protocol requires SGD')
        self.ends={step for step,(task,_) in enumerate(stream)
                   if step+1==len(stream) or stream[step+1][0]!=task}

    def decide(self,model,opt,candidate,bundle,indices,memory,order,task,step,cfg,loss_fn):
        factor=super().decide(model,opt,candidate,bundle,indices,memory,order,task,step,cfg,loss_fn)
        if self._opt_ref is None or self._opt_ref() is not opt:
            if self._step_handle is not None:self._step_handle.remove()
            self._step_handle=opt.register_step_post_hook(self._after_step)
            self._opt_ref=weakref.ref(opt)
        self._pending=(weakref.ref(model),bundle,list(memory.slots),order,task,cfg,self.records[-1]) if (
            task>0 and step in self.ends and memory.slots) else None
        return factor

    def _after_step(self,opt,args,kwargs):
        if self._pending is None:return
        ref,bundle,slots,order,task,cfg,record=self._pending;self._pending=None
        model=ref();_sync(cfg.device);started=time.perf_counter()
        before=capture_state(model,opt)
        try:
            model.eval();chunks=[];labels=[]
            with torch.no_grad():
                for start in range(0,len(slots),128):
                    x,y=_batch(bundle,slots[start:start+128],cfg.device)
                    chunks.append(model(x).detach().cpu());labels.append(y.cpu())
        finally:restore_state(model,opt,before)
        d=bias_direction(order,task,cfg.classes_per_task,cfg.num_classes)
        fit=apply_bias_calibration(model,opt,torch.cat(chunks),torch.cat(labels),d,self.state_control)
        record['calibration']={**fit,'memory_indices':slots,'direction':d.tolist()}
        _sync(cfg.device);self.feature_seconds+=time.perf_counter()-started

    def summary(self):
        return {**super().summary(),'policy_method':'boundary_bias_state' if self.state_control else 'boundary_bias',
                'calibration_events':sum('calibration' in r for r in self.records)}
