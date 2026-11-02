"""Shared fixed-model calibration operations; no evaluation-label fitting."""
import math
import statistics
import time

import torch

from scripts.bic_component import fit_scale_shift
from scripts.replay_bias import fit_bias_shift


VARIANTS=('la_memory','la_mixture','group_la')
TAUS=(0.,.25,.5,1.,2.,4.)


def frequency_priors(labels,classes,new_classes,incoming_fraction=.5,smoothing=1.):
    y=torch.as_tensor(labels).detach().cpu().long()
    new=list(new_classes)
    if (y.ndim!=1 or not len(y) or classes<2 or int(y.min())<0 or int(y.max())>=classes
            or not new or len(set(new))!=len(new) or min(new)<0 or max(new)>=classes
            or len(new)==classes or not 0<=incoming_fraction<=1 or smoothing<=0):
        raise ValueError('Expected memory labels, positive smoothing and a proper new-class subset')
    counts=torch.bincount(y,minlength=classes).double()+smoothing
    memory=counts/counts.sum()
    incoming=torch.zeros(classes,dtype=torch.float64);incoming[new]=1/len(new)
    mixture=(1-incoming_fraction)*memory+incoming_fraction*incoming
    old=[c for c in range(classes) if c not in new]
    grouped=mixture.clone();grouped[new]=mixture[new].mean();grouped[old]=mixture[old].mean()
    return dict(la_memory=memory,la_mixture=mixture,group_la=grouped)


def select_frequency(rows,expected_seeds,variants=VARIANTS,taus=TAUS):
    expected={(int(s),v,float(a)) for s in expected_seeds for v in variants for a in taus}
    keys=[(int(r['seed']),r['variant'],float(r['tau'])) for r in rows]
    if (any(r.get('stage')!='development' for r in rows) or len(keys)!=len(expected)
            or len(set(keys))!=len(keys) or set(keys)!=expected
            or any(not math.isfinite(r['accuracy_percent']) for r in rows)):
        raise ValueError('Complete, unique development-only frequency grid required')
    choices={}
    for variant in variants:
        scores={float(a):statistics.mean(r['accuracy_percent'] for r in rows
                    if r['variant']==variant and float(r['tau'])==float(a)) for a in taus}
        chosen=min(scores,key=lambda a:(-scores[a],a))
        choices[variant]=dict(tau=chosen,mean_validation_accuracy_percent=scores[chosen],
                              grid_mean_accuracy={str(a):value for a,value in scores.items()})
    selected=min(variants,key=lambda v:(-choices[v]['mean_validation_accuracy_percent'],list(variants).index(v)))
    return dict(complete=True,selected_variant=selected,variants=choices,
                selection_rule='Mean development accuracy; smaller tau then fixed variant order',
                development_seeds=list(expected_seeds))


def fit_one(name,logits,labels,order,classes_per_task,selection):
    classes=logits.shape[1];new=list(order[-classes_per_task:])
    if name=='raw':return dict(kind='raw')
    d=torch.full((classes,),-len(new)/classes,dtype=torch.float64);d[new]+=1.
    if name=='rlc':return dict(kind='rlc',direction=d.tolist(),**fit_bias_shift(logits,labels,d))
    if name=='bic_memory':return dict(kind='scale_shift',new_classes=new,**fit_scale_shift(logits,labels,new))
    if name in VARIANTS:
        prior=frequency_priors(labels,classes,new)[name]
        return dict(kind='frequency',prior=prior.tolist(),tau=selection['variants'][name]['tau'])
    raise ValueError(f'Unknown method {name}')


def fit_methods(logits,labels,order,classes_per_task,selection):
    return {name:fit_one(name,logits,labels,order,classes_per_task,selection)
            for name in ('raw','rlc','bic_memory',*VARIANTS)}


def measure_one(model,name,logits,labels,order,classes_per_task,selection):
    head=[m for m in model.modules() if isinstance(m,torch.nn.Linear)][-1]
    weight=head.weight.detach().clone();bias=head.bias.detach().clone()
    def sync():
        if head.weight.is_cuda:torch.cuda.synchronize(head.weight.device)
    sync();started=time.perf_counter()
    try:
        fit=fit_one(name,logits,labels,order,classes_per_task,selection)
        fitted=time.perf_counter();write_calibrator(model,fit);sync();ended=time.perf_counter()
    finally:
        with torch.no_grad():head.weight.copy_(weight);head.bias.copy_(bias)
        sync()
    return fit,dict(fit_seconds=fitted-started,writeback_seconds=ended-fitted,
                    fit_and_writeback_seconds=ended-started)


def apply_cached(logits,fit):
    z=torch.as_tensor(logits).detach().cpu().double().clone()
    kind=fit['kind']
    if kind=='raw':return z
    if kind=='rlc':return z+fit['shift']*torch.tensor(fit['direction'],dtype=torch.float64)
    if kind=='frequency':
        prior=torch.tensor(fit['prior'],dtype=torch.float64)
        if prior.shape!=(z.shape[1],) or not bool((prior>0).all()):raise ValueError('Positive full-head prior required')
        return z-fit['tau']*prior.log()
    if kind=='scale_shift':
        new=fit['new_classes'];z[:,new]=fit['alpha']*z[:,new]+fit['beta'];return z
    raise ValueError(f'Unknown calibration {kind}')


@torch.no_grad()
def write_calibrator(model,fit):
    layers=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    if not layers or layers[-1].bias is None:raise ValueError('A biased final linear head is required')
    head=layers[-1];kind=fit['kind']
    if kind=='raw':return
    if kind=='rlc':head.bias.add_(fit['shift']*torch.tensor(fit['direction'],device=head.bias.device,dtype=head.bias.dtype))
    elif kind=='frequency':head.bias.sub_(fit['tau']*torch.tensor(fit['prior'],device=head.bias.device,dtype=head.bias.dtype).log())
    elif kind=='scale_shift':
        ids=fit['new_classes'];head.weight[ids]=fit['alpha']*head.weight[ids]
        head.bias[ids]=fit['alpha']*head.bias[ids]+fit['beta']
    else:raise ValueError(f'Unknown calibration {kind}')


def prediction_metrics(logits,labels,order,classes_per_task,bins=15):
    z=torch.as_tensor(logits).detach().cpu().double();y=torch.as_tensor(labels).detach().cpu().long()
    if z.ndim!=2 or len(z)!=len(y) or not len(y) or not bool(torch.isfinite(z).all()):
        raise ValueError('Finite evaluation logits and matching labels required')
    prediction=z.argmax(1);correct=prediction==y
    totals=torch.bincount(y,minlength=z.shape[1]);success=torch.bincount(y[correct],minlength=z.shape[1])
    if not bool((totals>0).all()):raise ValueError('Evaluation must cover every class')
    per_class=success.double()/totals
    new=list(order[-classes_per_task:]);old=[c for c in order if c not in new]
    confidence=torch.softmax(z,dim=1).amax(1)
    positions=(confidence*bins).long().clamp(max=bins-1)
    counts=torch.bincount(positions,minlength=bins)
    conf=torch.bincount(positions,weights=confidence,minlength=bins)
    hits=torch.bincount(positions,weights=correct.double(),minlength=bins)
    nll=torch.logsumexp(z,dim=1)-z[torch.arange(len(y)),y]
    return dict(accuracy_percent=100*float(per_class.mean()),
                old_accuracy_percent=100*float(per_class[old].mean()),
                new_accuracy_percent=100*float(per_class[new].mean()),
                nll=float(nll.mean()),ece_percent=100*float((hits-conf).abs().sum()/len(y)),
                count=len(y),correct=int(correct.sum()),count_by_class=totals.tolist(),
                correct_by_class=success.tolist(),
                ece_bins=dict(counts=counts.tolist(),confidence_sums=conf.tolist(),correct_sums=hits.tolist()))
