"""Class-balanced two-parameter BiC-style output calibration on training memory."""
import numpy as np
from scipy.optimize import minimize
import torch


def fit_scale_shift(logits,labels,new_classes):
    z=torch.as_tensor(logits).detach().cpu().double().numpy().copy()
    y=torch.as_tensor(labels).detach().cpu().long().numpy().copy()
    new=list(new_classes)
    if (z.ndim!=2 or len(z)==0 or y.shape!=(len(z),) or not np.isfinite(z).all()
            or not new or len(set(new))!=len(new) or min(new)<0 or max(new)>=z.shape[1]
            or len(new)==z.shape[1] or int(y.min())<0 or int(y.max())>=z.shape[1]):
        raise ValueError('Expected finite memory logits/labels and a proper new-class subset')
    counts=np.bincount(y,minlength=z.shape[1]);weights=1/counts[y];weights/=weights.sum()
    indicator=np.zeros(z.shape[1]);indicator[new]=1.;slope=z*indicator
    positions=np.arange(len(z))
    target=np.array([np.sum(weights*slope[positions,y]),np.sum(weights*indicator[y])])
    def objective(parameters):
        alpha,beta=parameters;adjusted=z+(alpha-1)*slope+beta*indicator
        maximum=adjusted.max(1);exp=np.exp(adjusted-maximum[:,None]);normalizer=exp.sum(1)
        probabilities=exp/normalizer[:,None]
        loss=float(np.sum(weights*(maximum+np.log(normalizer)-adjusted[positions,y])))
        gradient=np.array([np.sum(weights*np.sum(probabilities*slope,1)),
                           np.sum(weights*np.sum(probabilities*indicator,1))])-target
        return loss,gradient
    before,_=objective(np.array([1.,0.]))
    result=minimize(objective,np.array([1.,0.]),jac=True,method='BFGS',options=dict(gtol=1e-10,maxiter=500))
    after,gradient=objective(result.x)
    if not np.isfinite(result.x).all() or not np.isfinite(after) or after>before+1e-9:
        raise RuntimeError('Two-parameter fit did not return a finite non-increasing likelihood')
    residual=float(np.max(np.abs(gradient)))
    return dict(alpha=float(result.x[0]),beta=float(result.x[1]),loss_before=before,loss_after=after,
                gradient_max=residual,converged=residual<1e-7,solver_success=bool(result.success),
                solver_status=int(result.status),iterations=int(result.nit),message=str(result.message))


def apply_scale_shift(logits,new_classes,fit):
    adjusted=torch.as_tensor(logits).detach().cpu().double().clone()
    adjusted[:,list(new_classes)]=fit['alpha']*adjusted[:,list(new_classes)]+fit['beta']
    return adjusted
