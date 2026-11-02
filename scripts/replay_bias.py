"""A one-dimensional, class-balanced memory likelihood fit; no audit inputs."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def fit_bias_shift(logits,labels,direction):
    z=logits.detach().cpu().double();y=labels.detach().cpu().long();d=direction.detach().cpu().double()
    if z.ndim!=2 or len(y)!=len(z) or d.shape!=(z.shape[1],):
        raise ValueError('Expected memory logits, labels and one class-direction vector')
    counts=torch.bincount(y,minlength=z.shape[1]).double()
    weights=counts[y].reciprocal();weights/=weights.sum()
    target=float((weights*d[y]).sum())
    def loss(b):return float((weights*F.cross_entropy(z+b*d,y,reduction='none')).sum())
    def gradient(b):return float((weights*(torch.softmax(z+b*d,dim=1)*d).sum(1)).sum())-target
    before=loss(0.)
    if float(d.max()-d.min())==0. or not float(d.min())<target<float(d.max()):
        return {'shift':0.,'loss_before':before,'loss_after':before,'gradient_after':gradient(0.),'identifiable':False}
    low,high=-1.,1.
    for _ in range(30):
        if gradient(low)<=0<=gradient(high):break
        low*=2;high*=2
    else:raise ValueError('Finite convex memory optimum was not bracketed')
    for _ in range(80):
        mid=(low+high)/2
        if gradient(mid)>0:high=mid
        else:low=mid
    shift=(low+high)/2;after=loss(shift)
    assert after<=before+1e-10
    return {'shift':shift,'loss_before':before,'loss_after':after,
            'gradient_after':gradient(shift),'identifiable':True}
