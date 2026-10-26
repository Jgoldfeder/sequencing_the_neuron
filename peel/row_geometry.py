"""Positive-gauge-invariant row comparisons using the recovered prefix."""
import torch,math
@torch.no_grad()
def input_scales(model,layer):
    w=model.layers[0].weight
    s=torch.ones(w.shape[1],device=w.device,dtype=w.dtype)
    for mod in model.layers[:layer]:
        s=(mod.weight*s[None]).norm(dim=1).clamp_min(1e-30)
    return s
@torch.no_grad()
def row_angle(model,layer,w,b,wg,bg):
    scales=input_scales(model,layer)
    v=torch.cat([w*scales,b.reshape(1)])
    g=torch.cat([wg*scales,bg.reshape(1)])
    return math.degrees(math.acos(float(((v@g).abs()/(v.norm()*g.norm())).clamp(max=1))))
