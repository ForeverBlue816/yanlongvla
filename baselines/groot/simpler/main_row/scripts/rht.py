"""Orthogonal input RHT: row x -> x D H, folded row W -> W D H."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F


def hadamard(x):
    n=x.shape[-1]
    if n<1 or n&(n-1):raise ValueError('RHT requires a power-of-two input dimension')
    result=x
    width=1
    while width<n:
        shape=result.shape
        pairs=result.reshape(*shape[:-1],n//(2*width),2,width)
        a,b=pairs[...,0,:],pairs[...,1,:]
        result=torch.stack((a+b,a-b),dim=-2).reshape(shape)
        width*=2
    return result/math.sqrt(n)


def transform(x,signs):
    if x.shape[-1]!=len(signs):raise ValueError('Sign vector/input dimension mismatch')
    return hadamard(x*signs.to(device=x.device,dtype=x.dtype))


def layer_signs(name,features,seed=20260930):
    # Private CPU RNG; does not change policy/algorithm streams or depend on Python hash randomization.
    key=hashlib.sha256(f'{seed}:{name}'.encode()).digest()
    g=torch.Generator(device='cpu').manual_seed(int.from_bytes(key[:8],'little')%(2**63))
    return (torch.randint(0,2,(features,),generator=g,dtype=torch.int8)*2-1)


def transform_covariance(h,signs):
    # H' = H_walsh D H_input D H_walsh; supports per-step leading axes.
    return transform(transform(h,signs).transpose(-1,-2),signs).transpose(-1,-2)


class RHTLinear(nn.Module):
    def __init__(self,source,name):
        super().__init__()
        self.in_features=source.in_features;self.out_features=source.out_features
        self.register_buffer('signs',layer_signs(name,self.in_features).to(source.weight.device))
        self.weight=nn.Parameter(transform(source.weight.detach().float(),self.signs).to(source.weight.dtype),requires_grad=False)
        self.bias=None if source.bias is None else nn.Parameter(source.bias.detach().clone(),requires_grad=False)
    def forward(self,x):
        return F.linear(transform(x.float(),self.signs).to(self.weight.dtype),self.weight,self.bias)


def install_expert_rht(model):
    names=[name for name,m in model.named_modules() if isinstance(m,nn.Linear) and '.gemma_expert.model.layers.' in name and 'layernorm' not in name]
    for name in names:
        parent,attribute=name.rsplit('.',1);source=model.get_submodule(name)
        setattr(model.get_submodule(parent),attribute,RHTLinear(source,name))
    if len(names)!=126:raise ValueError(f'Unexpected eligible expert Linear count:{len(names)}')
    return names
