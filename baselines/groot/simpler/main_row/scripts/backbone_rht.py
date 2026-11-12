"""Deterministic orthogonal RHT for non-power-of-two backbone widths.

Use a direct sum of maximal power-of-two Walsh blocks, in descending order:
1152=1024+128, 4304=4096+128+64+16. Expert RHT code is unchanged.
"""
import torch
from rht import hadamard,layer_signs


def block_widths(width):
    if width < 1: raise ValueError('Positive width required')
    result=[]
    while width:
        block=1 << (width.bit_length()-1); result.append(block); width-=block
    return result


def transform(x,signs):
    if x.shape[-1]!=len(signs): raise ValueError('Sign/input mismatch')
    signed=x*signs.to(device=x.device,dtype=x.dtype)
    widths=block_widths(len(signs))
    if len(widths)==1:return hadamard(signed)
    return torch.cat([hadamard(part) for part in signed.split(widths,dim=-1)],dim=-1)


def transform_covariance(h,signs):
    return transform(transform(h,signs).transpose(-1,-2),signs).transpose(-1,-2)
