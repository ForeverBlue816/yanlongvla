"""Additive group VQ primitives. No policy experiment, fitter or kernel claim in this module.

Group objectives may sum several step/subset decoders. GPTQ block feedback currently
supports the uniform full decoder used by reference(a). Conditional WLS uses the
allowed8x8 group-block metric, with codes fixed throughout its ALS updates.
"""
import torch


def subset_reconstruction(codebooks,codes,mask):
    # codes [rows,groups,M], output [rows,groups,8].
    shape=(*codes.shape[:-1],codebooks.shape[-1])
    result=torch.zeros(shape,dtype=codebooks.dtype,device=codebooks.device)
    for k in range(len(codebooks)):
        if mask[k]:result=result+codebooks[k][codes[...,k].long()]
    return result


def _decoder_loss(target,sums,metrics,weights):
    # sums [decoders,N,beams,K,group]. Targets identical across step decoders.
    error=target[None,:,None,None,:]-sums
    transformed=torch.einsum('snbki,sij->snbkj',error,metrics)
    return (transformed*error).sum(-1).mul(weights[:,None,None,None]).sum(0)


@torch.no_grad()
def joint_beam_encode(target,codebooks,metrics,masks=None,weights=None,beam=8):
    """Joint beam over all plane indices and all supplied subset decoder losses.

    target[N,8], books[M,K,8], metrics[S,8,8], masks[S,M]. The beam
    retains partial index tuples; it never freezes a residual-greedy choice.
    Use caller-side row batching to bound memory (e.g.N=64..256).
    """
    m,k,d=codebooks.shape;n=target.shape[0]
    if target.shape!=(n,d) or metrics.ndim!=3 or metrics.shape[-2:]!=(d,d):raise ValueError('Invalid beam shape')
    if not 1<=m<=3 or k>256 or beam<1:raise ValueError('Unsupported M/K/beam')
    s=metrics.shape[0]
    if masks is None:masks=torch.ones((s,m),device=target.device,dtype=torch.bool)
    if weights is None:weights=torch.ones(s,device=target.device,dtype=target.dtype)
    if masks.shape!=(s,m) or weights.shape!=(s,) or not masks.any(1).all() or not masks.any(0).all():raise ValueError('Each decoder/plane must be active')
    paths=torch.empty((n,1,0),dtype=torch.long,device=target.device)
    decoded=torch.zeros((s,n,1,d),dtype=target.dtype,device=target.device)
    for plane in range(m):
        candidates=decoded[:,:,:,None,:]+masks[:,plane,None,None,None,None]*codebooks[plane][None,None,None,:,:]
        score=_decoder_loss(target,candidates,metrics,weights)
        width=min(beam,score.shape[1]*k)
        _,selected=score.flatten(1).topk(width,dim=1,largest=False,sorted=True)
        parent=selected//k;index=selected%k
        paths=torch.cat((paths.gather(1,parent[...,None].expand(-1,-1,plane)),index[...,None]),-1)
        flat=candidates.reshape(s,n,-1,d)
        decoded=flat.gather(2,selected[None,:,:,None].expand(s,n,width,d))
    return paths[:,0].to(torch.uint8)


@torch.no_grad()
def gptq_encode_uniform(weight,hessian,codebooks,*,damping=.01,beam=8,row_batch=128):
    """8-column GPTQ blocks, full H error feedback and joint-M group encoding.

    Inputs are normalized folded weights (row_scale handled by the caller).
    Returns deployed indices and reconstructed normalized weights. Uses the
    Cholesky factor of inverse damped H; preserves the caller's input arrays.
    """
    out,features=weight.shape;m,k,group=codebooks.shape
    if features%group or hessian.shape!=(features,features):raise ValueError('Weight/H mismatch')
    h=hessian.clone();h=(h+h.T)/2
    if not torch.isfinite(h).all() or not torch.isfinite(weight).all():raise ValueError('Nonfinite input')
    damp=damping*h.diagonal().mean().clamp_min(torch.finfo(h.dtype).eps)
    h.diagonal().add_(damp)
    inverse=torch.cholesky_inverse(torch.linalg.cholesky(h))
    upper=torch.linalg.cholesky(inverse,upper=True)
    work=weight.clone();reconstructed=torch.empty_like(weight)
    codes=torch.empty((out,features//group,m),dtype=torch.uint8,device=weight.device)
    for start in range(0,features,group):
        stop=start+group;block=upper[start:stop,start:stop]
        inverse_block=torch.linalg.solve_triangular(block,torch.eye(group,dtype=h.dtype,device=h.device),upper=True)
        metric=inverse_block@inverse_block.T
        current=work[:,start:stop].clone()
        for row in range(0,out,row_batch):
            codes[row:row+row_batch,start//group]=joint_beam_encode(current[row:row+row_batch],codebooks,metric[None],beam=beam)
        ids=codes[:,start//group]
        quantized=sum(codebooks[plane][ids[:,plane].long()] for plane in range(m))
        reconstructed[:,start:stop]=quantized
        feedback=(current-quantized)@inverse_block
        work[:,stop:]-=feedback@upper[start:stop,stop:]
    return codes,reconstructed,{'relative_diagonal_damping':damping,'absolute_diagonal_damping':float(damp),'beam':beam,'group':group,'feedback':'full step-averaged covariance, group block inverse-Cholesky; uniform decoder'}


def group_distortion(target,reconstruction,metrics,row_weights):
    error=target-reconstruction
    return torch.einsum('rgi,gij,rgj,r->',error,metrics,error,row_weights)


@torch.no_grad()
def fixed_code_wls(target,codes,books,metrics,row_weights,*,iterations=20,chunk=8192):
    """Conditional-centroid ALS on fixed codes and group8 metrics.

    target[R,G,8] is normalized folded W; metrics[G,8,8] is the summed
    step-weighted covariance for this subset. row_weights are row_scale^2.
    Supply just the active subset's book planes/codes. Inactive entries retain
    their old value. Small ridge stabilizes singular normal equations; every
    accepted sweep must decrease the original unregularized objective.
    """
    rows,groups,d=target.shape;m,k,_=books.shape
    if codes.shape!=(rows,groups,m) or metrics.shape!=(groups,d,d) or row_weights.shape!=(rows,):raise ValueError('WLS shape mismatch')
    result=books.clone();ids=codes.reshape(-1,m).long();original=target.reshape(-1,d)
    if not (row_weights>=0).all():raise ValueError('Negative row weights')
    total=rows*groups;normal=[]
    for plane in range(m):
        a=torch.zeros((k,d,d),dtype=books.dtype,device=books.device)
        for begin in range(0,total,chunk):
            end=min(begin+chunk,total);index=torch.arange(begin,end,device=books.device)
            weighted=metrics[index%groups]*row_weights[index//groups,None,None]
            a.index_add_(0,ids[begin:end,plane],weighted)
        normal.append(a)
    losses=[];all_mask=[True]*m
    recon=subset_reconstruction(result,codes,all_mask)
    losses.append(float(group_distortion(target,recon,metrics,row_weights)))
    for iteration in range(iterations):
        previous=result.clone();previous_recon=recon.clone()
        for plane in range(m):
            residual=(target-recon+result[plane][codes[...,plane].long()]).reshape(-1,d)
            b=torch.zeros((k,d),dtype=books.dtype,device=books.device)
            for begin in range(0,total,chunk):
                end=min(begin+chunk,total);index=torch.arange(begin,end,device=books.device)
                h=metrics[index%groups]*row_weights[index//groups,None,None]
                rhs=torch.einsum('ni,nij->nj',residual[begin:end],h)
                b.index_add_(0,ids[begin:end,plane],rhs)
            a=normal[plane];strength=a.diagonal(dim1=-2,dim2=-1).mean(-1)
            active=strength>0
            ridge=(strength[active]*1e-7).clamp_min(torch.finfo(books.dtype).tiny)
            stabilized=a[active]+ridge[:,None,None]*torch.eye(d,dtype=books.dtype,device=books.device)
            # Ridge toward the current value, preserving unidentifiable directions.
            rhs=b[active]+ridge[:,None]*result[plane,active]
            new=torch.linalg.solve(stabilized,rhs[...,None])[...,0]
            recon-=result[plane][codes[...,plane].long()]
            result[plane,active]=new
            recon+=result[plane][codes[...,plane].long()]
        loss=float(group_distortion(target,recon,metrics,row_weights))
        if not torch.isfinite(torch.tensor(loss)) or loss>losses[-1]:
            result=previous;recon=previous_recon;break
        losses.append(loss)
    return result,{'losses':losses,'accepted_sweeps':len(losses)-1,'requested_sweeps':iterations,'codes_fixed':True,'metric':'group-block-diagonal8x8, summed weighted reader steps'}
