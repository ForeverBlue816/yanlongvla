"""Calibration-only uniform reference(a) additive VQ fitting; inference is elsewhere."""
import hashlib
import torch
from vq_core import fixed_code_wls,gptq_encode_uniform,subset_reconstruction


def private_generator(name,device,seed=20260930):
    digest=hashlib.sha256(f'{seed}:{name}'.encode()).digest()
    return torch.Generator(device=device).manual_seed(int.from_bytes(digest[:8],'little')%(2**63))


@torch.no_grad()
def metric_assign(points,books,metrics,chunk=1024):
    ids=[]
    for begin in range(0,len(points),chunk):
        x=points[begin:begin+chunk];h=metrics[begin:begin+chunk]
        xh=torch.einsum('ni,nij->nj',x,h)
        costs=torch.einsum('ki,nij,kj->nk',books,h,books)-2*xh@books.T
        ids.append(costs.argmin(1))
    return torch.cat(ids)


@torch.no_grad()
def weighted_kmeans(points,metrics,weights,*,entries=256,iterations=10,generator):
    if len(points)<entries:raise ValueError('Insufficient points for requested codebook')
    d=points.shape[-1]
    centers=points[torch.randperm(len(points),generator=generator,device=points.device)[:entries]].clone()
    weighted_h=metrics*weights[:,None,None]
    rhs=torch.einsum('ni,nij->nj',points,weighted_h)
    history=[]
    for _ in range(iterations):
        ids=metric_assign(points,centers,metrics)
        a=torch.zeros((entries,d,d),device=points.device,dtype=points.dtype)
        b=torch.zeros_like(centers)
        a.index_add_(0,ids,weighted_h);b.index_add_(0,ids,rhs)
        strengths=a.diagonal(dim1=-2,dim2=-1).mean(-1);active=strengths>0
        ridge=(strengths[active]*1e-7).clamp_min(torch.finfo(points.dtype).tiny)
        regularized=a[active]+ridge[:,None,None]*torch.eye(d,device=points.device,dtype=points.dtype)
        new=centers.clone();new[active]=torch.linalg.solve(regularized,(b[active]+ridge[:,None]*centers[active])[...,None])[...,0]
        delta=points-new[ids]
        loss=float(torch.einsum('ni,nij,nj,n->',delta,metrics,delta,weights))
        if history and loss>history[-1]*(1+1e-6):break
        centers=new;history.append(loss)
    return centers,history


def group_blocks(h,group=8):
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or h.shape[0]%group:raise ValueError('Bad covariance shape')
    count=h.shape[0]//group
    # Diagonal8x8 blocks in the SAME RHT input basis as the folded weights.
    return torch.stack([h[g*group:(g+1)*group,g*group:(g+1)*group] for g in range(count)])


@torch.no_grad()
def fit_uniform(weight,hessian,*,planes,name,entries=256,beam=8,rounds=3,kmeans_iterations=10,als_iterations=20,sample_groups=65536,damping=.01):
    """Three encode/bookfit rounds, then re-encode; select by calibration full-H error.

    K-means is initialization only. Every deployed code assignment comes from
    joint beam+GPTQ, using all M indices. Original row scales and books are
    rounded to deployedFP16 before scoring/encoding. No held-out data enters.
    """
    if weight.ndim!=2 or weight.shape[1]%8 or hessian.shape!=(weight.shape[1],weight.shape[1]):raise ValueError('Input shape mismatch')
    if planes not in (1,2,3):raise ValueError('Only M1/2/3 are requested')
    out,features=weight.shape;groups=features//8
    scale=weight.float().square().mean(1).sqrt().clamp_min(torch.finfo(torch.float16).tiny).half()
    normalized=weight.float()/scale.float()[:,None]
    h=(hessian.float()+hessian.float().T)/2
    h=h/h.diagonal().mean().clamp_min(torch.finfo(h.dtype).eps)
    blocks=group_blocks(h);target=normalized.reshape(out,groups,8)
    generator=private_generator(name,weight.device)
    count=min(out*groups,sample_groups)
    sample=torch.randperm(out*groups,generator=generator,device=weight.device)[:count]
    points=target.reshape(-1,8)[sample]
    metrics=blocks[sample%groups];row_weights=scale.float().square();sample_weights=row_weights[sample//groups]
    sample_weights=sample_weights/sample_weights.mean().clamp_min(torch.finfo(sample_weights.dtype).tiny)
    residual=points.clone();book_list=[];initial=[]
    for plane in range(planes):
        book,history=weighted_kmeans(residual,metrics,sample_weights,entries=entries,iterations=kmeans_iterations,generator=generator)
        book=book.half().float();book_list.append(book);initial.append(history)
        ids=metric_assign(residual,book,metrics);residual-=book[ids]
    books=torch.stack(book_list)
    best=None;history=[]
    energy=float(torch.sum((weight.float()@h)*weight.float()))
    for iteration in range(rounds+1):
        codes,q,encoding=gptq_encode_uniform(normalized,h,books,beam=beam,damping=damping)
        deployed=(q*scale.float()[:,None]).to(weight.dtype).float()
        error=weight.float()-deployed
        loss=float(torch.sum((error@h)*error))
        current=dict(iteration=iteration,full_covariance_error=loss,full_covariance_relative_error=loss/max(energy,1e-30))
        if best is None or loss<best['loss']:
            best=dict(loss=loss,iteration=iteration,codes=codes.clone(),books=books.clone())
        if iteration<rounds:
            fitted,als=fixed_code_wls(target,codes,books,blocks,row_weights,iterations=als_iterations)
            books=fitted.half().float();current['als']=als
        history.append(current)
    books=best['books'].half();codes=best['codes']
    q=subset_reconstruction(books.float(),codes,[True]*planes).reshape(out,features)*scale.float()[:,None]
    q=q.to(weight.dtype).float();error=weight.float()-q
    metadata={'name':name,'variant':'a','planes':planes,'entries':entries,'group':8,'beam':beam,'rounds':rounds,'selected_round':best['iteration'],
              'selection':'minimum full step-averaged covariance error on calibration; no heldout inputs',
              'covariance_relative_error':best['loss']/max(energy,1e-30),
              'relative_weight_frobenius_error':float(torch.linalg.vector_norm(error)/torch.linalg.vector_norm(weight.float()).clamp_min(1e-30)),
              'initialization_sample_groups':count,'kmeans_iterations':kmeans_iterations,'initialization_loss':initial,'history':history,'encoding':encoding,
              'row_scale':'FP16 per-output-row RMS, minimum FP16 normal; normalization uses rounded stored scale',
              'metric':'full uncentered equal-step-mean H for GPTQ/selection; group8diagonal blocks for book WLS','damping':damping}
    tensors={'codebooks':books.contiguous(),'codes':codes.permute(2,0,1).contiguous(),'row_scale':scale.contiguous(),
             'masks':torch.full((10,),(1<<planes)-1,dtype=torch.uint8,device=weight.device)}
    return tensors,metadata
