import torch, json, glob, numpy as np, sys
from safetensors import safe_open
torch.set_num_threads(8)
R='/scratch/d250009/pi05-handover/runs/groot_baselines/s5bq/'
fp=safe_open(R+'checkpoints/bridge/model.safetensors','pt'); q=safe_open(R+'ckpt/quip_w2/bridge/model.safetensors','pt')
orth=safe_open(R+'ckpt/orth/bridge/model.safetensors','pt')
keys=set(json.load(open(R+'perturb/aqlm_w2_error.json')))
def hfile(root,part,n): return f'{R}{root}/{part}/hessians/'+n.replace('.','_')+'.pt'
def stats(E,W,H):
    WH=(W@H).double(); EH=(E@H).double()
    wHw=(WH*W).sum(); eHw=(EH*W).sum(); eHe=(EH*E).sum()
    return wHw.item(),eHw.item(),eHe.item()
out={}; maxdev=0
for k in sorted(keys):
    part='backbone' if k.startswith('backbone') else 'head_bridge'
    n=k[:-7]
    W=fp.get_tensor(k).float(); Wq=q.get_tensor(k).float(); E=Wq-W
    ww=(W.double()**2).sum().item(); cw=(E.double()*W.double()).sum().item()/ww
    Ep=E-cw*W
    r=dict(shape=list(W.shape),c_w=cw,rel_w=(E.double()**2).sum().item()/ww)
    ok=True
    for tag,root in (('cal','calibration'),('held','calibration_heldout')):
        try: H=torch.load(hfile(root,part,n),weights_only=False)['covariance']
        except Exception as e: ok=False; continue
        wHw,eHw,eHe=stats(E,W,H); ca=eHw/wHw
        _,pHw,pHp=stats(Ep,W,H)
        r[tag]=dict(c_a=ca,rel_out=eHe/wHw,par_rel=ca*ca,perp_rel=eHe/wHw-ca*ca,
            orth_c_a=pHw/wHw,orth_c_a_check=ca-cw,orth_rel_out=pHp/wHw,orth_par_rel=(pHw/wHw)**2,
            cos_out=eHw/(wHw*eHe)**.5)
        maxdev=max(maxdev,abs(pHw/wHw-(ca-cw)))
    out[k]=r
    print(k,r.get('cal',{}).get('c_a'),cw,flush=True)
# orth checkpoint check on 5 tensors
chk=[]
for k in sorted(keys)[::90]:
    W=fp.get_tensor(k).float(); E=q.get_tensor(k).float()-W; cw=(E*W).sum().item()/(W*W).sum().item()
    chk.append((k,(orth.get_tensor(k).float()-(W+E-cw*W)).abs().max().item()))
def grp(k):
    if k.startswith('action_head'):
        return 'head_attn' if '.attn' in k else 'head_mlp' if '.ff.' in k else 'head_other'
    if '.visual.' in k: return 'vision'
    if 'language_model' in k:
        if any(s in k for s in ('v_proj','o_proj')): return 'lm_vo'
        if any(s in k for s in ('q_proj','k_proj')): return 'lm_qk'
        return 'lm_mlp'
    return 'other'
G={}
for k,r in out.items(): G.setdefault(grp(k),[]).append(r)
summ={}
for g,rs in G.items():
    s={'n':len(rs),'c_w_mean':np.mean([r['c_w'] for r in rs]),'c_w_median':np.median([r['c_w'] for r in rs])}
    for tag in ('cal','held'):
        rr=[r[tag] for r in rs if tag in r]
        if not rr: continue
        s['n_'+tag]=len(rr)
        for f in ('c_a','orth_c_a','rel_out','orth_rel_out','par_rel','orth_par_rel','cos_out'):
            v=[x[f] for x in rr]; s[f'{tag}_{f}_mean']=float(np.mean(v)); s[f'{tag}_{f}_median']=float(np.median(v))
    summ[g]=s
json.dump(dict(per_tensor=out,groups=summ,max_dev_orth_ca_check=maxdev,orth_ckpt_check=chk),open('x25.json','w'),indent=1)
print(json.dumps(summ,indent=1)); print(maxdev,chk)
