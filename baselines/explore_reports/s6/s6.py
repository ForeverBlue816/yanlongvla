"""S6: shrink coefficient c=<E,W>/<W,W> per tensor for 2-bit Bridge methods. CPU only."""
import json, glob, os, re, statistics as st, torch
from safetensors import safe_open
G='/scratch/d250009/pi05-handover/runs/groot_baselines/'
FP=G+'s5bq/checkpoints/bridge/model.safetensors'
keys=list(json.load(open(G+'s5bq/perturb/aqlm_w2_error.json')))
fp=safe_open(FP,'pt')
def stats(W,Wq):
    W=W.double(); E=Wq.double()-W
    ww=(W*W).sum().item(); c=(E*W).sum().item()/ww
    en=E.norm().item(); wn=ww**.5
    orth=(E-c*W).norm().item()
    return dict(c=c,rel=en/wn,orth_rel=orth/wn)
def grp(k):
    g='backbone' if k.startswith('backbone') else 'action_head'
    n=k
    if re.search(r'(to_q|to_k|q_proj|k_proj)\.weight$',n): t='qk'
    elif re.search(r'(to_v|to_out\.0|v_proj|o_proj|out_proj|attn\.proj)\.weight$',n) or re.search(r'attn.*(proj|out)\.weight$',n): t='vo'
    elif re.search(r'(mlp|ff|fc)',n): t='mlp'
    else: t='other'
    return g,t
res={}
# source of quantized backbone weights: q/<method>/backbone/*.safetensors (key 'weight'); name from sidecar json
def from_q(m):
    out={}
    for j in glob.glob(G+f'q/{m}/backbone/*.json'):
        d=json.load(open(j))
        if 'name' not in d: continue
        k=d['name']+'.weight'
        sf=j[:-5]+'.safetensors'
        if not os.path.exists(sf): continue
        with safe_open(sf,'pt') as f: Wq=f.get_tensor('weight')
        if k in keys: out[k]=stats(fp.get_tensor(k),Wq)
    return out
res['aqlm_w2_backbone']=from_q('aqlm_w2')
res['qtip_w2_backbone']=from_q('qtip_w2')
res['quip_w2_backbone_fromq']=from_q('quip_w2')
# QuIP# full Bridge checkpoint (432 keys)
qc=safe_open(G+'s5bq/ckpt/quip_w2/bridge/model.safetensors','pt')
res['quip_w2_ckpt']={k:stats(fp.get_tensor(k),qc.get_tensor(k)) for k in keys}
# AQLM error json: rel only (no c)
ej=json.load(open(G+'s5bq/perturb/aqlm_w2_error.json'))
res['aqlm_w2_errjson']={k:dict(rel=v['err_rms']/v['w_rms']) for k,v in ej.items()}
# summarise
def summ(d,field):
    out={}
    for gname,sel in [('backbone',lambda k:grp(k)[0]=='backbone'),('action_head',lambda k:grp(k)[0]=='action_head')]+[(f'{g}/{t}',(lambda g,t:lambda k:grp(k)==(g,t))(g,t)) for g in('backbone','action_head') for t in('qk','vo','mlp','other')]+[('all',lambda k:True)]:
        v=[x[field] for k,x in d.items() if sel(k) and field in x]
        if v: out[gname]=dict(n=len(v),mean=st.mean(v),median=st.median(v),min=min(v),max=max(v))
    return out
summary={m:{f:summ(d,f) for f in ('c','rel','orth_rel')} for m,d in res.items()}
# consistency checks
chk={}
a=res['aqlm_w2_backbone']; e=res['aqlm_w2_errjson']
chk['aqlm_backbone_rel_computed_vs_errjson_maxabsdiff']=max(abs(a[k]['rel']-e[k]['rel']) for k in a)
q=res['quip_w2_backbone_fromq']; qq=res['quip_w2_ckpt']
chk['quip_backbone_c_q_vs_ckpt_maxabsdiff']=max(abs(q[k]['c']-qq[k]['c']) for k in q) if q else None
json.dump(dict(per_tensor=res,summary=summary,checks=chk),open('/scratch/d250009/pi05-handover/runs/groot_baselines/explore/s6/s6.json','w'),indent=1)
print(json.dumps(chk,indent=1))
for m,s in summary.items():
    print(m)
    for f,v in s.items():
        for g,x in v.items(): print(' ',f,g,x['n'],*(f"{x[y]:.4f}" for y in('mean','median','min','max')))
