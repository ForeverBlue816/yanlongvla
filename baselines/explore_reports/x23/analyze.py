import json,numpy as np,collections
from chain import *
R=json.load(open('episodes.json'));Dc=json.load(open('decode.json'))
NAME={'fp_fp1e1':'1env r1','fp_fp1e2':'1env r2','fp_fpof1':'5env official r1','fp_fpof2':'5env official r2','fp_fpof3':'5env official r3','fp_fpgv1':'5env pinned-venv r1','fp_fpgv2':'5env pinned-venv r2','fp_fpgv3':'5env pinned-venv r3','fp':'seeded 7 (SDPA)','fp_fpfa':'seeded 7 (flash)','fp_fprr':'seeded 7 rerun','fp_fpseed8':'seeded 8','fp_fpseed9':'seeded 9','fp_fpbig':'seeded 7, 500 ep'}
GROUPS={'1env (2 runs)':['fp_fp1e1','fp_fp1e2'],'5env official (3 runs)':['fp_fpof1','fp_fpof2','fp_fpof3'],'5env pinned venv (3 runs)':['fp_fpgv1','fp_fpgv2','fp_fpgv3'],
 'seeded 5env 100 ep (fp,fpfa,fprr,seed8,seed9)':['fp','fp_fpfa','fp_fprr','fp_fpseed8','fp_fpseed9'],'seeded 7 500 ep (fpbig)':['fp_fpbig']}
def cells(arms):
    n=np.zeros((3,9));s=np.zeros((3,9))
    for a in arms:
        for e in R[a]: n[DR.index(e['drawer']),e['idx']]+=1;s[DR.index(e['drawer']),e['idx']]+=e['succ']
    return n,s
out={'per_arm':{},'per_group':{}}
# full 108-config per episode when decodable
def full_cfg(a):
    d=Dc.get(a)
    if not d or not d.get('first'):return None
    C=[[x for _,x in chain(2022,160)] for _ in range(5)] if a in('fp_fpof1','fp_fpof2','fp_fpof3','fp_fpgv1','fp_fpgv2','fp_fpgv3') else None
    if C is None:
        seed={'fp_fpfa':7,'fp_fpseed8':8,'fp_fpseed9':9,'fp':7}[a];C=[[x for _,x in chain(seed+i,160,include_start=True)] for i in range(5)]
    return [C[e][k] for e,k in d['first']]
for a in R:
    n,s=cells([a]);fc=full_cfg(a)
    row=dict(n=int(n.sum()),succ=int(s.sum()),rate=s.sum()/n.sum(),cells_seen=int((n>0).sum()),drawer_n=n.sum(1).tolist(),drawer_succ=s.sum(1).tolist(),idx_n=n.sum(0).tolist(),idx_succ=s.sum(0).tolist())
    if fc: row['configs108_seen']=len(set(fc))
    out['per_arm'][a]=row
    print(NAME[a].ljust(22),row['n'],row['succ'],'cells27 seen',row['cells_seen'],'cfg108',row.get('configs108_seen'),'drawer n',n.sum(1).astype(int).tolist(),'rate by drawer',(s.sum(1)/n.sum(1)).round(2).tolist())
# 1env exact
ch=[x for _,x in chain(2022,100)]
print('1env chain: distinct 108 configs',len(set(ch)),'distinct 27 cells',len(set(x[:2] for x in ch)))
for k in (21,22,23,24): 
    c=[x for _,x in chain(2022,k)];print('chain items 1..%d'%k,'distinct108',len(set(c)),'cells27',len(set(x[:2] for x in c)))
# seeded rate table (independent draws): pool of seeded arms
SEEDED=['fp','fp_fpfa','fp_fprr','fp_fpseed8','fp_fpseed9','fp_fpbig']
ns,ss=cells(SEEDED)
print('seeded pool n',ns.sum(),'succ',ss.sum(),'rate',ss.sum()/ns.sum())
rate_s=ss/np.maximum(ns,1)
print('cell rates (rows top/middle/bottom, cols idx0..8):\n',rate_s.round(2));print('cell n:\n',ns.astype(int))
print('drawer rates seeded',(ss.sum(1)/ns.sum(1)).round(3).tolist(),'n',ns.sum(1).astype(int).tolist())
print('idx rates seeded',(ss.sum(0)/ns.sum(0)).round(3).tolist(),'n',ns.sum(0).astype(int).tolist())
out['seeded_pool']=dict(n=ns.tolist(),s=ss.tolist(),rate=float(ss.sum()/ns.sum()))
# uniform-cell reweighting
def se_uniform(n,s):
    r=s/np.maximum(n,1);var=(r*(1-r)/np.maximum(n,1)).sum()/27**2
    return float(r.mean()),float(np.sqrt(var))
print('uniform over 27 cells, seeded pool: %.3f +- %.3f'%se_uniform(ns,ss))
out['uniform_seeded']=se_uniform(ns,ss)
# per group: actual rate, predicted by seeded cell rates on the group's mix, group's own cells uniform
print('\nGROUP table: actual | seeded-rate prediction on group mix | uniform with group own cell rates (cells seen) | uniform with seeded rates')
for g,arms in GROUPS.items():
    n,s=cells(arms);act=s.sum()/n.sum();pred=float((n*rate_s).sum()/n.sum())
    own=np.where(n>0,s/np.maximum(n,1),np.nan);un_own_seen=float(np.nanmean(own))
    # fill unseen cells with seeded rates
    fill=np.where(n>0,s/np.maximum(n,1),rate_s);un_fill=float(fill.mean())
    out['per_group'][g]=dict(n=int(n.sum()),succ=int(s.sum()),rate=float(act),pred_from_seeded_rates=pred,uniform_own_seen=un_own_seen,cells_seen=int((n>0).sum()),uniform_fill_seeded=un_fill,drawer_share=(n.sum(1)/n.sum()).tolist(),cell_n=n.tolist(),cell_s=s.tolist())
    print(g.ljust(48),'n=%d rate %.3f | pred %.3f | uni own(%d cells) %.3f | uni filled %.3f'%(n.sum(),act,pred,(n>0).sum(),un_own_seen,un_fill),'drawer share',(n.sum(1)/n.sum()).round(2).tolist())
# Tilt: what mix gives 65 using seeded cell rates (27 cells), min-KL to uniform
from scipy.optimize import brentq
r=rate_s.flatten()
def mixrate(l):
    w=np.exp(l*(r-r.mean()));w/=w.sum();return float((w*r).sum()),w
print('max cell rate',r.max().round(2),'top5',np.sort(r)[-5:].round(2))
for tgt in (0.55,0.60,0.65):
    try:
        l=brentq(lambda l:mixrate(l)[0]-tgt,0,200);m,w=mixrate(l);print('target',tgt,'tilt',round(l,1),'max weight',w.max().round(3),'cells with w>5%',int((w>.05).sum()),'KL vs uniform',float((w*np.log(w*27+1e-12)).sum().round(2)))
    except Exception as e: print(tgt,'no',e)
# drawer/idx marginal: what would it need? shares
dr=ss.sum(1)/ns.sum(1);print('drawer-only: best single drawer',dr.max().round(3))
# bootstrap: best-case upper bound for any mix over the 3 drawers
json.dump(out,open('configs.json','w'),indent=1,default=float)
# chain visited cells in order for official 5-env: share by cell
c5=[x for _,x in chain(2022,24)]
print('official 5-env visited chain items 1..24, drawer share',collections.Counter(x[0] for x in c5))
