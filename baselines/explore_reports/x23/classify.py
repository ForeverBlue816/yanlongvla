"""Classify each Fractal open_drawer episode (first video frame) into (drawer, overlay idx).
Drawer: burned-in instruction text strip, nearest template. Overlay idx (robot init pose + background): top 256 rows, nearest template.
Templates: the two 1-env official runs (fp1e1, fp1e2), whose configs are the simulated 2022 chain (validated: 100/100)."""
import numpy as np,json
from frames2 import *; from chain import *
ARMS=['fp_fp1e1','fp_fp1e2','fp_fpof1','fp_fpof2','fp_fpof3','fp_fpgv1','fp_fpgv2','fp_fpgv3','fp_fpfa','fp_fpbig','fp_fpseed8','fp_fpseed9','fp_fprr','fp']
ch=[x for _,x in chain(2022,100)]
TI=[];TT=[];TD=[];TX=[]
out={}
data={}
for a in ARMS:
    fs,succ,L=load(a)
    vs=[f.endswith('_s1.mp4') for f in fs]
    # fix adjacent mtime swaps so frame order matches results order
    fs=list(fs);i=0;fix=0
    while i<len(fs)-1:
        if vs[i]!=succ[i] and vs[i+1]!=succ[i+1] and vs[i]==succ[i+1] and vs[i+1]==succ[i]:
            fs[i],fs[i+1]=fs[i+1],fs[i];vs[i],vs[i+1]=vs[i+1],vs[i];fix+=1;i+=2
        else:i+=1
    mism=sum(a_!=b_ for a_,b_ in zip(vs,succ))
    F=[feats(f) for f in fs]
    data[a]=dict(I=np.stack([x for x,y in F]),T=np.stack([y for x,y in F]),succ=succ,L=L,files=fs,fix=fix,mism=mism)
    print(a,len(fs),'swap fixes',fix,'residual mismatch',mism,flush=True)
for a in ['fp_fp1e1','fp_fp1e2']:
    for k in range(100):
        TI.append(data[a]['I'][k]);TT.append(data[a]['T'][k]);TD.append(DR.index(ch[k][0]));TX.append(ch[k][1])
TI=np.stack(TI);TT=np.stack(TT);TD=np.array(TD);TX=np.array(TX)
res={}
for a in ARMS:
    I=data[a]['I'];T=data[a]['T'];eps=[]
    for k in range(len(I)):
        dt=((TT-T[k])**2).mean((1,2,3));di=((TI-I[k])**2).mean((1,2,3))
        # for self-templates, exclude exact self (same arm, same index) so that 1e runs are classified leave-one-out
        if a in('fp_fp1e1','fp_fp1e2'):
            off=0 if a=='fp_fp1e1' else 100
            dt[off+k]=1e18;di[off+k]=1e18
            tw=(100 if a=='fp_fp1e1' else 0)+k  # twin has identical config -> also exclude to be honest
            dt[tw]=1e18;di[tw]=1e18
        j=np.argmin(dt);m=np.argmin(di)
        eps.append(dict(drawer=DR[TD[j]],idx=int(TX[m]),dt=float(dt[j]),di=float(di[m]),succ=bool(data[a]['succ'][k]),length=int(data[a]['L'][k]),file=data[a]['files'][k].split('/')[-1]))
    res[a]=eps
    print(a,'max dt %.1f max di %.1f'%(max(e['dt'] for e in eps),max(e['di'] for e in eps)),'succ',sum(e['succ'] for e in eps),flush=True)
# self-check of the 1e runs against the chain
bad=0
for a in['fp_fp1e1','fp_fp1e2']:
    for k,e in enumerate(res[a]): bad+=(e['drawer'],e['idx'])!=ch[k][:2]
print('1e runs: classification vs simulated chain mismatches (leave-one-out, twin excluded):',bad,'of 200')
json.dump(res,open('episodes.json','w'))
