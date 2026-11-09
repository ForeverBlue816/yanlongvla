import json,numpy as np,collections
from scipy.stats import chisquare
R=json.load(open('episodes.json'))
out={};
for k,v in R.items(): print(k,len(v),sum(e['succ'] for e in v),'maxd1',round(max(e['d1'] for e in v),1),'min ratio',round(min(e['d2']/max(e['d1'],1e-9) for e in v),1))
S={}
for arm in ['fp_fa','fp_fa1']:
    eps=[e for r in range(1,7) for e in R[f'{arm}/r{r}']]
    n=np.zeros(24);s=np.zeros(24)
    for e in eps: n[e['layout']]+=1;s[e['layout']]+=e['succ']
    rate=s.sum()/n.sum()
    uni=np.where(n>0,s/np.maximum(n,1),np.nan)
    # reweight: uniform over layouts seen; unseen layouts filled with pooled seeded rate later
    S[arm]=dict(n=n.tolist(),s=s.tolist(),rate=rate,chi2=chisquare(n),per_run=[np.bincount([e['layout'] for e in R[f'{arm}/r{r}']],minlength=24).tolist() for r in range(1,7)],uni_seen=float(np.nanmean(uni)),nseen=int((n>0).sum()))
    print(arm,'rate',rate,'chi',S[arm]['chi2'],'seen',S[arm]['nseen'],'uni_seen',S[arm]['uni_seen'])
    print(' n',n.astype(int).tolist());print(' s',s.astype(int).tolist())
# seeded per-layout rate
d=json.load(open('/scratch/d250009/pi05-handover/runs/groot_baselines/s5b/eval/fpsl/widowx_stack_cube.json'))
sn=np.zeros(24);ss=np.zeros(24)
for e in d: sn[e['layout']]+=1;ss[e['layout']]+=e['success']
sr=ss/sn;print('seeded rate',ss.sum()/sn.sum(),sr.round(2).tolist())
for arm in S:
    n=np.array(S[arm]['n']);
    S[arm]['reweight_seededrates_on_arm_mix']=float((n*sr).sum()/n.sum())  # expected rate if per-layout success = seeded
    S[arm]['pred_arm_mix_seeded']=S[arm]['reweight_seededrates_on_arm_mix']
    print(arm,'seeded per-layout rates under arm layout mix ->',S[arm]['pred_arm_mix_seeded'], 'uniform seeded', sr.mean())
S['seeded']=dict(n=sn.tolist(),s=ss.tolist(),rate=float(ss.sum()/sn.sum()))
for a in ['fp_fa','fp_fa1']: S[a]['chi2']=[float(x) for x in S[a]['chi2']]
json.dump(S,open('layouts.json','w'),indent=1)
