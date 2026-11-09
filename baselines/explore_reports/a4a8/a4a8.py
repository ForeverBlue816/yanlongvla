import json, itertools, math, numpy as np
from pathlib import Path
B='/scratch/d250009/pi05-handover/runs/groot_baselines/'
TASKS=('widowx_spoon_on_towel','widowx_carrot_on_plate','widowx_stack_cube','widowx_put_eggplant_in_basket','widowx_put_eggplant_in_sink','widowx_open_drawer','widowx_close_drawer')
ROWS=['fp','qtip_w2','quip_w2','vptq_w2','aqlm_w2']
D={r:{t:{x['episode']:x for x in json.loads(Path(B+f's5b/eval/{r}/{t}.json').read_text())} for t in TASKS} for r in ROWS}
def holm(ps,a=0.05):
    o=sorted(range(len(ps)),key=lambda i:ps[i]);ok=[False]*len(ps)
    for k,i in enumerate(o):
        if ps[i]<=a/(len(ps)-k): ok[i]=True
        else: break
    return ok
def boot(a,b,tasks,N=20000,seed=20260930):
    cl={}
    for t in tasks:
        d={}
        for e,x in D[a][t].items():
            y=D[b][t][e]; assert (x['layout'],x['seed'])==(y['layout'],y['seed'])
            d.setdefault(x['layout'],[]).append(float(x['success'])-float(y['success']))
        cl[t]=[(sum(v),len(v)) for v in d.values()]
    # per-task point diff
    pt=np.mean([sum(s for s,_ in cl[t])/sum(n for _,n in cl[t]) for t in tasks])*100
    rng=np.random.default_rng(seed); diffs=np.empty(N)
    arr={t:np.array(cl[t]) for t in tasks}
    for i in range(N):
        td=[]
        for t in tasks:
            A=arr[t]; p=A[rng.integers(0,len(A),len(A))]; td.append(p[:,0].sum()/p[:,1].sum())
        diffs[i]=np.mean(td)*100
    lo,hi=np.quantile(diffs,[.025,.975])
    p=min(1,2*min((np.sum(diffs<=0)+1)/(N+1),(np.sum(diffs>=0)+1)/(N+1)))
    return pt,lo,hi,p
def part1(tasks,name):
    res=[(a,b)+boot(a,b,tasks) for a,b in itertools.combinations(ROWS,2)]
    ok=holm([r[5] for r in res]); print('##',name)
    print('pair | diff pp | 95% CI | raw p | Holm')
    for r,h in zip(res,ok): print(f'{r[0]}-{r[1]} | {r[2]:+.1f} | [{r[3]:+.1f},{r[4]:+.1f}] | {r[5]:.4f} | {"sig" if h else "ns"}')
part1(TASKS,'Bridge all 7 tasks')
part1([t for t in TASKS if t not in('widowx_carrot_on_plate','widowx_stack_cube')],'Bridge 5 tasks')
# Part 2
E=B+'eval/'
def load(d,f):
    return {(r['suite'],r['name']):(bool(r['success']),(r['suite'],r.get('base_task',r['name']))) for r in map(json.loads,open(E+d+'/'+f))}
def est(pairs):  # pairs: list of (diff, cluster)
    n=len(pairs); m=sum(d for d,_ in pairs)/n; g={}
    for d,c in pairs: x=g.setdefault(c,[0,0]); x[0]+=d; x[1]+=1
    G=len(g); se=math.sqrt(G/(G-1)*sum((s-m*k)**2 for s,k in g.values()))/n
    return 100*m,100*se,n,G
def cmp(q,f,fn):
    Q,F=load(q,fn),load(f,fn); return {k:(int(Q[k][0])-int(F[k][0]),Q[k][1]) for k in Q}
c7=cmp('qtip_w2','fp','plus_1000.jsonl'); c8=cmp('qtip_w2_seed8','fp_seed8','plus_1000.jsonl'); cd=cmp('qtip_w2_disjoint500','fp_disjoint500','plus_500.jsonl')
assert not set(c7)&set(cd); print('units',len(c7),len(c8),len(cd),set(c7)==set(c8))
def show(n,m,se,*x): print(f'{n}: {m:+.2f} pp, se {se:.2f}, 95% [{m-1.96*se:+.2f},{m+1.96*se:+.2f}] n={x}')
r7,r8,rd=est(list(c7.values())),est(list(c8.values())),est(list(cd.values()))
for n,r in(('seed7 1000',r7),('seed8 1000',r8),('disjoint500',rd)): show(n,r[0],r[1],r[2],r[3])
ra=est(list(c7.values())+list(cd.values())); show('(a) pooled 1500 (seed7+disjoint)',ra[0],ra[1],ra[2],ra[3])
rs=[r7,r8,rd]; w=[1/r[1]**2 for r in rs]; m=sum(wi*r[0] for wi,r in zip(w,rs))/sum(w); se=1/math.sqrt(sum(w))
show('(b) IVW of 3, treated independent (se too small)',m,se)
# conservative: seed7/seed8 averaged per unit, then IVW with disjoint (valid: seeds average, units shared)
avg={k:((c7[k][0]+c8[k][0])/2,c7[k][1]) for k in c7}; ra2=est(list(avg.values())+list(cd.values())); show('(b2) 1500 units, seed-averaged on shared 1000 + disjoint',ra2[0],ra2[1])
