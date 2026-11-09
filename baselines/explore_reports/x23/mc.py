"""Monte Carlo: how would an official-style run (5 envs, all on one fixed chain, ~20 items each) score for a random chain start,
using seeded-pool per-cell rates? Where does the real 2022 chain sit? Also bound: what is needed for 65."""
import json,numpy as np
from chain import *
J=json.load(open('configs.json'));ns=np.array(J['seeded_pool']['n']);ss=np.array(J['seeded_pool']['s']);rate=ss/ns
def pred(items):
    return np.mean([rate[DR.index(c[0]),c[1]] for c in items])
rng=np.random.RandomState(0)
def run(start,n_items):
    return pred([x for _,x in chain(start,n_items)])
P5=np.array([run(int(rng.randint(2**32)),20) for _ in range(20000)])
P1=np.array([run(int(rng.randint(2**32)),100) for _ in range(5000)])
c2022_5=run(2022,20);c2022_5b=run(2022,23);c2022_1=run(2022,100)
sd=np.sqrt(P5*(1-P5)/100)  # run noise (policy), 100 episodes
print('5env-style chain (20 items x5), predicted rate over random chain starts: mean %.3f sd %.3f min %.3f max %.3f'%(P5.mean(),P5.std(),P5.min(),P5.max()))
print(' share of random chains with predicted rate >= 0.60: %.4f ; >= 0.65: %.5f'%((P5>=.60).mean(),(P5>=.65).mean()))
print(' real 2022 chain (20 items): %.3f ; (23 items) %.3f ; percentile %.1f%%'%(c2022_5,c2022_5b,100*(P5<c2022_5).mean()))
print('1env-style (100 items) mean %.3f sd %.3f ; 2022 chain %.3f'%(P1.mean(),P1.std(),c2022_1))
# probability an observed 100-ep run >= 65 including binomial noise
b=np.random.RandomState(1).binomial(100,P5)/100
print(' with binomial noise: P(score>=0.65) = %.4f ; P(score>=0.60)=%.4f'%((b>=.65).mean(),(b>=.60).mean()))
# overlays: share of cells at ~0
print('idx rate',(ss.sum(0)/ns.sum(0)).round(3).tolist())
# if overlay idx 2 never drawn / idx 2 and 8 never drawn: uniform rate over the rest
r=rate.copy()
for drop in ([2],[2,8],[2,8,6]):
    keep=[i for i in range(9) if i not in drop];print('uniform over drawers x overlays without idx',drop,'-> %.3f'%r[:,keep].mean())
json.dump(dict(mc5_mean=float(P5.mean()),mc5_sd=float(P5.std()),p_ge_065_chain=float((P5>=.65).mean()),p_ge_065_run=float((b>=.65).mean()),c2022_5=float(c2022_5),c2022_1=float(c2022_1),mc1_mean=float(P1.mean())),open('mc.json','w'))
# chain cell composition for official 5env (items 1..22)
import collections
c=[x for _,x in chain(2022,22)]
print('official chain items 1..22 overlay idx counts',sorted(collections.Counter(x[1] for x in c).items()),'drawer',collections.Counter(x[0] for x in c))
print('per-cell rate in chain items 1..22:',[ (x[0][0],x[1],round(float(rate[DR.index(x[0]),x[1]]),2)) for x in c])
