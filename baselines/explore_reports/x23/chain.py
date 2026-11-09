import numpy as np
DR=["top","middle","bottom"];URDF=["","recolor_tabletop_visual_matching_1","recolor_tabletop_visual_matching_2","recolor_cabinet_visual_matching_1"]
def cfg(seed):
    r=np.random.RandomState(int(seed))
    d=r.choice(DR);i=int(r.choice(9));u=r.choice(URDF)
    return (str(d),i,URDF.index(u))
def chain(start,n,include_start=False):
    """official unseeded: items 1..n of chain from 2022 (construct-time reset(2022) is not an episode).
    seeded env i: start=7+i, include_start=True (first reset(seed) episode uses seed itself)."""
    out=[];s=start
    if include_start: out.append((s,cfg(s)))
    while len(out)<n:
        s=int(np.random.RandomState(s).randint(2**32)); out.append((s,cfg(s)))
    return out
if __name__=="__main__":
    c=chain(2022,100)
    for k,(s,x) in enumerate(c[:30],1): print(k,s,x)
    import collections
    print(len(set(x for _,x in c)),'distinct of 100;',len(set(x[1:] for _,x in c[:25])),'distinct (overlay,urdf) in first 25')
    print(collections.Counter(x[0] for _,x in c))
