"""Decode which chain item each observed 5-env episode is. Envs step in lock-step; an episode of length l that started at t ends at t+l.
Completion order = order of (end time, env index). Max length 300. Config of env e's k-th episode = chain_e[k]."""
import json,sys,numpy as np
from chain import *
sys.setrecursionlimit(10000)
SLACK=int(__import__("os").environ.get("SLACK","0"))
R=json.load(open('episodes.json'))
def chains(arm,ne=5):
    if arm in OFFICIAL: return [[x for _,x in chain(2022,160)] for _ in range(ne)]
    seed=SEED[arm]; return [[x for _,x in chain(seed+i,160,include_start=True)] for i in range(ne)]
OFFICIAL=['fp_fpof1','fp_fpof2','fp_fpof3','fp_fpgv1','fp_fpgv2','fp_fpgv3']
SEED={'fp_fpfa':7,'fp_fpbig':7,'fp_fpseed8':8,'fp_fpseed9':9,'fp_fprr':7,'fp':7}
def decode(arm,maxsol=200000,ne=5):
    eps=R[arm];C=chains(arm,ne);N=len(eps)
    obs=[(DR.index(e['drawer']),e['idx']) for e in eps];ln=[e['length'] for e in eps]
    cobs=[[(DR.index(c[0]),c[1]) for c in cc] for cc in C]
    dead=set();sols=[];nsol=[0]
    from functools import lru_cache
    def rec(n,state,seq):
        # state: tuple of (t_e,k_e)
        if n==N:
            nsol[0]+=1
            if len(sols)<2000: sols.append(list(seq))
            return True
        key=(n,state)
        if key in dead: return False
        found=False;prevT=max([0]+[0])
        ok_any=False
        for e in range(ne):
            t,k=state[e]
            if cobs[e][k]!=obs[n]: continue
            T=t+ln[n]
            if n>0 and T<lastT[n-1]-SLACK:  # completion times must be non-decreasing
                continue
            # no other env can have been overdue
            bad=False
            for f in range(ne):
                if f!=e and state[f][0]+300+SLACK<T: bad=True;break
            if bad: continue
            ns=list(state);ns[e]=(T,k+1)
            lastT[n]=T
            seq.append((e,k))
            r=rec(n+1,tuple(ns),seq)
            seq.pop()
            found|=r
            if nsol[0]>=maxsol: break
        if not found: dead.add(key)
        return found
    lastT=[0]*(N+1)
    ok=rec(0,tuple((0,0) for _ in range(ne)),[])
    return ok,nsol[0],sols
if __name__=="__main__":
    out={}
    for arm in OFFICIAL+['fp_fpfa','fp_fpseed8','fp_fpseed9','fp']+[]:
        ok,ns,sols=decode(arm,maxsol=5000)
        # per-episode chain index sets over sampled solutions (env-agnostic: config k)
        ks=[sorted(set(s[n][1] for s in sols)) for n in range(len(R[arm]))] if sols else None
        first=sols[0] if sols else None
        print(arm,'feasible',ok,'nsol(capped)',ns,'max k',max(max(x) for x in ks)+1 if ks else None,'ambiguous eps',sum(len(x)>1 for x in ks) if ks else None,flush=True)
        out[arm]=dict(feasible=ok,nsol=ns,ks=ks,first=first)
    json.dump(out,open('decode.json','w'))
