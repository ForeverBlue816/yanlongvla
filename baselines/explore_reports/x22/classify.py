import cv2,json,glob,os,numpy as np,collections
B='/scratch/d250009/pi05-handover/runs/groot_baselines/s5b/'
def f0(p):
    c=cv2.VideoCapture(p);ok,f=c.read();c.release();assert ok,p
    return cv2.resize(f,(128,96),interpolation=cv2.INTER_AREA).astype(np.float32)
d=json.load(open(B+'eval/fpsl/widowx_stack_cube.json'))
fs=sorted(glob.glob(B+'eval/fpsl/videos/widowx_stack_cube/*.mp4'),key=os.path.getmtime)
T=np.stack([f0(f) for f in fs]);TL=np.array([e['layout'] for e in d])
# template self-check: leave-one-out
ok=0
for i in range(len(T)):
    dist=((T-T[i])**2).mean((1,2,3));dist[i]=1e18;ok+=TL[np.argmin(dist)]==TL[i]
print('LOO acc',ok,len(T))
# margin
res={}
for arm in ['fp_fa','fp_fa1']:
  for r in range(1,7):
    fl=sorted(glob.glob(B+f'official/{arm}/r{r}/videos/widowx_stack_cube/*.mp4'))
    rows=[]
    for f in fl:
        x=f0(f);dist=((T-x)**2).mean((1,2,3))
        # best distance per layout
        per=np.array([dist[TL==l].min() for l in range(24)]);o=np.argsort(per)
        rows.append(dict(layout=int(o[0]),d1=float(per[o[0]]),d2=float(per[o[1]]),succ=f.endswith('_s1.mp4'),mtime=os.path.getmtime(f)))
    res[f'{arm}/r{r}']=rows
json.dump(res,open('episodes.json','w'))
