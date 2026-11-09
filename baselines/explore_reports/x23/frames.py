import cv2,glob,os,re,numpy as np,json,ast
E='/scratch/d250009/pi05-handover/runs/groot_baselines/s5/eval/'
def f0(p):
    c=cv2.VideoCapture(p);ok,f=c.read();c.release();assert ok,p
    return cv2.resize(f,(128,96),interpolation=cv2.INTER_AREA).astype(np.float32)
def results(arm):
    t=open(E+arm+'/open_drawer.log').read()
    m=re.search(r"results:\s+\('[^']*', (\[.*?\]), \{'episode_lengths': (\[.*?\])",t,re.S)
    return [bool(x) for x in ast.literal_eval(m.group(1))],ast.literal_eval(m.group(2))
def load(arm):
    fs=sorted(glob.glob(E+arm+'/videos/open_drawer/*.mp4'),key=os.path.getmtime)
    succ,L=results(arm)
    return fs,succ,L
if __name__=="__main__":
    out={}
    for arm in ['fp_fp1e1','fp_fp1e2','fp_fpof1','fp_fpof2','fp_fpof3','fp_fpgv1','fp_fpgv2','fp_fpgv3','fp_fpfa','fp_fpbig','fp_fpseed8','fp_fpseed9','fp_fprr','fp']:
        try: fs,succ,L=load(arm)
        except Exception as e: print(arm,'skip',e);continue
        vs=[f.endswith('_s1.mp4') for f in fs]
        print(arm,len(fs),len(succ),'mtime-order succ match',sum(a==b for a,b in zip(vs,succ)))
