import cv2,numpy as np
from frames import load,E
def full(p):
    c=cv2.VideoCapture(p);ok,f=c.read();c.release();assert ok,p
    return f
def feats(p):
    f=full(p);H=f.shape[0]
    img=cv2.resize(f[:256],(128,96),interpolation=cv2.INTER_AREA).astype(np.float32)
    txt=cv2.resize(f[256:],(160,18),interpolation=cv2.INTER_AREA).astype(np.float32)
    return img,txt
