import sys, pickle; sys.path.insert(0,"dificil")
import numpy as np
from exp import VAL
from train_forecast import COLS, epoch_hours
ev=epoch_hours(VAL.index).values
truth={k:dict(zip(ev.tolist(),VAL[COLS[k]].values)) for k in ("price","co2","load","pv")}
pos={e:i for i,e in enumerate(ev)}
first=np.concatenate([[True],np.diff(ev)!=1]); bstart=np.maximum.accumulate(np.where(first,np.arange(len(ev)),0))
def maes(rec):
    out={}
    for j,k in enumerate(("price","co2","load","pv")):
        E=[[] for _ in range(7)]; Eh=[[] for _ in range(24)]
        for t,arrs in rec.items():
            if t not in pos: continue
            day=(pos[t]-bstart[pos[t]])//24
            for h in range(len(arrs[j])):
                v=truth[k].get(t+h)
                if v is not None and (k in("load","pv") or h>=3):
                    E[day].append(abs(arrs[j][h]-v)); Eh[h].append(abs(arrs[j][h]-v))
        out[k]=(np.mean(np.concatenate(E)),[np.mean(e) for e in E],[np.mean(e) if e else 0 for e in Eh])
    return out
if __name__=="__main__":
    for f in sys.argv[1:]:
        R=maes(pickle.load(open(f,"rb")))
        for k,(m,d,h) in R.items():
            print(f"{f.split('/')[-1]:16s} {k:5s} MAE {m:.4g} | dia:", " ".join(f"{x:.4g}" for x in d)," | h0,3,6,12,23:", " ".join(f"{h[i]:.4g}" for i in (0,3,6,12,23)))
