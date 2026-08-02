#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import matplotlib as mpl
mpl.rcParams['pdf.fonttype']=42; mpl.rcParams['ps.fonttype']=42
import matplotlib.pyplot as plt
import numpy as np

REWARDS=(2.2,2.6,5.0,9.0); PSEUDOCOUNTS=(0.0,0.1,0.5,1.0); D=12; TOL=1e-10

def sha256(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

def bayes_action(p,o):
 n,m=p.shape; v=np.full(m,1/o); b=np.zeros_like(p); r=np.zeros(n); phase=np.zeros(n,np.int8); ss=(p>TOL).sum(1)
 for i,eta in enumerate(p):
  s=eta>TOL; vs=v[s].sum()
  if vs<1-TOL: b[i]=eta; continue
  if abs(vs-1)<=TOL: b[i]=eta; phase[i]=1; continue
  ids=np.flatnonzero(s); ratios=eta[ids]/v[ids]; q=np.argsort(-ratios,kind='stable'); ids=ids[q]; ratios=ratios[q]
  pe=np.r_[0.,np.cumsum(eta[ids])]; pv=np.r_[0.,np.cumsum(v[ids])]; chosen=None
  for k in range(len(ids)+1):
   den=1-pv[k]
   if den<=TOL: continue
   rr=(1-pe[k])/den
   up=(k==0) or ratios[k-1]>rr-5*TOL; lo=(k==len(ids)) or rr>=ratios[k]-5*TOL
   if -TOL<=rr<=1+TOL and up and lo: chosen=float(np.clip(rr,0,1))
  if chosen is None: raise RuntimeError((i,o))
  r[i]=chosen; b[i]=np.maximum(eta-chosen*v,0); phase[i]=2
  if abs(b[i].sum()+r[i]-1)>2e-8: raise RuntimeError('budget')
 return b,r,phase,ss

class Fenwick:
 def __init__(self,n): self.a=np.zeros(n+1,np.int64)
 def add(self,i):
  i+=1
  while i<len(self.a): self.a[i]+=1; i+=i&-i
 def prefix(self,i):
  if i<0:return 0
  i+=1;s=0
  while i:s+=int(self.a[i]);i-=i&-i
  return s

def c2(x): return x*(x-1)//2

def pairwise(e,u):
 e=np.round(np.asarray(e,float),D);u=np.round(np.asarray(u,float),D);n=len(e);n0=n*(n-1)//2
 _,ec=np.unique(e,return_counts=True);_,uc=np.unique(u,return_counts=True);_,jc=np.unique(np.c_[e,u],axis=0,return_counts=True)
 te=int(c2(ec).sum());tu=int(c2(uc).sum());tb=int(c2(jc).sum());info=n0-te;ut=tu-tb
 ur=np.searchsorted(np.unique(u),u);order=np.lexsort((u,e));tree=Fenwick(int(ur.max())+1);prior=disc=pos=0
 while pos<n:
  end=pos+1;ev=e[order[pos]]
  while end<n and e[order[end]]==ev:end+=1
  for q in order[pos:end]:disc+=prior-tree.prefix(int(ur[q]))
  for q in order[pos:end]:tree.add(int(ur[q]));prior+=1
  pos=end
 both=n0-te-tu+tb;conc=both-disc;den_tau=math.sqrt(max(n0-te,0)*max(n0-tu,0));tau=(conc-disc)/den_tau if den_tau else float('nan');den=max(info,1)
 return dict(informative_pairs=int(info),strict_inversion_pairs=int(disc),unequal_risk_tie_pairs=int(ut),strict_inversion_pct=100*disc/den,unequal_risk_tie_pct=100*ut/den,tie_aware_discordance_pct=100*(disc+.5*ut)/den,kendall_tau_b=float(tau))

def groups(u):
 u=np.round(np.asarray(u,float),D);order=np.argsort(u,kind='stable');s=u[order];g=[];a=0
 for b in range(1,len(u)+1):
  if b==len(u) or s[b]!=s[a]:g.append((a,b));a=b
 return order,g

def curve(e,u):
 n=len(e);order,gs=groups(u);es=np.asarray(e)[order];G=np.zeros(n+1);acc=0;cum=0.;maxatom=0.
 for a,b in gs:
  block=es[a:b];z=b-a;bs=float(block.sum());op=np.cumsum(np.sort(block))
  for t in range(1,z+1):
   ep=t/z*bs;G[acc+t]=(cum+ep)/n;maxatom=max(maxatom,(ep-op[t-1])/n)
  acc+=z;cum+=bs
 c=np.arange(n+1)/n;go=np.r_[0.,np.cumsum(np.sort(e))/n];gap=G-go
 return dict(max_generalized_risk_gap=float(gap.max()),max_atom_local_gap=float(maxatom),augrc_gap=float(np.trapezoid(gap,c)),aurc_gap=float(np.mean(G[1:]/c[1:]-go[1:]/c[1:])))

def audit(C):
 rows=[];n,m=C.shape;votes=C.sum(1)
 for pc in PSEUDOCOUNTS:
  eta=(C+pc)/(votes[:,None]+m*pc);err=1-eta.max(1)
  for o in REWARDS:
   b,r,ph,ss=bayes_action(eta,o);z=b+r[:,None]/o
   for name,u in [('reserve',r),('effective_wealth',1-z.max(1))]:
    row=dict(n=n,pseudocount=pc,reward=o,score=name,mean_votes=float(votes.mean()),min_votes=int(votes.min()),max_votes=int(votes.max()),mean_support_size=float(ss.mean()),full_reserve_pct=100*float(np.mean(np.isclose(r,1,atol=1e-10))),no_reserve_pct=100*float(np.mean(np.isclose(r,0,atol=1e-10))),nonidentified_pct=100*float(np.mean(ph==1)))
    row.update(pairwise(err,u));row.update(curve(err,u));rows.append(row)
 return rows

def figure(rows,out):
 R=[r for r in rows if r['score']=='reserve']
 def mat(k):return np.array([[next(float(r[k]) for r in R if r['pseudocount']==pc and r['reward']==o) for o in REWARDS] for pc in PSEUDOCOUNTS])
 panels=[(mat('tie_aware_discordance_pct'),'Tie-aware discordance (%)'),(100*mat('max_generalized_risk_gap'),'Max generalized-risk gap (pp)'),(mat('full_reserve_pct'),'Full-reserve rows (%)'),(mat('no_reserve_pct'),'No-reserve rows (%)')]
 fig,ax=plt.subplots(2,2,figsize=(7.1,5.2),constrained_layout=True)
 for a,(M,title) in zip(ax.ravel(),panels):
  im=a.imshow(M,aspect='auto');a.set_xticks(range(4),[str(x) for x in REWARDS]);a.set_yticks(range(4),[str(x) for x in PSEUDOCOUNTS]);a.set_xlabel('reward $o$');a.set_ylabel('pseudocount');a.set_title(title,fontsize=9)
  mid=(np.nanmin(M)+np.nanmax(M))/2
  for i in range(4):
   for j in range(4):a.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if M[i,j]>mid else 'black')
  fig.colorbar(im,ax=a,fraction=.046,pad=.04)
 fig.savefig(out,bbox_inches='tight');fig.savefig(out.with_suffix('.png'),dpi=220,bbox_inches='tight');plt.close(fig)

def main_table(rows,out):
 cols=['pseudocount','reward','reserve_inversion_pct','reserve_tie_pct','reserve_tie_aware_pct','reserve_tau_b','max_gap_pp','augrc_gap_pp','aurc_gap_pp','full_reserve_pct','no_reserve_pct','ew_tie_aware_pct','ew_tau_b'];z=[','.join(cols)]
 for pc in (0.,.5):
  for o in REWARDS:
   r=next(x for x in rows if x['pseudocount']==pc and x['reward']==o and x['score']=='reserve');e=next(x for x in rows if x['pseudocount']==pc and x['reward']==o and x['score']=='effective_wealth')
   v=[pc,o,r['strict_inversion_pct'],r['unequal_risk_tie_pct'],r['tie_aware_discordance_pct'],r['kendall_tau_b'],100*r['max_generalized_risk_gap'],100*r['augrc_gap'],100*r['aurc_gap'],r['full_reserve_pct'],r['no_reserve_pct'],e['tie_aware_discordance_pct'],e['kendall_tau_b']];z.append(','.join(map(str,v)))
 out.write_text('\n'.join(z)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--counts',type=Path,required=True);ap.add_argument('--probs',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 C=np.load(a.counts).astype(float)
 if C.shape!=(10000,10):raise ValueError(C.shape)
 if np.any(C<0) or not np.allclose(C,np.rint(C)):raise ValueError('counts')
 diff=float('nan')
 if a.probs and a.probs.exists():P=np.load(a.probs);diff=float(np.max(np.abs(P-C/C.sum(1,keepdims=True))));assert diff<1e-12,diff
 rows=audit(C);p=a.out/'cifar10h_full_audit.csv'
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 main_table(rows,a.out/'cifar10h_main_table.csv');figure(rows,a.out/'cifar10h_full_audit.pdf')
 meta=dict(shape=list(C.shape),counts_sha256=sha256(a.counts),probs_sha256=sha256(a.probs) if a.probs and a.probs.exists() else None,counts_probs_max_abs_diff=diff,rewards=REWARDS,pseudocounts=PSEUDOCOUNTS,human_distribution='empirical human-label distribution; not a true conditional posterior',pairwise_rates='exact over all unordered pairs with unequal Bayes error',tie_randomization='independent uniform randomization inside exact score atoms',aurc='mean selective risk at k/n, k=1,...,n')
 (a.out/'cifar10h_full_audit_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 for p in sorted(a.out.iterdir()):
  if p.is_file():print(p.name,p.stat().st_size,sha256(p))
if __name__=='__main__':main()
