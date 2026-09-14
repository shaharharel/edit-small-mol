"""DECISIVE MASK: locate the fragment by ATOM IDENTITY, not by string alignment.

BOTH string-diff masks are wrong, in opposite directions, and the numbers show it:
  difflib(anchor, target) -> WARHEAD share 0.223. UNDER-covers: frag_from and frag_to share a role
      and thus many tokens, so difflib aligns them "equal" and I measured the difference BETWEEN
      FRAGMENTS rather than the fragment.
  difflib(keep,   target) -> DECORATION share 0.487 where the fragment is ~8 of ~66 tokens (0.12).
      OVER-covers: `keep` is stored in its own canonical form with a dummy atom, and canonical
      SMILES REORDERS ATOMS, so its tokens do not appear as a contiguous aligned run inside the
      target. difflib scores the leftovers as fragment.
SMILES string alignment cannot locate a substructure, because canonicalisation reorders atoms.

This maps atoms instead. RDKit's _smilesAtomOutputOrder gives the atom index emitted at each atom
token position, so a substructure match of `keep` inside `target` converts directly into the set of
token positions that belong to the fragment. No alignment heuristic anywhere.
"""
import sys, csv, collections, torch, numpy as np
sys.path.insert(0,'experiments/covalentformer'); sys.path.insert(0,'/Users/shaharharel/Documents/github/REINVENT4')
import torch.nn as nn
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from train_phaseA import PhaseA, subsequent_mask, ROLES, R2I
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
ckpt, src = sys.argv[1], sys.argv[2]
torch.manual_seed(101)
ck=torch.load(ckpt,map_location='cpu',weights_only=False)
m=PhaseA(mode=ck.get('mode','role'),**dict(ck['network_parameter'])); m.load_state_dict(ck['model_state']); m.eval()
_v=ck['vocabulary']; vocab=_v if isinstance(_v,Vocabulary) else Vocabulary(_v['tokens']); tok=SMILESTokenizer()
ce=nn.CrossEntropyLoss(ignore_index=0,reduction='none')
NONATOM=set('()[]=#-+/\\@.%0123456789')
def atom_token(t):
    if t in ('^','$') or t=='': return False
    if t[0] in NONATOM and not t.startswith('['): return False
    return True
def frag_positions(target, keep):
    """-> set of TOKEN indices in tokenize(target) that belong to the fragment, or None."""
    T=Chem.MolFromSmiles(target); K=Chem.MolFromSmiles(keep)
    if T is None or K is None: return None
    K=Chem.DeleteSubstructs(K, Chem.MolFromSmarts('[#0]'))       # drop the attachment dummy
    if K.GetNumAtoms()<1: return None
    try: Chem.SanitizeMol(K)
    except Exception: return None
    mt=T.GetSubstructMatch(K)
    if not mt: return None
    keepatoms=set(mt)
    smi=Chem.MolToSmiles(T)
    order=list(map(int,T.GetProp('_smilesAtomOutputOrder')[1:-1].split(','))) if T.HasProp('_smilesAtomOutputOrder') else None
    if order is None: return None
    toks=tok.tokenize(smi); pos=set(); ai=0
    for i,t in enumerate(toks):
        if atom_token(t):
            if ai<len(order) and order[ai] not in keepatoms: pos.add(i)
            ai+=1
    return (smi,toks,pos) if pos else None
rows=[r for r in csv.DictReader(open(src)) if r.get('keep')]
print('ATOM-MAPPED FRAGMENT GAP [DETERMINISTIC 3-WRONG CONTROL]  ckpt=%s ep=%s n=%d src=%s'%(ckpt.split('/')[-2],ck.get('epoch'),len(rows),src.split('/')[-1]),flush=True)
acc=collections.defaultdict(lambda:[0.,0.,0,0.]); g=torch.Generator().manual_seed(7)
skipped=0; B=24; buf=[]
for r in rows:
    fp=frag_positions(r['target'], r['keep'])
    if fp is None: skipped+=1; continue
    buf.append((r,fp))
print('  rows with a clean atom mapping: %d (skipped %d)'%(len(buf),skipped),flush=True)
for i in range(0,len(buf),B):
    ch=buf[i:i+B]
    tt=[c[1][1] for c in ch]; at=[tok.tokenize(c[0]['anchor']) for c in ch]
    ids=[np.asarray(vocab.encode(t)).astype(np.int64) for t in tt]
    aid=[np.asarray(vocab.encode(t)).astype(np.int64) for t in at]
    K=max(len(x) for x in aid); T=max(len(x) for x in ids)
    src_t=torch.zeros(len(ch),K,dtype=torch.long); sm=torch.zeros(len(ch),1,K,dtype=torch.bool)
    tgt=torch.zeros(len(ch),T,dtype=torch.long); fmask=torch.zeros(len(ch),T-1)
    for j,(a_,t_) in enumerate(zip(aid,ids)):
        src_t[j,:len(a_)]=torch.from_numpy(a_); sm[j,0,:len(a_)]=True; tgt[j,:len(t_)]=torch.from_numpy(t_)
        for p in ch[j][1][2]:
            if 1<=p<T: fmask[j,p-1]=1.0
    cond=torch.tensor([R2I[c[0]['role']] for c in ch],dtype=torch.long)
    ti,to=tgt[:,:-1],tgt[:,1:]; msk=subsequent_mask(ti.size(1),'cpu'); pad=(to!=0).float(); fm=fmask*pad
    def L(c):
        with torch.no_grad(): lg=m.generator(m.forward_cond(src_t,sm,ti,msk,c))
        per=ce(lg.reshape(-1,lg.size(-1)),to.reshape(-1)).view(to.shape)
        return (per*fm).sum(1)/fm.sum(1).clamp(min=1)
    # DETERMINISTIC ALL-WRONG CONTROL. The original drew ONE random wrong role per row
    # (off=torch.randint(1,4,...)). That is unbiased in expectation -- the draw is uniform over the
    # three wrong roles -- so it does not bias the pooled GAP, but it injects variance into every
    # per-role cell, and #95 measured per-role run-to-run swings up to 0.052. This version averages
    # the model's loss over ALL THREE wrong roles per row, which is the same estimand with the
    # sampling variance removed. Any difference from the random-draw version is variance, not bias;
    # a systematic shift would mean one of those two statements is wrong.
    lt=L(cond)
    lc=sum(L((cond+k)%4) for k in (1,2,3))/3.0
    for j,c in enumerate(ch):
        if float(fm[j].sum())<1: continue
        a=acc[c[0]['role']]; a[0]+=float(lt[j]); a[1]+=float(lc[j]); a[2]+=1
        a[3]+=float(fm[j].sum())/max(float(pad[j].sum()),1)
    if i%960==0: print('  %d/%d'%(i,len(buf)),flush=True)
print('  %-16s %7s %9s %11s %11s %10s'%('role','n','fragshare','true','ctrl','GAP'))
tot=[0.,0.,0,0.]
for r in ROLES:
    a=acc.get(r)
    if not a or a[2]==0: continue
    print('  %-16s %7d %9.3f %11.4f %11.4f %+10.4f'%(r,a[2],a[3]/a[2],a[0]/a[2],a[1]/a[2],(a[1]-a[0])/a[2]))
    for k in range(4): tot[k]+=a[k]
print('  %-16s %7d %9.3f %11.4f %11.4f %+10.4f'%('ALL',tot[2],tot[3]/tot[2],tot[0]/tot[2],tot[1]/tot[2],(tot[1]-tot[0])/tot[2]))
