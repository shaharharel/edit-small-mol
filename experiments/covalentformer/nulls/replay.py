"""Replay Phase B rows/split/gstats exactly as train_phaseB.main() does, READ-ONLY."""
import os,sys,csv,random,collections
import numpy as np, torch, torch.nn as nn
CF="/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer"
ROOT="/Users/shaharharel/Documents/github/edit-small-mol"
sys.path.insert(0,CF); sys.path.insert(0,'/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer
from train_phaseA import PhaseA, subsequent_mask
from train_phaseB import build_rows, PhaseBData, collate, de_leak_holdout
DATA=os.path.join(CF,'data/phaseB/phaseB.csv')
PRIOR=os.path.join(ROOT,'models/reinvent4_mol2mol_warhead_tokens.prior')

def replay(seed, deleak_version, data=DATA):
    # deleak_version IS REQUIRED AND HAS NO DEFAULT. It used to default to 1, which was correct for
    # every checkpoint that existed when it was written and silently wrong for every one written
    # after 14:02. Every caller today passes it explicitly, so nothing was ever mis-scored -- but a
    # future `replay(seed)` would have gotten v1 with no signal, and this would have been the sixth
    # estimator straddle of the night. A default that is right only until the next artifact lands
    # is a trap, not a convenience; make the caller state which filter it means.
    rng=random.Random(seed)
    rows,fail=build_rows(data,rng)
    by_t=collections.defaultdict(list)
    for r in rows: by_t[r['target_id']].append(r)
    targets=sorted(by_t); rng.shuffle(targets)
    n_val=max(2,int(0.2*len(targets)))
    val_t,tr_t=set(targets[:n_val]),set(targets[n_val:])
    tr=[r for r in rows if r['target_id'] in tr_t]
    va_pre=[r for r in rows if r['target_id'] in val_t]
    # APPLY THE FILTER THE CHECKPOINT WAS BUILT WITH, not whichever is newest. train_phaseB stamps
    # 'deleak_version'; absent means v1 (target + exact keep), which is correct for every checkpoint
    # written before 14:02 today. v2 adds isotope-stripped keep and parent. Hardcoding either one
    # mis-scores half the checkpoints on disk, and per_channel_permutation's assert turns that into a
    # crash for one half and a silent wrong number for the other.
    # IMPORTED, NOT RETYPED. This was the third of four independent copies; they agreed by luck.
    va=de_leak_holdout(tr,va_pre,version=deleak_version)
    g=torch.tensor([[float(r['d']),float(r['cos_theta']),float(r['v_free'])] for r in tr],dtype=torch.float)
    mu,sd=g.mean(0),g.std(0); sd=torch.where(sd<1e-6,torch.ones_like(sd),sd)
    return dict(rows=rows,fail=fail,tr=tr,va_pre=va_pre,va=va,tr_t=tr_t,val_t=val_t,
                gstats=(mu,sd),targets=targets,by_t=by_t)

_PR=None
def prior():
    global _PR
    if _PR is None:
        ck=torch.load(PRIOR,map_location='cpu',weights_only=False)
        _v=ck['vocabulary']
        vocab=_v if isinstance(_v,Vocabulary) else Vocabulary(_v['tokens'] if isinstance(_v,dict) and 'tokens' in _v else _v)
        _PR=(ck,vocab,SMILESTokenizer())
    return _PR

def load_model(ckpath,dev='cpu'):
    ck,vocab,tok=prior()
    npm=dict(ck['network_parameter'])
    m=PhaseA(mode="geom",**npm)
    # RECONSTRUCT THE ENCODER THE CHECKPOINT WAS TRAINED WITH, from its own `cond_enc` stamp.
    # This line used to hardcode the pre-GeomEncoder Sequential. That is correct for every
    # checkpoint written before the encoder work and WRONG for every one written after: the
    # state_dict keys become geom_mlp.net.* / geom_mlp.emb.*+edges / W+net.*. strict=True means
    # it would have CRASHED rather than mis-scored -- right polarity -- but it would have
    # crashed only after a multi-hour train, which is the expensive place to find out.
    c_peek=torch.load(ckpath,map_location='cpu',weights_only=False)
    _enc=c_peek.get('cond_enc','mlp')
    if _enc=='mlp' and not any(k.startswith('geom_mlp.net.') for k in c_peek['model_state']):
        m.geom_mlp=nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,npm['model_dimension']))
    else:
        from train_phaseB import GeomEncoder
        d=npm['model_dimension']
        if _enc=='bin':
            ed=c_peek['model_state'].get('geom_mlp.edges')
            if ed is None: raise SystemExit('FATAL %s stamped cond_enc=bin but carries no edges buffer'%ckpath)
            m.geom_mlp=GeomEncoder('bin',d,n_bins=c_peek.get('n_bins',16),edges=ed.numpy())
        elif _enc=='rff':
            m.geom_mlp=GeomEncoder('rff',d,n_rff=c_peek.get('n_rff',32))
        else:
            m.geom_mlp=GeomEncoder('mlp',d)
    c=torch.load(ckpath,map_location='cpu',weights_only=False)
    m.load_state_dict(c['model_state'],strict=True)
    m.to(dev).eval()
    return m,c,vocab,tok

def eval_loss(model,rws,vocab,tok,gstats,dev='cpu',bs=16):
    ce=nn.CrossEntropyLoss(ignore_index=0,reduction='none')
    dl=torch.utils.data.DataLoader(PhaseBData(rws,vocab,tok,gstats=gstats),batch_size=bs,shuffle=False,collate_fn=collate)
    num=den=0.0
    with torch.no_grad():
        for b in dl:
            if b is None: continue
            src,sm,tgt,g=[x.to(dev) for x in b[:4]]
            ti,to=tgt[:,:-1],tgt[:,1:]
            lg=model.generator(model.forward_cond(src,sm,ti,subsequent_mask(ti.size(1),dev),g))
            per=ce(lg.reshape(-1,lg.size(-1)),to.reshape(-1)).view(to.shape)
            msk=(to!=0).float()
            num+=float((per*msk).sum()); den+=float(msk.sum())
    return num/max(den,1.0)
