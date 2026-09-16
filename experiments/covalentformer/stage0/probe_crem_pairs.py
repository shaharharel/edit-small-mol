"""STAGE 0 PROTOTYPE: can CReM produce MW-matched, TC-banded, phi-DIFFERING covalent pairs?

Everything upstream of this is an estimate. The 160M pair count came from CReM's own mw column with
no molecules assembled; the TC band came from model-generated cohorts, not CReM. This assembles REAL
molecules and measures the three quantities the whole Stage 0 design rests on:
    MW match   -- enforced at GENERATION via min_inc/max_inc (atom-count delta), not filtered after
    TC         -- between PARENT and VARIANT, the actual training pair
    dphi       -- from the VALIDATED producer (scripts/compute_planar_2d_e2.py), imported not retyped

The worry this is built to falsify: CReM variants share everything but one swapped core, so they may
be TOO similar -- clustered at TC>0.8 where the earlier sample showed dphi supply collapsing.
If median |dphi| here is near zero, phi-steering pairs cannot be built this way and the plan changes.
Reported as a DISTRIBUTION with both tails, not a mean.
"""
import sys, os, time, json
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from crem.crem import mutate_mol
RDLogger.DisableLog('rdApp.*')

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from compute_planar_2d_e2 import _compute_2d_one

DB = os.path.join(ROOT, 'data/crem_db/chembl33_sa2_f5.db')
ACR = Chem.MolFromSmarts('[CH2]=[CH]C(=O)N')

def phi(smi):
    o = _compute_2d_one((0, smi))
    v = o.get('planar_dev_deg')
    return None if v is None else float(v)

def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)

# Parents: real covalent molecules that already carry an acrylamide AND a scored phi.
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT,'paper/reproducibility/metrics/planar_2d/*.csv'))])
df = df[(df.acryl_match==True)&(df.embed_ok==True)&df.planar_dev_deg.notna()].drop_duplicates('smi')
parents = df.sample(int(sys.argv[1]) if len(sys.argv)>1 else 40, random_state=17)

rows, t0 = [], time.time()
n_par = 0
for smi, p_phi_stored in zip(parents.smi, parents.planar_dev_deg):
    m = Chem.MolFromSmiles(smi)
    if m is None or not m.HasSubstructMatch(ACR): continue
    try:
        # min_inc/max_inc bound the ATOM-COUNT change -> MW matching enforced at generation.
        variants = list(mutate_mol(m, db_name=DB, radius=3, min_size=1, max_size=8,
                                   min_inc=-1, max_inc=1, max_replacements=12, ncores=1))
    except Exception as e:
        print('  mutate failed: %s' % e); continue
    if not variants: continue
    n_par += 1
    p_phi = phi(smi)
    if p_phi is None: continue
    pm_fp, p_mw = fp(m), Descriptors.MolWt(m)
    for v in variants:
        vm = Chem.MolFromSmiles(v)
        if vm is None or not vm.HasSubstructMatch(ACR):  # variant must KEEP the warhead
            continue
        v_phi = phi(v)
        if v_phi is None: continue
        rows.append(dict(parent=smi, variant=v,
                         tc=float(DataStructs.TanimotoSimilarity(pm_fp, fp(vm))),
                         dmw=float(abs(Descriptors.MolWt(vm)-p_mw)),
                         dphi=float(abs(v_phi-p_phi)), p_phi=p_phi, v_phi=v_phi))
    if n_par % 10 == 0:
        print('  %d parents -> %d pairs  (%.1fs)' % (n_par, len(rows), time.time()-t0), flush=True)

print('\nPARENTS USED %d   PAIRS %d   elapsed %.1fs' % (n_par, len(rows), time.time()-t0))
if not rows: raise SystemExit('NO PAIRS -- the assembly path does not work as designed.')
tc  = np.array([r['tc'] for r in rows]); dmw = np.array([r['dmw'] for r in rows])
dph = np.array([r['dphi'] for r in rows])
print('  TC     : med %.3f  p10 %.3f  p90 %.3f' % (np.median(tc), *np.percentile(tc,[10,90])))
print('  |dMW|  : med %.2f  p90 %.2f  (generation-enforced, not post-filtered)' % (np.median(dmw), np.percentile(dmw,90)))
print('  |dphi| : med %.2f  mean %.2f  p90 %.2f' % (np.median(dph), dph.mean(), np.percentile(dph,90)))
print('  frac |dphi|>10 %.1f%%   >20 %.1f%%' % (100*(dph>10).mean(), 100*(dph>20).mean()))
print('\n  TC band      n   med|dphi|  frac>20deg')
for lo,hi in [(0,.4),(.4,.6),(.6,.8),(.8,1.01)]:
    mk=(tc>=lo)&(tc<hi)
    if mk.sum()<5: print('  %.1f-%.1f %5d   (too few)'%(lo,hi,mk.sum())); continue
    print('  %.1f-%.1f %5d   %8.2f   %8.1f%%'%(lo,hi,mk.sum(),np.median(dph[mk]),100*(dph[mk]>20).mean()))
out = os.path.join(ROOT,'experiments/covalentformer/results/stage0_crem_probe.json')
json.dump(dict(n_parents=n_par, n_pairs=len(rows), rows=rows[:2000]), open(out,'w'), indent=1)
print('\nwritten -> %s' % out)
