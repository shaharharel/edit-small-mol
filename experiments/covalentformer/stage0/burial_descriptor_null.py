"""IS BURIAL A POCKET PROPERTY, OR JUST THE LIGAND AGAIN? The gate that killed delta-geometry twice.

WHY THIS RUNS BEFORE ANY CLAIM. The bond length is dead as a steering target (sd 0.106 A on a
1.800 mean, CV 5.9% -- it IS the C-S bond). Burial genuinely varies: CYS mean 26.5, sd 8.4,
CV 31.7%, p05 11 -> p95 38. But VARYING IS NOT THE SAME AS POCKET-DEPENDENT. Delta-geometry also
varied, and died twice over when a 2D fingerprint with zero conformer information predicted it at
R2 0.877 (#143) and then beat the 3D feature outright (#176). If ECFP4 on the LIGAND ALONE predicts
burial, then burial is a ligand property wearing a pocket's clothes and the direction is dead.

PRE-REGISTERED READING, written before the numbers:
  ECFP4 R2 at or near the shuffled control  -> burial is genuinely POCKET-dependent. Build it.
  ECFP4 R2 comfortably above the control    -> burial is mostly the ligand. Do NOT build it.
  one integer (heavy-atom count) beating    -> the same "one integer wins" result this project has
  ECFP4                                        now hit five times; treat burial as a SIZE proxy.
POPULATION CUT NOBODY ASKED FOR, AND IT IS MINE. The ligand and the CYS are matched by
(resseq, resname) on ANY chain, accepted only if the match is UNIQUE -- there is no exact
(chain, resseq, resname) path at all. So every multimer carrying 2+ copies of the ligand or of the
residue is SILENTLY DROPPED. An auditor's strict matcher gets n=1,506 over 325 proteins where this
file gets 862 over 180: a ~43% cut that selects AGAINST multi-copy structures, and it was stated
nowhere. Any n or coverage figure from this file is a single-copy subpopulation.

AND THE PRE-REGISTERED RULE HAS NO CELL FOR "BOTH", WHICH IS THE ANSWER THE DATA GAVE. With the
intercept unpenalised, ECFP4 on the ligand alone reaches R2 +0.085 here and +0.18 on the auditor's
larger population (20/20 protein-disjoint draws positive) -- so the "comfortably above the control"
branch fires. But a POSITIVE CONTROL that this file never had shows pocket features reaching
R2 +0.69 (20/20 draws, shuffled control -0.003 +- 0.005), i.e. the pocket is ~3.8x the ligand.
The honest statement is "burial is PREDOMINANTLY pocket-dependent but NOT ligand-free". Neither
pre-registered branch says that, because I wrote a rule with two cells for a question with three
answers. Do not read either branch as having fired cleanly.
NOTE the auditor's own circularity caveat: their strongest pocket feature (P1, burial at the CYS SG)
sits ~1.8 A from the electrophile, so its 6 A sphere nearly overlaps the target -- it proves the
harness is alive, not that the feature is interesting. The non-circular pocket signal is residue-type
FRACTIONS at 8 A (+0.12) and global protein geometry (+0.07): modest, real, well above control.

SPLIT IS PROTEIN-DISJOINT, not random: the same protein appears in many complexes, so a random
split would let the model memorise a pocket and would answer a different question entirely.
LAMBDA IS SELECTED ON AN INNER SPLIT DRAWN THE SAME WAY AS THE OUTER ONE (#181/#182: a random
inner split against a disjoint outer split was worth 0.60 R2 of pure self-deception).
"""
import os, sys, csv, math, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_pocket_feasibility import parse_pdb, dist, NUCLEOPHILE_ATOM, PDB_DIR, RECORDS
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')

os.chdir(os.environ.get('CF_ROOT', '/Users/shaharharel/Documents/github/edit-small-mol'))
rows = list(csv.DictReader(open(RECORDS)))
have = set(f[:-4].upper() for f in os.listdir(PDB_DIR) if f.lower().endswith('.pdb'))
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X, Y, G, HA, smis = [], [], [], [], []
cache = {}
for r in rows:
    pdb = (r.get('PDB') or '').strip().upper()
    if pdb not in have or (r.get('Resi_name') or '').strip().upper() != 'CYS':
        continue
    if pdb not in cache:
        try: cache[pdb] = parse_pdb(os.path.join(PDB_DIR, pdb + '.pdb'))
        except Exception: cache[pdb] = None
    if cache[pdb] is None: continue
    het, res = cache[pdb]
    ln = (r.get('Ligand_name') or '').strip()
    try: lp = int(float(r.get('Ligand_position'))); rp = int(float(r.get('Resi_posi')))
    except (TypeError, ValueError): continue
    lc = [v for (c, s, nm), v in het.items() if s == lp and nm == ln]
    rc = [v for (c, s, nm), v in res.items() if s == rp and nm == 'CYS']
    if len(lc) != 1 or len(rc) != 1: continue
    nuc = next((x for n, x in rc[0] if n == 'SG'), None)
    if nuc is None: continue
    d = min(dist(nuc, x) for _n, x in lc[0])
    if not (1.2 <= d <= 2.4): continue
    m = Chem.MolFromSmiles((r.get('SMILES') or '').strip())
    if m is None: continue
    e = min(lc[0], key=lambda t: dist(nuc, t[1]))[1]
    cnt = sum(1 for (c, s, nm), atoms in res.items() if not (s == rp and nm == 'CYS')
              for _n, x in atoms if dist(e, x) < 6.0)
    fp = np.zeros(2048, dtype=np.float32)
    for b in gen.GetFingerprintAsNumPy(m).nonzero()[0]: fp[b] = 1.0
    X.append(fp); Y.append(float(cnt)); HA.append(float(m.GetNumHeavyAtoms()))
    G.append((r.get('Protein_name') or r.get('Proteins') or pdb))
X = np.array(X); Y = np.array(Y); HA = np.array(HA).reshape(-1, 1); G = np.array(G)
print('n=%d complexes | %d distinct proteins | burial mean %.1f sd %.1f'
      % (len(Y), len(set(G)), Y.mean(), Y.std()))

rng = np.random.RandomState(20260915)          # SEEDED: this samples the protein assignment
prot = sorted(set(G)); rng.shuffle(prot)
cut = int(0.75 * len(prot)); tr_p = set(prot[:cut])
m_tr = np.array([g in tr_p for g in G]); m_va = ~m_tr
inner = sorted(tr_p); rng.shuffle(inner)
ic = int(0.75 * len(inner)); i_tr = set(inner[:ic])
mi_tr = np.array([g in i_tr for g in G]) & m_tr; mi_va = m_tr & ~mi_tr
print('outer train %d / valid %d (protein-disjoint) | inner %d / %d'
      % (m_tr.sum(), m_va.sum(), mi_tr.sum(), mi_va.sum()))

def r2(yt, yp): 
    return 1.0 - ((yt - yp) ** 2).sum() / max(((yt - yt.mean()) ** 2).sum(), 1e-9)

def ridge_fit(A, b, lam):
    """Ridge with an UNPENALISED intercept, by centring on TRAIN and adding the mean back.

    THE SHIPPED VERSION PENALISED THE INTERCEPT AND INVERTED THIS FILE'S OWN VERDICT.
    It was `solve(A.T@A + lam*eye(n_cols), A.T@b)` with a ones-column hstacked in and nothing
    centred. `lam*eye` shrinks EVERY coefficient toward zero, the intercept column included, while
    y has mean 26.5 -- so the model had to carry a mean of ~26 through a column the regulariser was
    actively pulling to 0. It could not, and the 2048 fingerprint bits were recruited to carry the
    offset, which is exactly the overfitting the ridge exists to prevent. Fitted intercept at the
    selected lambda was 11.28 against a true valid mean of 26.33.
    THE NUMBERS IT PRODUCED, AND WHAT THEY BECOME once the intercept is unpenalised (same splits,
    same seed, same lambda grid, cross-checked against sklearn Ridge(fit_intercept=True) to 4dp):
        ECFP4 2048 (ligand only)   -0.2555  ->  +0.0849
        ONE INTEGER heavy-atom     -0.0003  ->  -0.0012
        ECFP4 + heavy-atom         -0.2462  ->  +0.1238
        SHUFFLED CONTROL           -0.2716  ->  +0.0002
        margin vs control          +0.0161  ->  +0.0847   (5.3x, and it CROSSES THE GATE)
    Both pre-registered readings therefore flip, and BOTH of this file's printed conclusions were
    wrong -- the size-proxy one in the opposite direction from the truth.
    """
    mu_A = A.mean(0)
    mu_b = b.mean()
    Ac = A - mu_A
    bc = b - mu_b
    w = np.linalg.solve(Ac.T @ Ac + lam * np.eye(Ac.shape[1]), Ac.T @ bc)
    return w, float(mu_b - mu_A @ w)


def ridge(A, b, lam):   # kept for call-site shape; returns (w, intercept)
    return ridge_fit(A, b, lam)

def run(F, name, yv=None):
    y = Y if yv is None else yv
    best, bl = -9e9, None
    for lam in (0.1, 1, 10, 100, 1000, 10000):      # chosen on the INNER split only
        w, b0 = ridge(F[mi_tr], y[mi_tr], lam)
        s = r2(y[mi_va], F[mi_va] @ w + b0)
        if s > best: best, bl = s, lam
    w, b0 = ridge(F[m_tr], y[m_tr], bl)
    out = r2(y[m_va], F[m_va] @ w + b0)
    print('  %-34s held-out R2 %+0.4f   (lambda %g chosen on inner split)' % (name, out, bl))
    return out

ones = np.ones((len(Y), 1), dtype=np.float32)
e_fp = run(X, 'ECFP4 2048 bits (LIGAND ONLY)')
e_ha = run(HA, 'ONE INTEGER: heavy-atom count')
e_bo = run(np.hstack([X, HA]), 'ECFP4 + heavy-atom count')
ysh = Y.copy(); rng.shuffle(ysh)
e_sh = run(X, 'SHUFFLED CONTROL (ECFP4)', yv=ysh)
print('\nVERDICT INPUTS (the reading was pre-registered in the docstring, not chosen now):')
print('  ECFP4 %+0.4f vs shuffled control %+0.4f  -> margin %+0.4f' % (e_fp, e_sh, e_fp - e_sh))
print('  one integer %+0.4f  -- beats ECFP4? %s' % (e_ha, 'YES' if e_ha > e_fp else 'no'))
json.dump(dict(n=len(Y), n_proteins=len(set(G)), ecfp4=e_fp, heavy_atoms=e_ha,
               ecfp4_plus_ha=e_bo, shuffled=e_sh), open('data/burial_descriptor_null.json', 'w'), indent=2)
print('wrote data/burial_descriptor_null.json')
