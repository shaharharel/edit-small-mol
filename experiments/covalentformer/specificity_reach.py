"""SPECIFICITY TEST: does reachability REJECT a cysteine the molecule cannot reach?

WHY THIS EXISTS. validate_reach.py measures RECALL -- given a real covalent complex, does the engine
recover the true placement (69.7% against a 71.6% structural ceiling). That is the right test for
using reachability as an EVALUATION metric on molecules known to be covalent.

It is the WRONG test for using it as a LABELLING ORACLE. A labeller that says "reachable" to
everything scores 100% recall and carries zero information. If we are going to mint 10^6 training
examples from this program, the number that governs whether the labels mean anything is precision:
when the engine says PASS, is it right?

There is no deposited set of "molecules that provably cannot react", so precision cannot be measured
directly. The substitute is a NEGATIVE CONTROL built by moving the target: hold the molecule and its
crystal scaffold pose FIXED, and ask the engine about a DIFFERENT cysteine in the same protein. The
molecule demonstrably reacts with its own cysteine; there is no evidence it reaches any other, and
for a cysteine 20 A away across the fold it certainly does not.

    true cysteine      -> must PASS at the recall rate (~70%)
    decoy, 8-15 A      -> ambiguous; a nearby cysteine may be genuinely reachable
    decoy, > 20 A      -> must FAIL at ~100%. Any pass here is a false positive, full stop.

THE NUMBER THAT MATTERS is the spread between the true rate and the far-decoy rate. If they are
close, the engine is measuring "this molecule has a flexible linker", not "this molecule reaches
THIS cysteine", and every label it mints is noise.

This is the same wrong-pocket contrast (the N4 rung) that the cofold campaign used, turned on our own
labeller instead of on Boltz. We required it of the structure predictor; we do not get to skip it
for ourselves.
"""
import os, sys, json, collections
import numpy as np, gemmi
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import reachable, electrophile_index

SCR = '/private/tmp/claude-501/-Users-shaharharel-Documents-github-quris-ml-play-CB1/160c04b4-9b18-4865-b718-cd0ea00a4622/scratchpad'
CIF = os.path.join(SCR, 'ladder/cif')
N_CONF = 120
# distance bands for decoy cysteines, measured SG_decoy to SG_true
BANDS = [('near 8-15 A', 8.0, 15.0), ('mid 15-25 A', 15.0, 25.0), ('far >25 A', 25.0, 1e9)]


def load(pid, cys, ch, lig):
    """-> (true SG, all CYS SG coords, ligand heavy atoms, protein heavy xyz, protein elements)."""
    st = gemmi.read_structure(os.path.join(CIF, '%s.cif.gz' % pid))
    st.setup_entities()
    sg = None
    all_sg = []
    ligatoms, prot, pelem = [], [], []
    for c in st[0]:
        for r in c:
            if r.name == 'CYS':
                a = next((a for a in r if a.name == 'SG'), None)
                if a is not None:
                    xyz = np.array([a.pos.x, a.pos.y, a.pos.z])
                    all_sg.append((c.name, int(r.seqid.num), xyz))
                    if str(r.seqid.num) == str(cys) and c.name == ch:
                        sg = xyz
            if r.name == lig:
                for a in r:
                    if a.element.name != 'H':
                        ligatoms.append((a.name, np.array([a.pos.x, a.pos.y, a.pos.z])))
            elif r.name not in ('HOH',):
                for a in r:
                    if a.element.name != 'H':
                        prot.append([a.pos.x, a.pos.y, a.pos.z])
                        pelem.append(a.element.name.upper())
    return sg, all_sg, ligatoms, (np.array(prot) if prot else None), pelem


def scaffold_anchor(mol, ligatoms, sg):
    """MCS correspondence between the pre-reactive SMILES and the crystal ADDUCT.

    Identical to validate_reach.py -- the crystal ligand is post-reaction so a direct template match
    fails, but the scaffold is common to both. Atoms within 6 A of SG are dropped so the frozen part
    excludes the reactive end that we are asking the engine to place.
    """
    blk = []
    for i, (nm, xyz) in enumerate(ligatoms):
        blk.append('HETATM%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s'
                   % (i + 1, nm[:4], 'LIG', 1, xyz[0], xyz[1], xyz[2],
                      ''.join(c for c in nm if c.isalpha())[:2]))
    cry = Chem.MolFromPDBBlock('\n'.join(blk) + '\nEND\n', removeHs=True, sanitize=False)
    if cry is None:
        return None, None, 'crystal_unparseable'
    try:
        Chem.SanitizeMol(cry, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE)
    except Exception:
        pass
    try:
        m = rdFMCS.FindMCS([mol, cry], timeout=10,
                           atomCompare=rdFMCS.AtomCompare.CompareElements,
                           bondCompare=rdFMCS.BondCompare.CompareAny,
                           ringMatchesRingOnly=True, completeRingsOnly=False)
    except Exception:
        return None, None, 'mcs_failed'
    if m.numAtoms < 8:
        return None, None, 'mcs_too_small'
    patt = Chem.MolFromSmarts(m.smartsString)
    mm, cm = mol.GetSubstructMatch(patt), cry.GetSubstructMatch(patt)
    if not mm or not cm or len(mm) != len(cm):
        return None, None, 'mcs_match_failed'
    conf = cry.GetConformer()
    ref = np.array([list(conf.GetAtomPosition(i)) for i in cm])
    keep = [k for k in range(len(mm)) if np.linalg.norm(ref[k] - sg) > 6.0]
    if len(keep) < 6:
        keep = list(range(len(mm)))
    scaf = tuple(mm[k] for k in keep)
    ref = ref[keep]
    if len(scaf) < 6:
        return None, None, 'scaffold_too_small'
    return scaf, ref, None


def main():
    P = {}
    for f in ('P_rung_final.json', 'P_rung_expanded.json'):
        p = os.path.join(SCR, 'ladder', f)
        if os.path.exists(p):
            for e in json.load(open(p))['entries']:
                P[e['pid']] = e
    print(__doc__.split('This is the same')[0])
    print('scanning %d P-rung entries, %d conformers each\n' % (len(P), N_CONF))

    rec = collections.defaultdict(list)   # band -> [hit, ...]
    skip = collections.Counter()
    n_entries = 0
    per_entry = []
    for pid, e in sorted(P.items()):
        pre, cys, ch, lig = e.get('pre'), e.get('cys'), e.get('cys_ch', 'A'), e.get('lig')
        if not all([pre, cys, lig]):
            skip['no_entry_fields'] += 1
            continue
        mol = Chem.MolFromSmiles(pre)
        if mol is None:
            skip['unparseable'] += 1
            continue
        el, cls = electrophile_index(mol)
        if el is None:
            skip['warhead_%s' % cls] += 1
            continue
        try:
            sg, all_sg, ligatoms, prot, pelem = load(pid, cys, ch, lig)
        except Exception:
            skip['cif_unreadable'] += 1
            continue
        if sg is None or len(ligatoms) < 8:
            skip['no_crystal_partners'] += 1
            continue
        scaf, ref, err = scaffold_anchor(mol, ligatoms, sg)
        if err:
            skip[err] += 1
            continue

        # TRUE TARGET
        r_true = reachable(mol, scaf, ref, sg, prot, protein_elem=pelem, n_conf=N_CONF)
        rec['TRUE cysteine'].append(bool(r_true.get('hit')))
        n_entries += 1
        row = {'pid': pid, 'true': bool(r_true.get('hit')), 'd_true': r_true.get('d')}

        # DECOY TARGETS -- one per band, nearest qualifying cysteine in that band
        ds = [(np.linalg.norm(x - sg), c, n, x) for c, n, x in all_sg
              if np.linalg.norm(x - sg) > 1e-3]
        for label, lo, hi in BANDS:
            cand = [t for t in ds if lo <= t[0] < hi]
            if not cand:
                continue
            _dist, _c, _n, dxyz = min(cand, key=lambda t: t[0])
            rd = reachable(mol, scaf, ref, dxyz, prot, protein_elem=pelem, n_conf=N_CONF)
            rec[label].append(bool(rd.get('hit')))
            row[label] = bool(rd.get('hit'))
        per_entry.append(row)

    if not n_entries:
        print('NOTHING SCORED. skipped %s' % dict(skip))
        return 1

    print('SPECIFICITY OF THE REACHABILITY ORACLE')
    print('  entries scored %d ; skipped %s\n' % (n_entries, dict(skip)))
    print('  %-16s %6s %10s' % ('target', 'n', 'PASS rate'))
    order = ['TRUE cysteine'] + [b[0] for b in BANDS]
    rates = {}
    for k in order:
        v = rec.get(k)
        if not v:
            continue
        rates[k] = float(np.mean(v))
        print('  %-16s %6d %9.1f%%' % (k, len(v), 100 * rates[k]))

    t = rates.get('TRUE cysteine')
    far = rates.get('far >25 A')
    print()
    if t is not None and far is not None:
        print('  THE NUMBER: true %.1f%% vs far-decoy %.1f%%  -> spread %.1f points'
              % (100 * t, 100 * far, 100 * (t - far)))
        # precision if the far band stands in for the negative class at a 1:1 prior
        if (t + far) > 0:
            print('  implied precision at a 1:1 pos:neg prior = %.1f%%' % (100 * t / (t + far)))
        print()
        if far > 0.25:
            print('  VERDICT: NOT SAFE AS A LABELLER. The engine passes cysteines >25 A away at')
            print('  %.0f%%. It is reporting linker flexibility, not target-specific reach.' % (100 * far))
        elif (t - far) < 0.35:
            print('  VERDICT: MARGINAL. Spread too small to mint labels on.')
        else:
            print('  VERDICT: discriminative. Far decoys are rejected; PASS carries information.')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'specificity_reach.json')
    json.dump({'rates': rates, 'n': n_entries, 'per_entry': per_entry,
               'n_conf': N_CONF, 'bands': [list(b) for b in BANDS]}, open(out, 'w'), indent=1)
    print('\n  wrote %s' % out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
