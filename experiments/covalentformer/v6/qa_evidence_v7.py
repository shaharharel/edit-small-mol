#!/usr/bin/env python3
"""Hard QA of v7 evidence. Every check here corresponds to a defect that actually shipped in v6,
so a pass means those specific failures are gone -- not that the data is perfect.

Run: python qa_evidence_v7.py [v7_evidence.jsonl]
"""
import collections, json, re, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from evidence_v4 import (IDEAL_BOND, COV_TOL, COV_SOFT, VDW, VDW_FRAC, CONTACT_CUTOFF, POLAR,
                         METALS)

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
MODEL_FIELDS = {'atom', 'element', 'faces', 'chain', 'distance_A', 'interaction', 'angle_deg',
                'polar_partner', 'polar_distance_A', 'polar_interaction', 'polar_angle_deg',
                'waters_within_3.5A', 'nearest_water_A', 'lining', 'open_fraction', 'reading'}
RES = re.compile(r'^([A-Z]{1,3})(\d+)\.')
ev = [json.loads(l) for l in open(D + '/' + (sys.argv[1] if len(sys.argv) > 1 else 'v7_evidence.jsonl'))]
n = len(ev)
bad, warn, info = collections.Counter(), collections.Counter(), collections.Counter()
sitefrac, dists, waters = [], [], []


def is_bonded(t):
    u = t.upper()
    return ('COVALENT' in u or 'METAL COORDINATION' in u or 'cofactor' in t)


for e in ev:
    mi, lo = e['MODEL_INPUT'], e['LABEL_ONLY']
    ats = mi['atoms']
    if not ats:
        bad['no atoms']+= 1
        continue

    # ---- 1. reproducibility (v6 site_of_change depended on wall clock)
    if e['QA'].get('NOT_REPRODUCIBLE'):
        bad['NOT reproducible across two builds']+= 1

    # ---- 2. schema and the inference firewall
    for a in ats:
        if set(a.keys()) != MODEL_FIELDS:
            bad['MODEL_INPUT row has the wrong field set']+= 1
            break
    blob = json.dumps([{k: v for k, v in a.items()} for a in ats]).lower()
    if 'analogue' in blob or 'lead' in blob:
        bad['the word lead/analogue appears in MODEL_INPUT']+= 1

    # ---- 3. site of change: v6 marked EVERY atom in 501 rows (MCS discarded on `canceled`)
    names = {a['atom'] for a in ats}
    soc = [s for s in lo['site_of_change'] if s['atom'] in names]
    f = len(soc) / len(ats)
    sitefrac.append(f)
    if f == 1.0:
        bad['every atom marked CHANGED (MCS failure)']+= 1
    if not lo['site_of_change']:
        bad['no site_of_change']+= 1

    # ---- 4. contact typing
    for a in ats:
        t, d, el = a['interaction'], a['distance_A'], a['element']
        dists.append(d)
        waters.append(a['waters_within_3.5A'])
        if not RES.match(a['faces']):
            bad['malformed residue reference']+= 1
            break
        m = re.match(r'^COVALENT BOND to ([A-Z]{1,3})\d+\.(\S+)', t)
        if m:
            info['covalent bonds']+= 1
            pel = ''.join(c for c in m.group(2) if c.isalpha())[:1]
            ideal = IDEAL_BOND.get('%s-%s' % (el, pel))
            if ideal and d > ideal + COV_TOL:
                bad['COVALENT BOND beyond ideal+%.2f A' % COV_TOL]+= 1
        if t.startswith('probable covalent'):
            info['probable covalent (disclosed)']+= 1
        if 'METAL COORDINATION' in t:
            info['metal coordination']+= 1
            if d > 2.8:
                bad['METAL COORDINATION beyond 2.8 A']+= 1
        if t.startswith('H-bond'):
            info['H-bonds (nearest atom)']+= 1
            if not a['angle_deg']:
                bad['H-bond without an angle']+= 1
            elif a['angle_deg'] < 90:
                bad['H-bond with an impossible angle']+= 1
            if d > 3.6:
                bad['H-bond beyond 3.6 A']+= 1
        if a['polar_interaction'] and a['polar_interaction'].startswith('H-bond'):
            info['H-bonds found ONLY via polar_partner']+= 1
        # the vdW disclosure (v6 compared against bond length and disclosed 314 of 3,521)
        _res = ''.join(c for c in a['faces'].split('.')[0] if c.isalpha()).upper()
        _an = ''.join(c for c in a['faces'].split('.')[-1] if c.isalpha()).upper()
        _std = _res in {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS',
                        'MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}
        _pel = _an[:1] if _std else (_an if _an in VDW else _an[:1])
        s, q = VDW.get(el.upper()), VDW.get(_pel)
        polar_pair = el in POLAR and q and _pel in POLAR
        if s and q and not polar_pair and d < VDW_FRAC * (s + q) and not is_bonded(t) \
                and 'uncertain' not in t and 'unusually close' not in t:
            bad['sub-vdW contact NOT disclosed']+= 1
        if not (0.0 <= a['open_fraction'] <= 1.0):
            bad['open_fraction out of range']+= 1
        exp = {'buried': a['open_fraction'] < 0.2,
               'partly enclosed': 0.2 <= a['open_fraction'] < 0.45,
               'solvent-facing': a['open_fraction'] >= 0.45}
        if not exp.get(a['reading'], False):
            bad['reading inconsistent with open_fraction']+= 1
        if a['polar_partner'] and a['polar_distance_A'] > CONTACT_CUTOFF:
            bad['polar_partner beyond the contact cutoff']+= 1

    # ---- 5. the lead side: symmetry and the atom-name bridge
    lc = lo.get('analogue_contacts') or []
    if not lc:
        bad['no analogue_contacts']+= 1
    else:
        l = Chem.MolFromSmiles(e['ANSWER']['analogue_smiles'])
        if l is not None and len(lc) < l.GetNumHeavyAtoms():
            warn['fewer analogue_contacts rows than analogue heavy atoms']+= 1
        for c in lc:
            if c['faces'] is None and c['distance_A'] <= CONTACT_CUTOFF:
                bad['lead no-contact row within the cutoff']+= 1
            if c['faces'] is not None and c['distance_A'] > CONTACT_CUTOFF:
                bad['lead contact beyond the cutoff']+= 1
            if c['input_atom'] is not None and c['input_atom'] not in names:
                bad['lead input_atom names an atom not in the hit table']+= 1
            if c['status'] not in ('conserved', 'NEW'):
                bad['lead contact without a status']+= 1
        info['lead rows conserved']+= sum(1 for c in lc if c['status'] == 'conserved')
        info['lead rows NEW']+= sum(1 for c in lc if c['status'] == 'NEW')
        info['lead rows with no contact within %.1f A' % CONTACT_CUTOFF]+= \
            sum(1 for c in lc if c['faces'] is None)

    # ---- 6. the molecules still agree with the record
    h = Chem.MolFromSmiles(mi['input_smiles'])
    l = Chem.MolFromSmiles(e['ANSWER']['analogue_smiles'])
    if h is None or l is None:
        bad['unparseable SMILES']+= 1
        continue
    t = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(h, 2, 2048),
                                       AllChem.GetMorganFingerprintAsBitVect(l, 2, 2048))
    if t < 0.50:
        bad['Tanimoto below the 0.50 floor']+= 1
    if abs(t - e['tanimoto']) > 0.02:
        bad['stated Tanimoto disagrees with the molecules']+= 1

sitefrac.sort(); dists.sort()
print('v7 evidence rows        %6s   | %d targets' % (format(n, ','), len({e['target'] for e in ev})))
print('site-of-change fraction  p50 %.2f  p90 %.2f  p99 %.2f  max %.2f'
      % (sitefrac[n // 2], sitefrac[9 * n // 10], sitefrac[99 * n // 100], sitefrac[-1]))
print('contact distance         p5 %.2f  p50 %.2f  p95 %.2f A'
      % (dists[len(dists) // 20], dists[len(dists) // 2], dists[19 * len(dists) // 20]))
print('atoms with >=1 ordered water within 3.5 A: %.0f%%'
      % (100.0 * sum(1 for w in waters if w) / max(len(waters), 1)))
print('\nCOUNTS')
for k, v in info.most_common():
    print('  %-52s %s' % (k, format(v, ',')))
print('\nBLOCKERS' if bad else '\nBLOCKERS: none')
for k, v in bad.most_common():
    print('  %-52s %6s (%.1f%%)' % (k, format(v, ','), 100.0 * v / n))
print('\nWARNINGS' if warn else 'WARNINGS: none')
for k, v in warn.most_common():
    print('  %-52s %6s (%.1f%%)' % (k, format(v, ','), 100.0 * v / n))
sys.exit(1 if bad else 0)
