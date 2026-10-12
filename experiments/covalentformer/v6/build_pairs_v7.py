#!/usr/bin/env python3
"""v7 pair corpus. Rebuilt from meta.json + ligand_smiles.json to one spec:

    same paper/series  ·  Tanimoto >= 0.50  ·  unique at the PAIR level

Three bugs in the v6 builder are fixed here, each of which inflated or corrupted the corpus:

1. ACCESSION LOOP.  v6 did `for u in v['unp']`, so an entry carrying several UniProt accessions
   emitted the same chemical pair once per accession -- 33,551 duplicate lines out of 64,403, all
   differing only in `target`, and for fusion constructs the accession chosen was often the wrong
   chain (P0ABE7 = the BRIL crystallisation tag appeared as a "target"). The series key here is the
   PAPER; the target is the accession shared by both members' entries.

2. GLOBAL LEAD LOOKUP.  v6 left the lead structure to the evidence builder, which took
   `sorted(LIG2PDB[analogue_code])[0]` -- the alphabetically first entry in the entire 43k-file mirror
   containing that ligand code, with no requirement that it be the same protein. 12% of rows ended
   up pairing two unrelated proteins while a same-series candidate existed. Both structures are
   chosen here, from the same series, and the pair is dropped if they share no accession.

3. UNIQUENESS.  Deduplication is on the canonical SMILES PAIR, not on either molecule, so a hit
   that was optimised three different ways keeps all three edits; only the identical edit recurring
   across crystals collapses. The surviving representative keeps pointers to the others.

Lipids, detergents and cofactors are excluded: `BCR` (beta-carotene) in photosystem and `PCW`
(phosphatidylcholine) in membrane proteins are correct depositions that are not medicinal chemistry,
and they were the bulk of the pairs whose two members turned out to be different proteins.
"""
import collections, hashlib, itertools, json, os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
TC_FLOOR = 0.50
HEAVY_MIN, HEAVY_MAX = 10, 70
ROTB_MAX = 15            # a generic lipid/detergent cut: acyl chains and PEGs run well past this
PER_TARGET_CAP = 500

COFACTOR = set('NAD NAP NDP NAI NDE FAD FMN FDA ATP ADP AMP GTP GDP GMP ANP ACP AGS GNP GSP SAM '
               'SAH COA ACO MLC HEM HEC HEA DHE UDP UTP CTP TTP UMP CMP UGA PLP TPP TDP BTN FES '
               'SF4 MGD GSH GDN PQQ F43 COM B12 CLA BCL PHO'.split())
# Lipids, detergents, cryo/buffer components and pigments. Not junk depositions -- just not
# hit-to-lead chemistry, and the source of most cross-protein "pairs" in v6.
NOT_MEDCHEM = set('PCW PC1 PCF PEE PEF PGT PGV PSF POV DMU LMT LMU BNG BOG SOG OGA LDA C8E PEG '
                  'PGE P6G 1PE 2PE PG4 PE4 EDO GOL MPD TRS MES EPE DTT BME SDS ACT FMT IPA OCT '
                  'HEX D10 D12 UND MYR PLM STE OLA LNK ARA PIO PIP BCR LUT CRT XAT NEX ZEX SQD '
                  'DGD LHG LMG CDL CDN TGL TRD HTG HTO F6P SUC TRE GLC BGC MAN NAG NDG BMA FUC '
                  'XYS GAL GLA SIA CHD CHS Y01 OLC OLB PX4 7PH 6PL 9PE'.split())
EXCLUDE = COFACTOR | NOT_MEDCHEM


NULLS = {'', '?', '.', 'None', 'null'}


def cite_of(v):
    """The paper, or None. mmCIF writes an absent value as `?`, and `v['doi'] or v['pm']` returns
    that string -- truthy. v6 therefore merged every uncited entry into a single "paper" holding
    4,045 ligands, which is 8.2M of the 8.3M candidate combinations and not a series in any sense:
    those pairs share no paper, and they are a third independent source of the cross-protein pairs.
    """
    for k in ('doi', 'pm'):
        x = v.get(k)
        if x is not None and str(x).strip() not in NULLS:
            return str(x).strip()
    return None


def main():
    M = json.load(open(D + '/meta.json'))
    SMI = json.load(open(D + '/ligand_smiles.json'))

    # ---- series = one PAPER. No accession loop: that is bug 1.
    ent = {}
    for k, v in M.items():
        unp = [u for u in (v.get('unp') or ()) if str(u).strip() not in NULLS]
        if unp and cite_of(v):
            ent[k] = dict(v, unp=unp)
    series = collections.defaultdict(lambda: collections.defaultdict(set))   # paper -> lig -> pdbs
    for pid, v in ent.items():
        paper = cite_of(v)
        for l in v['lig']:
            series[paper][l].add(pid)
    # paper -> lig -> accession -> [pdbs]. The accession is used only to require that the hit's and
    # the lead's structures are the same protein; it is NOT part of the series key, which is what
    # duplicated v6 once per accession.
    acc_of = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    for paper, ligs in series.items():
        for l, pdbs in ligs.items():
            for pid in sorted(pdbs):
                for u in (ent[pid].get('unp') or ()):
                    acc_of[paper][l][u].append(pid)

    print('entries with a target and a paper   %s' % format(len(ent), ','))
    print('distinct papers (series)            %s' % format(len(series), ','))

    mol, fp, hv, rb = {}, {}, {}, {}

    def get(l):
        if l in mol:
            return mol[l]
        s = SMI.get(l)
        m = Chem.MolFromSmiles(s) if s else None
        mol[l] = m
        if m is not None:
            fp[l] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
            hv[l] = m.GetNumHeavyAtoms()
            rb[l] = Descriptors.NumRotatableBonds(m)
        return m

    att = collections.Counter()
    cand = []
    for paper, ligs in sorted(series.items()):
        usable = [l for l in sorted(ligs) if l not in EXCLUDE]
        att['ligands excluded as cofactor/lipid/detergent'] += len(ligs) - len(usable)
        if len(usable) < 2:
            continue
        for a, b in itertools.combinations(usable, 2):
            att['0. same paper, both ligands in scope'] += 1
            if get(a) is None or get(b) is None:
                continue
            att['1. both parse from the CCD'] += 1
            if not (HEAVY_MIN <= hv[a] <= HEAVY_MAX and HEAVY_MIN <= hv[b] <= HEAVY_MAX):
                continue
            att['2. both %d-%d heavy atoms' % (HEAVY_MIN, HEAVY_MAX)] += 1
            if rb[a] > ROTB_MAX or rb[b] > ROTB_MAX:
                continue
            att['3. both <=%d rotatable bonds (drops acyl chains/PEGs)' % ROTB_MAX] += 1
            if Chem.MolToSmiles(mol[a], isomericSmiles=False) == \
               Chem.MolToSmiles(mol[b], isomericSmiles=False):
                continue
            att['4. not identical ignoring stereochemistry'] += 1
            t = DataStructs.TanimotoSimilarity(fp[a], fp[b])
            if t < TC_FLOOR:
                continue
            att['5. Tanimoto >= %.2f' % TC_FLOOR] += 1

            # DIRECTION IS A CONVENTION, NOT A CLAIM.
            #
            # There is no hit->lead criterion anywhere in this pipeline: these are same-series
            # analogues, and the task is "given this molecule and its pocket, suggest an analogue".
            # Assigning the smaller molecule as the input therefore asserted something we never
            # measured -- and worse, it biased the target. Measured on the previous build: the
            # answer was SMALLER in 0.0% of 7,032 pairs and bigger in 81.5%, with every property
            # shifted (heavy +2.4, clogP +0.26, TPSA +6.9, rotatable bonds +0.62). A model trained
            # on that learns "always grow, never trim", which is an artifact of the ordering and
            # not chemistry.
            #
            # The flip is a hash of the two ligand codes, so it is deterministic and reproducible
            # without depending on iteration order, and it is independent of size.
            first, second = sorted((a, b))
            flip = int(hashlib.md5(('%s|%s' % (first, second)).encode()).hexdigest(), 16) & 1
            lo, hi = (second, first) if flip else (first, second)

            # ---- bug 2: BOTH structures come from this series, and must be the same protein.
            # Indexed by accession rather than scanned as a double loop: a fragment-screening paper
            # contributes hundreds of entries per ligand and the O(n^2) form does not finish.
            best = None
            ha, la = acc_of[paper][lo], acc_of[paper][hi]
            for tgt in sorted(set(ha) & set(la)):
                for ph in ha[tgt]:
                    pa = next((x for x in la[tgt] if x != ph), None)
                    if pa:                  # both ligands in one crystal is not a hit->lead pair
                        best = (ph, pa, tgt)
                        break
                if best:
                    break
            if not best:
                continue
            ph, pa, tgt = best
            att['6. hit and lead structures in the series, same protein'] += 1
            cand.append({'target': tgt, 'paper': paper, 'pdb': ph, 'analogue_pdb': pa,
                         'input_code': lo, 'analogue_code': hi,
                         'input': Chem.MolToSmiles(mol[lo]), 'analogue': Chem.MolToSmiles(mol[hi]),
                         'tanimoto': round(t, 3),
                         'heavy': [hv[lo], hv[hi]],
                         'clogp': [round(Descriptors.MolLogP(mol[lo]), 2),
                                   round(Descriptors.MolLogP(mol[hi]), 2)],
                         'tpsa': [round(Descriptors.TPSA(mol[lo]), 1),
                                  round(Descriptors.TPSA(mol[hi]), 1)],
                         'rotb': [rb[lo], rb[hi]],
                         'hbd': [Descriptors.NumHDonors(mol[lo]), Descriptors.NumHDonors(mol[hi])],
                         'hba': [Descriptors.NumHAcceptors(mol[lo]),
                                 Descriptors.NumHAcceptors(mol[hi])]})

    # ---- bug 3: unique at the PAIR level, on canonical SMILES, not per molecule.
    by = collections.defaultdict(list)
    for r in cand:
        by[(r['input'], r['analogue'])].append(r)
    uniq = []
    for k, rows in by.items():
        rows.sort(key=lambda r: (r['pdb'], r['analogue_pdb']))
        rep = dict(rows[0])
        rep['n_crystal_realisations'] = len(rows)
        rep['other_crystals'] = ['%s->%s' % (r['pdb'], r['analogue_pdb']) for r in rows[1:]][:20]
        uniq.append(rep)
    att['7. unique chemical pairs (hit SMILES, lead SMILES)'] = len(uniq)

    # per-target cap, applied to the LARGEST series first so a cap never silently keeps only the
    # alphabetical head of an over-represented target the way v6's did
    uniq.sort(key=lambda r: (r['target'], -r['n_crystal_realisations'], r['pdb']))
    cap, out = collections.Counter(), []
    for r in uniq:
        if cap[r['target']] >= PER_TARGET_CAP:
            continue
        cap[r['target']] += 1
        out.append(r)
    att['8. cap %d pairs per target' % PER_TARGET_CAP] = len(out)

    with open(D + '/v7_pairs.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')

    print('\nattrition (each line is the survivor count after that filter):')
    for k in sorted(att):
        if k[0].isdigit():
            print('  %-54s %s' % (k, format(att[k], ',')))
    print('  %-54s %s' % ('(ligands skipped as cofactor/lipid/detergent)',
                          format(att['ligands excluded as cofactor/lipid/detergent'], ',')))
    print('\nv7_pairs.jsonl  %s pairs | %d targets | %d papers'
          % (format(len(out), ','), len({r['target'] for r in out}), len({r['paper'] for r in out})))
    rz = collections.Counter(r['n_crystal_realisations'] for r in out)
    print('crystal realisations per unique pair: %s'
          % ', '.join('%dx:%d' % (k, rz[k]) for k in sorted(rz)[:6]))


if __name__ == '__main__':
    main()
