"""CAN WE BUILD THE THREE POCKET-DEPENDENT PARAMS AT ALL? Measured before anything is built.

THE USER'S ASK, verbatim: "we should have pocket dependant params - we had!!! e.g. distance to
cysteine etc. its important to also train the steering to take pocket into account". Three of the
six params need a pocket and none of them exists on disk:
    d_cys      distance from the ligand electrophile to the target cysteine SG
    attack     the SG -> electrophile approach angle
    burial     how enclosed the electrophile is by protein atoms

WHY THIS FILE EXISTS RATHER THAN A BUILDER. Every previous geometry direction on this project died
at its FEASIBILITY gate, and each time the gate was measured AFTER the build: delta-geometry was
built and then found to be 100% WARHEAD-only (#135/#136), then found to be a topological
restatement an ECFP4 fingerprint predicts at R2 0.877 (#143). The gating number here -- how many
CovInDB ligands have a structure OF THEIR OWN TARGET in which BOTH the ligand and the annotated
covalent residue are resolvable -- has never been measured. It is measured here first.

THE SUBSTRATE, which is better than I expected. Covalent_Complex_Records.csv carries, per row:
    PDB, Ligand_chain, Ligand_position, Ligand_name      <- where the ligand is
    Resi_chain, Resi_posi, Resi_name                     <- WHICH residue it is bonded to
so the covalent partner is annotated per complex, not inferred. 3,445 PDB files sit beside it.

THE FALSIFIER, STATED BEFORE THE NUMBERS. A real covalent C-S bond is ~1.8 A. If the measured
ligand-to-SG minimum distance does NOT concentrate near 1.8 A, then either the annotation or this
parser is wrong, and the whole direction is not buildable from this source -- regardless of how
many rows survive the coverage funnel. A large yield with a wrong distance distribution is the
worse outcome of the two, because it looks like success. Both are reported; neither is assumed.

WHAT THIS FILE ESTABLISHED, WITH THE NUMBERS THAT SURVIVED INDEPENDENT RECOMPUTATION.
An auditor reran all 3,598 records with their own parser. Every funnel cell agrees within 1-4
records: nucleophile_found 3,274 vs 3,273 (90.995% vs 90.97%), in-window 98.931% vs 98.839%,
621 distinct proteins both ways. The funnel is real.

THE PER-RESIDUE TABLE I QUOTED CAME FROM A DIFFERENT FUNNEL THAN THIS FILE'S.
"CYS 1.800, sd 0.106, n=862" is reproducible to the digit -- but its producer is
burial_descriptor_null.py's row filter, not this file. That filter has ONLY the unique-any-chain
fallback, where this file tries exact (chain, resseq, resname) FIRST, so it silently drops every
structure with 2+ copies of the ligand or the CYS: 653 records, 43% of the CYS set. This file's own
by_residue says CYS n=1523. Two funnels were being quoted in one sentence.
  THIS FILE'S POPULATION, in-window:  CYS n=1506  mean 1.8075  sd 0.1141  CV 6.31%
  the monomer-enriched burial subset: CYS n= 862  mean 1.7996  sd 0.1057  CV 5.87%
Quote the first if you mean this file. (SER agrees to 0.0016; THR/LYS/TYR differ by 0.019-0.030 and
are probably from a third funnel again.)

THE CONCLUSION HOLDS, AND IS NOW BOUNDED. Variance decomposition on in-window CYS: reaction type
27.7% and warhead class 27.5% of variance, but the top groups are all S-S and Se-S --
isoselenazolinone 2.2085, thiosulfonate 2.0735, disulfide 2.0220 -- against Michael acceptor 1.7785
and nitrile 1.7708. The ONLY subpopulation in which d genuinely moves is THE IDENTITY OF THE ATOM
BONDED TO SG, which is a categorical consequence of the warhead already chosen, not an independent
degree of freedom: you cannot ask a model for 1.95 A, only for a disulfide. Restricted to C-S
chemistry (n=1403, mean 1.7908 sd 0.0957), warhead class spans 0.080 A across 12 classes -- LESS
than one within-class sd -- and TARGET PROTEIN explains 4-6% at F=2.4-3.3.

AND DO NOT CALL THE RESIDUAL "CRYSTALLOGRAPHIC NOISE". I did, and it is wrong. Binned by resolution:
  <1.5 A sd 0.1081 | 1.5-1.8 0.1005 | 1.8-2.1 0.1205 | 2.1-2.5 0.1194 | 2.5-3.0 0.1099 | >3.0 0.1138
The spread is FLAT -- it does not shrink at high resolution, so it is not dominated by coordinate
error. (The MEAN drifts -0.033 A from best to worst resolution; the SPREAD does not.) The supported
statement is STRONGER than "it is noise": the variance in d is not explained by pocket, protein,
resolution, warhead class OR reaction type once the bonded element is held fixed. One alternative
this file cannot rule out: part of that sd may be min-over-ligand-atoms selecting a non-bonded atom,
which would also be resolution-independent.
"""
import os, sys, csv, json, math, argparse
from collections import Counter, defaultdict

PDB_DIR = 'data/covbinder/raw_covindb2/PDB'
RECORDS = 'data/covbinder/raw_covindb2/Covalent_Complex_Records.csv'

# Residues whose covalent atom we know how to name. CovInDB annotates 200 distinct Site strings,
# most CYS, but SER/THR/TYR/LYS covalent chemistry is real and is counted separately rather than
# folded into a "cysteine" number it does not belong to.
NUCLEOPHILE_ATOM = {'CYS': 'SG', 'SER': 'OG', 'THR': 'OG1', 'TYR': 'OH', 'LYS': 'NZ'}


def parse_pdb(path):
    """(het, res) -- HETATM by (chain, resseq, resname), ATOM by (chain, resseq, resname).

    Hand-rolled on purpose: column-indexed PDB parsing has no failure mode that returns a
    plausible wrong coordinate, whereas a library that silently skips altlocs or picks the last
    occupancy would. Altloc is handled explicitly -- blank or 'A' only.
    """
    # FIRST MODEL ONLY. An NMR ensemble is one FILE containing 20-32 MODELs, and the original
    # loop stacked every model into the same coordinate dict -- so a residue's atoms appeared 20x
    # over and any COUNT over them was multiplied by the model count. Found by hunting an outlier
    # rather than by reading the code: CYS burial had sd 58.8 against a mean of 31.8 with p05/p95
    # of 11/39, which is not a wide distribution but a handful of enormous ones. The top eight were
    # 2LXY (1032 atoms, 32 models), 3ZGP (706, 20), 2KID (667, 20), 2MLM (589, 20), 2RUI (533, 20),
    # 6R1V (493, 20) -- every one SOLUTION NMR, and 6 of 8 single-chain, so it was never the
    # multimer explanation I first reached for. 18 of 3,445 files are multi-model (0.52%).
    # WHY THE BOND-LENGTH RESULT SURVIVES THIS: distances are a MIN over ligand atoms, and the min
    # over a stacked ensemble is still a real distance in some model. Counts are not -- they are
    # a SUM, so they scale with the model count. Any COUNT taken before this fix is inflated on
    # those 18 files; any DISTANCE is not.
    het, res = defaultdict(list), defaultdict(list)
    with open(path, errors='ignore') as fh:
        for line in fh:
            rec = line[:6]
            if rec == 'ENDMDL':
                break
            if rec not in ('ATOM  ', 'HETATM'):
                continue
            altloc = line[16]
            if altloc not in (' ', 'A'):
                continue
            name = line[12:16].strip()
            resn = line[17:20].strip()
            chain = line[21].strip()
            try:
                seq = int(line[22:26])
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                continue
            elem = line[76:78].strip().upper()
            if elem == 'H' or (not elem and name.startswith('H')):
                continue
            (het if rec == 'HETATM' else res)[(chain, seq, resn)].append((name, xyz))
    return het, res


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/Users/shaharharel/Documents/github/edit-small-mol')
    ap.add_argument('--limit', type=int, default=0, help='0 = all records')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    os.chdir(a.root)

    rows = list(csv.DictReader(open(RECORDS)))
    if a.limit:
        rows = rows[:a.limit]
    have = set(f[:-4].upper() for f in os.listdir(PDB_DIR) if f.lower().endswith('.pdb'))

    # THE FUNNEL. Every stage is counted, including the ones that drop rows, because a yield
    # quoted without its funnel is how "80% are discarded" went unnoticed until the user asked.
    n = dict(records=len(rows), pdb_on_disk=0, pdb_parsed=0, ligand_found=0,
             residue_found=0, nucleophile_atom_found=0, distance_ok=0)
    dists, by_resn, targets, pdbs_ok, unparsed = [], Counter(), set(), set(), Counter()
    cache = {}

    for r in rows:
        pdb = (r.get('PDB') or '').strip().upper()
        if pdb not in have:
            continue
        n['pdb_on_disk'] += 1
        if pdb not in cache:
            try:
                cache[pdb] = parse_pdb(os.path.join(PDB_DIR, pdb + '.pdb'))
            except Exception as e:
                unparsed[type(e).__name__] += 1
                cache[pdb] = None
        if cache[pdb] is None:
            continue
        het, res = cache[pdb]
        n['pdb_parsed'] += 1

        lc = (r.get('Ligand_chain') or '').strip()
        ln = (r.get('Ligand_name') or '').strip()
        try:
            lp = int(float(r.get('Ligand_position')))
        except (TypeError, ValueError):
            lp = None
        lig = het.get((lc, lp, ln))
        if lig is None and lp is not None:
            # chain label mismatch is common between the annotation and the file; accept a unique
            # (resseq, resname) match on ANY chain, but only if it is unique -- an ambiguous match
            # would silently pick an arbitrary copy in a multimer
            cand = [v for (c, s, nm), v in het.items() if s == lp and nm == ln]
            lig = cand[0] if len(cand) == 1 else None
        if not lig:
            continue
        n['ligand_found'] += 1

        rc = (r.get('Resi_chain') or '').strip()
        rn = (r.get('Resi_name') or '').strip().upper()
        try:
            rp = int(float(r.get('Resi_posi')))
        except (TypeError, ValueError):
            rp = None
        rr = res.get((rc, rp, rn))
        if rr is None and rp is not None:
            cand = [v for (c, s, nm), v in res.items() if s == rp and nm == rn]
            rr = cand[0] if len(cand) == 1 else None
        if not rr:
            continue
        n['residue_found'] += 1

        want = NUCLEOPHILE_ATOM.get(rn)
        nuc = next((xyz for nm, xyz in rr if nm == want), None) if want else None
        if nuc is None:
            continue
        n['nucleophile_atom_found'] += 1

        d = min(dist(nuc, xyz) for _nm, xyz in lig)
        dists.append(d)
        by_resn[rn] += 1
        targets.add(r.get('Protein_name') or r.get('Proteins') or pdb)
        pdbs_ok.add(pdb)
        if 1.2 <= d <= 2.4:
            n['distance_ok'] += 1

    print('=== POCKET-PARAM FEASIBILITY, CovInDB covalent complexes ===')
    print('PDB files on disk: %d' % len(have))
    order = ['records', 'pdb_on_disk', 'pdb_parsed', 'ligand_found', 'residue_found',
             'nucleophile_atom_found', 'distance_ok']
    base = n['records']
    for k in order:
        print('  %-24s %6d  %6.2f%% of records' % (k, n[k], 100.0 * n[k] / max(base, 1)))
    if unparsed:
        print('  parse failures: %s' % dict(unparsed))

    if not dists:
        print('\nNO MEASURABLE COMPLEX. The direction is NOT buildable from this source.')
        return 1

    dists.sort()
    def q(p):
        return dists[min(len(dists) - 1, int(p * len(dists)))]
    print('\n--- THE FALSIFIER: ligand-to-nucleophile minimum distance (A) ---')
    print('  n=%d   min %.2f   p05 %.2f   p25 %.2f   MEDIAN %.2f   p75 %.2f   p95 %.2f   max %.2f'
          % (len(dists), dists[0], q(.05), q(.25), q(.50), q(.75), q(.95), dists[-1]))
    near = sum(1 for d in dists if 1.2 <= d <= 2.4)
    print('  within 1.2-2.4 A (a real covalent bond is ~1.8): %d / %d = %.2f%%'
          % (near, len(dists), 100.0 * near / len(dists)))
    print('  VERDICT: %s' % ('the annotation and this parser AGREE -- the bond is where it should be'
                             if near / len(dists) > 0.8 else
                             'DISTRIBUTION IS WRONG -- do not build on this until it is explained'))
    print('\n--- coverage of what survives ---')
    print('  distinct PDB entries %d | distinct proteins %d' % (len(pdbs_ok), len(targets)))
    print('  by nucleophile residue: %s' % dict(by_resn.most_common()))

    if a.out:
        json.dump(dict(funnel=n, n_pdb_on_disk=len(have), n_measured=len(dists),
                       dist_quantiles=dict(min=dists[0], p05=q(.05), p25=q(.25), median=q(.50),
                                           p75=q(.75), p95=q(.95), max=dists[-1]),
                       frac_in_covalent_window=near / len(dists),
                       distinct_pdbs=len(pdbs_ok), distinct_proteins=len(targets),
                       by_residue=dict(by_resn)), open(a.out, 'w'), indent=2)
        print('\nwrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
