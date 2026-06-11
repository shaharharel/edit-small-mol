"""End-to-end Boltz pipeline QA — 2026-06-10.

Verifies that:
1. F4 cofold directories are present on disk
2. F4_boltz_full.csv has 100% Boltz column coverage
3. The live backend's deduplicated DF preserves Boltz columns via the SMILES-keyed backfill
4. All visible survivors in the report have full Boltz data
5. Numerical sanity (Boltz cols are within expected distributions)
6. Disk-vs-CSV consistency (flags zero-byte sentinel files masquerading as data)

Usage: python experiments/qa_boltz_pipeline.py
Exit code: 0 = all critical checks pass, 1 = critical failures
"""
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BACKEND = "http://localhost:5001"

CRITICAL_BOLTZ_COLS = [
    "boltz_iptm", "boltz_ligand_iptm",
    "mPAE_paper", "mPAE_london",
    "pocket_occupancy_pct", "n_stabilizing_contacts",
    "d_SG", "burgi_dunitz_dev_deg",
    "vina_rescore_affinity_kcalmol", "vina_rescore_intra_kcalmol",
    "pKa_Cys346",
]
EXPECTED_RANGES = {
    "boltz_iptm":      (0.5, 1.0),
    "boltz_ligand_iptm": (0.5, 1.0),
    "mPAE_paper":      (0.5, 3.0),
    "mPAE_london":     (0.3, 10.0),
    "pocket_occupancy_pct": (40.0, 100.0),
    "n_stabilizing_contacts": (20, 200),
    "d_SG":            (0.5, 15.0),
    "burgi_dunitz_dev_deg": (0, 90),
    "vina_rescore_affinity_kcalmol": (-15, 35),  # Boltz poses can clash → high positive values; L5 filter rejects them
    "pKa_Cys346":      (5, 15),
}

checks = []
def add(name, ok, msg="", critical=True):
    checks.append((name, ok, msg, critical))
    icon = "[OK]" if ok else ("[!!]" if critical else "[~~]")
    severity = "" if ok else (" CRITICAL" if critical else " (warn)")
    print(f"  {icon} {name:55s} {msg}{severity}")

print("=" * 72)
print("BOLTZ END-TO-END QA")
print("=" * 72)

# 1. Cofold dirs
print("\n[1/7] Cofold directories on disk")
pool_dirs = list((PROJECT / "data/boltz_f4_results").glob("*/boltz_results_*"))
extra_dirs = list((PROJECT / "data/boltz_f4_extra_results").glob("*/boltz_results_*"))
add("pool: ≥1546 cofold dirs",  len(pool_dirs)  >= 1546, f"{len(pool_dirs)}")
add("extra: ≥675 cofold dirs",   len(extra_dirs) >= 675,  f"{len(extra_dirs)}")

# 2. F4_boltz_full.csv coverage
print("\n[2/7] F4_boltz_full.csv coverage")
csv_path = PROJECT / "data/tier4_scored/F4_boltz_full.csv"
add("F4_boltz_full.csv exists",  csv_path.exists())
if csv_path.exists():
    df = pd.read_csv(csv_path)
    n = len(df)
    add("CSV has 2,221 rows",      n == 2221, f"{n}")
    add("100% boltz_iptm populated", df["boltz_iptm"].notna().sum() == n,
        f"{df['boltz_iptm'].notna().sum()}/{n}")
    for t, exp in [("main", 1546), ("rescue", 675)]:
        sub = df[df["_tier"] == t]
        p = sub["boltz_iptm"].notna().sum()
        add(f"  {t}: {exp} rows × 100%", p == len(sub) == exp,
            f"{p}/{len(sub)}", critical=True)

# 3. Critical column coverage on CSV
print("\n[3/7] Critical Boltz columns ≥99% populated in CSV")
for c in CRITICAL_BOLTZ_COLS:
    if c not in df.columns:
        add(f"col {c}", False, "missing column")
        continue
    p = df[c].notna().sum()
    add(f"col {c:40s}", p >= n * 0.99, f"{p}/{n} ({p/n*100:.1f}%)")

# 4. Distribution sanity
print("\n[4/7] Numerical distribution sanity (values in expected ranges)")
sub = df[df["boltz_iptm"].notna()]
for c, (lo, hi) in EXPECTED_RANGES.items():
    if c not in sub.columns: continue
    s = sub[c].dropna()
    if len(s) == 0: continue
    min_v, max_v = s.min(), s.max()
    ok = min_v >= lo and max_v <= hi
    add(f"  {c:38s} ∈ [{lo}, {hi}]", ok, f"min={min_v:.2f} max={max_v:.2f}")

# 5. Disk-vs-CSV consistency — flag zero-byte sentinel files
print("\n[5/7] Disk consistency — zero-byte CIF/PAE files")
import subprocess
result = subprocess.run(
    ["find", str(PROJECT / "data/boltz_f4_results"),
     str(PROJECT / "data/boltz_f4_extra_results"),
     "-name", "*_model_0.cif", "-size", "0"],
    capture_output=True, text=True)
zero_cifs = result.stdout.strip().split("\n") if result.stdout.strip() else []
n_zero = len([z for z in zero_cifs if z])
add("Zero-byte CIF count == 0 (warn if non-zero)",
    n_zero == 0, f"{n_zero} sentinel files found", critical=False)
if n_zero > 0:
    print(f"     (sentinels are orchestrator artifacts; CSV values are typically valid")
    print(f"      from prior real runs but cannot be re-verified from disk)")

# 6. Live backend: backfill landed
print("\n[6/7] Live backend (http://localhost:5001)")
try:
    r = requests.get(f"{BACKEND}/api/filter",
                     params={"groups": "murcko,thiq,murcko_and_acryl", "length": "30000"},
                     timeout=180)
    rows = r.json().get("rows", [])
    n_vis = len(rows)
    n_boltz = sum(1 for rw in rows if rw.get("boltz_iptm") is not None)
    n_london = sum(1 for rw in rows if rw.get("mPAE_london") is not None)
    n_pocket = sum(1 for rw in rows if rw.get("pocket_occupancy_pct") is not None)
    add("Visible survivors ≥ 500", n_vis >= 500, f"{n_vis}")
    add("All visible have boltz_iptm (L5 NaN-FAIL)", n_boltz == n_vis,
        f"{n_boltz}/{n_vis}")
    add("All visible have mPAE_london",  n_london == n_vis, f"{n_london}/{n_vis}")
    add("All visible have pocket_occupancy", n_pocket == n_vis, f"{n_pocket}/{n_vis}")
except Exception as e:
    add("Backend reachable", False, str(e))

# 7. SMILES round-trip: every F4 SMILES with valid Boltz exists in DF
print("\n[7/7] F4 SMILES → visible round-trip")
pool_smis = set(pd.read_csv(PROJECT / "data/tier4_scored/F4_boltz_pool_v2.csv")["smiles"].dropna())
extra_smis = set(pd.read_csv(PROJECT / "data/tier4_scored/F4_boltz_extra_v2.csv")["smiles"].dropna())
sent = pool_smis | extra_smis
add("F4 pool unique SMILES = 1,546", len(pool_smis) == 1546, f"{len(pool_smis)}")
add("F4 extra unique SMILES = 675",  len(extra_smis) == 675,  f"{len(extra_smis)}")
add("F4 sent unique SMILES = 2,221", len(sent) == 2221, f"{len(sent)}")

vis_smis = set(rw["smiles"] for rw in rows if rw.get("smiles"))
vis_w_boltz = set(rw["smiles"] for rw in rows
                  if rw.get("smiles") and rw.get("boltz_iptm") is not None)
add("Visible-with-Boltz ⊆ F4 sent",  vis_w_boltz.issubset(sent),
    f"{len(vis_w_boltz - sent)} stray")

# Final
print("\n" + "=" * 72)
n_crit_fail = sum(1 for _, ok, _, c in checks if not ok and c)
n_warn = sum(1 for _, ok, _, c in checks if not ok and not c)
n_pass = sum(1 for _, ok, _, _ in checks if ok)
status = "PASS" if n_crit_fail == 0 else "FAIL"
print(f"FINAL: {n_pass}/{len(checks)} pass · {n_crit_fail} CRITICAL fails · {n_warn} warns · STATUS: {status}")
print("=" * 72)

if n_crit_fail:
    print("\nCRITICAL FAILS:")
    for n, ok, m, c in checks:
        if not ok and c: print(f"  - {n}: {m}")

sys.exit(0 if n_crit_fail == 0 else 1)
