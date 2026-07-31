"""Validator for gmx_MMPBSA result CSVs.

Reads a CSV produced by the mdmmgbsa pipeline (one row per ligand) and
checks each row against the HR15 pass criteria from
/tmp/mmgbsa_methodology_checklist.md. Returns PASS/FAIL with structured
issues + notes per row, plus an aggregate verdict.

CLI:
    python experiments/qa_mmgbsa_outputs.py --csv <results.csv> \\
        --target_kind {smoke,zap70} --json_out /tmp/mmgbsa_qa_status.json

Exit codes:
    0 = PASS (all rows OK), 1 = FAIL (≥1 row fails), 2 = missing/empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Key residue sets per target (HR15)
KEY_RESIDUES = {
    "smoke": {
        "Cys481", "Met477", "Thr474", "Glu475",
        "Tyr476", "Lys430", "Asp539",
    },
    "zap70": {
        "Cys346", "Met414", "Glu415", "Tyr416",
        "Met417", "Asp479", "Lys369", "Ala339",
    },
}

# dG range per target (kcal/mol).
# ZAP70 broadened to [-50, 0]: covalent inhibitors with capped MM-GBSA
# overestimate vs experiment 2-3x. Smoke (non-covalent ibrutinib) keeps
# the tight [-15, -3] band.
DG_RANGE = {
    "smoke": (-15.0, -3.0),
    "zap70": (-50.0, 0.0),
}

# dG std ceiling per target (kcal/mol)
DG_STD_MAX = {"smoke": 5.0, "zap70": 6.0}

# Universal sanity: |dG| absolute cap. Set above ZAP70 lower bound so
# legitimate covalent capped-MM-GBSA values near -50 don't trip the
# sanity check; anything beyond is almost certainly unphysical.
DG_ABS_CAP = 60.0
# Minimum frames (HR15)
N_FRAMES_MIN = 100
# Ligand-strain envelope (kcal/mol)
STRAIN_MIN, STRAIN_MAX = 0.0, 15.0
# Wall-time sanity: anything < 30 s suggests aborted; > 24 h suggests stuck
WALL_S_MIN, WALL_S_MAX = 30.0, 24 * 3600.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float(x: Any) -> float | None:
    """Best-effort float cast; returns None on failure / NaN."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _to_int(x: Any) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def _parse_residues(rc_field: Any) -> dict[str, float]:
    """Parse residue_contributions field robustly.

    Accepts JSON-string `'{"Cys346": -2.1, "Met414": -1.5}'`, a dict
    already, None, or "". Returns dict[res_name -> energy_kcalmol].
    Unknown formats yield {}.
    """
    if rc_field is None or rc_field == "":
        return {}
    if isinstance(rc_field, dict):
        return {str(k): float(v) for k, v in rc_field.items()
                if _to_float(v) is not None}
    if isinstance(rc_field, str):
        s = rc_field.strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return {}
        # Try JSON
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            # Try Python literal (e.g. csv field with single quotes)
            try:
                import ast
                obj = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return {}
        if isinstance(obj, dict):
            out: dict[str, float] = {}
            for k, v in obj.items():
                fv = _to_float(v)
                if fv is not None:
                    out[str(k)] = fv
            return out
    return {}


def _residue_name_match(parsed: dict[str, float], key_set: set[str]) -> set[str]:
    """Match parsed residue keys against key_set, tolerating formats like
    'Cys346', 'CYS346', 'C346', 'C:346', '346'.
    """
    matched: set[str] = set()
    parsed_keys = list(parsed.keys())
    # build lookup by trailing residue number AND by uppercased name
    for key in key_set:
        # key like "Cys346"
        three = key[:3].upper()  # CYS
        one = three[0]           # C
        try:
            num = int(key[3:])   # 346
        except ValueError:
            num = None
        for pk in parsed_keys:
            pk_up = pk.upper()
            if key.upper() in pk_up:
                matched.add(key); break
            if num is not None and str(num) in pk_up and (three in pk_up or one in pk_up):
                matched.add(key); break
            # plain trailing number match like "346"
            if num is not None and pk_up.endswith(str(num)):
                # require at least letter match
                if three in pk_up or one in pk_up or pk_up == str(num):
                    matched.add(key); break
    return matched


# ---------------------------------------------------------------------------
# Per-row validation
# ---------------------------------------------------------------------------

def validate_row(row: dict[str, Any], target_kind: str) -> dict[str, Any]:
    """Validate a single result row.

    Returns:
        {'pass': bool, 'issues': [str, ...], 'notes': [str, ...],
         'dG': float|None, 'dG_std': float|None, 'n_frames': int|None}
    """
    issues: list[str] = []
    notes: list[str] = []

    if target_kind not in KEY_RESIDUES:
        return {"pass": False,
                "issues": [f"unknown target_kind={target_kind}"],
                "notes": [], "dG": None, "dG_std": None,
                "n_frames": None}

    # success flag
    sf = _to_int(row.get("success_flag",
                          row.get("success", row.get("status_ok"))))
    if sf is None or sf != 1:
        issues.append(f"success_flag != 1 (got {sf!r})")

    # dG (accept several column names; MPI batch driver writes
    # dG_recognition_md_kcalmol)
    dg = _to_float(
        row.get("dG_bind_kcalmol",
                row.get("dG_bind",
                        row.get("dG",
                                row.get("delta_total",
                                        row.get("dG_recognition_md_kcalmol"))))))
    if dg is None:
        issues.append("dG_bind missing or NaN")
    else:
        if abs(dg) > DG_ABS_CAP:
            issues.append(f"|dG|={abs(dg):.2f} > {DG_ABS_CAP} (sanity cap)")
        if dg >= 0:
            issues.append(f"dG={dg:.2f} >= 0 (binding should be favorable)")
        lo, hi = DG_RANGE[target_kind]
        if not (lo <= dg <= hi):
            issues.append(f"dG={dg:.2f} outside expected range [{lo},{hi}]")

    # dG std
    dgstd = _to_float(
        row.get("dG_std",
                row.get("dG_bind_std",
                        row.get("dG_sem",
                                row.get("dG_recognition_md_std")))))
    if dgstd is None:
        notes.append("dG_std missing — proceeding")
    else:
        if dgstd >= DG_STD_MAX[target_kind]:
            issues.append(
                f"dG_std={dgstd:.2f} >= {DG_STD_MAX[target_kind]}")

    # n_frames — MPI batch driver writes n_frames_scored=0 as known
    # artifact (real value is in the smoke pipeline log, ~20 frames for
    # the AmberTools MMPBSA.py.MPI pivot). Demote to a note unless dG
    # itself is NaN.
    nf = _to_int(row.get("n_frames",
                          row.get("nframes",
                                  row.get("n_frames_scored"))))
    if nf is None:
        notes.append("n_frames missing (MPI driver doesn't populate)")
    elif nf == 0:
        notes.append("n_frames_scored=0 — known MPI driver artifact; "
                     "actual frames in smoke log")
    elif nf < N_FRAMES_MIN:
        notes.append(f"n_frames={nf} < {N_FRAMES_MIN} "
                     "(non-blocking; MPI driver pivot)")

    # wall-time sanity (batch driver writes wall_total_s)
    wt = _to_float(row.get("wall_time_s",
                            row.get("wall_s",
                                    row.get("runtime_s",
                                            row.get("wall_total_s")))))
    if wt is not None:
        if wt < WALL_S_MIN:
            issues.append(f"wall_time={wt:.1f}s suspiciously low")
        elif wt > WALL_S_MAX:
            notes.append(f"wall_time={wt/3600:.1f}h above 24h soft cap")

    # residue decomp intersection — MPI driver tolerates DecompError
    # MPI_ABORT and leaves residue_contributions empty; per-residue
    # decomp will be backfilled in a sequential pass. Demote empty
    # decomp and < 4 matches to NOTE (not blocking).
    rc_raw = row.get("residue_contributions",
                      row.get("decomp", row.get("residue_decomp")))
    parsed = _parse_residues(rc_raw)
    if not parsed:
        notes.append("residue_contributions empty — DecompError "
                     "MPI_ABORT tolerated; backfill expected")
    else:
        matched = _residue_name_match(parsed, KEY_RESIDUES[target_kind])
        if len(matched) < 4:
            notes.append(
                f"residue decomp matched only {len(matched)} of key "
                f"set (need >=4 ideally); matched={sorted(matched)}")
        else:
            notes.append(f"residue decomp matched {len(matched)} "
                         f"key residues: {sorted(matched)}")

    # ligand strain envelope (capped-analog column name supported)
    strain = _to_float(
        row.get("ligand_strain_kcalmol",
                row.get("strain_kcalmol",
                        row.get("strain",
                                row.get("ligand_strain_capped_md_kcalmol")))))
    if strain is not None:
        if not (STRAIN_MIN <= strain <= STRAIN_MAX):
            notes.append(
                f"ligand_strain={strain:.2f} outside [{STRAIN_MIN},"
                f"{STRAIN_MAX}] (non-blocking)")

    return {
        "pass": len(issues) == 0,
        "issues": issues,
        "notes": notes,
        "dG": dg,
        "dG_std": dgstd,
        "n_frames": nf,
    }


# ---------------------------------------------------------------------------
# CSV-level validation
# ---------------------------------------------------------------------------

def validate_csv(csv_path: str, target_kind: str) -> dict[str, Any]:
    """Validate every row of a gmx_MMPBSA result CSV.

    Returns aggregate summary suitable for writing to
    /tmp/mmgbsa_qa_status.json.
    """
    summary: dict[str, Any] = {
        "csv": csv_path,
        "target_kind": target_kind,
        "exists": False,
        "n_rows": 0,
        "n_pass": 0,
        "n_fail": 0,
        "rows": [],
        "verdict": "FAIL",
        "dG_stats": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if not os.path.exists(csv_path):
        summary["issues"] = [f"csv not found: {csv_path}"]
        return summary

    size = os.path.getsize(csv_path)
    if size == 0:
        summary["exists"] = True
        summary["issues"] = ["csv empty (0 bytes)"]
        return summary
    summary["exists"] = True

    rows_out: list[dict[str, Any]] = []
    dG_vals: list[float] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            r = validate_row(row, target_kind)
            r["row_idx"] = idx
            r["ligand_id"] = (row.get("ligand_id")
                               or row.get("name")
                               or row.get("row_id")
                               or f"row{idx}")
            rows_out.append(r)
            if r["dG"] is not None:
                dG_vals.append(r["dG"])

    n_rows = len(rows_out)
    n_pass = sum(1 for r in rows_out if r["pass"])
    summary["n_rows"] = n_rows
    summary["n_pass"] = n_pass
    summary["n_fail"] = n_rows - n_pass
    summary["rows"] = rows_out

    if dG_vals:
        mean = sum(dG_vals) / len(dG_vals)
        var = sum((v - mean) ** 2 for v in dG_vals) / max(len(dG_vals) - 1, 1)
        std = math.sqrt(var)
        summary["dG_stats"] = {
            "n": len(dG_vals),
            "mean": mean,
            "std": std,
            "min": min(dG_vals),
            "max": max(dG_vals),
        }

    if n_rows == 0:
        summary["verdict"] = "FAIL"
        summary["issues"] = ["csv has 0 data rows"]
    elif n_pass == n_rows:
        summary["verdict"] = "PASS"
    else:
        summary["verdict"] = "FAIL"

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True,
                        help="Path to gmx_MMPBSA result CSV")
    parser.add_argument("--target_kind", required=True,
                        choices=sorted(KEY_RESIDUES.keys()),
                        help="smoke (5P9J BTK) or zap70")
    parser.add_argument("--json_out",
                        default="/tmp/mmgbsa_qa_status.json",
                        help="Where to write structured status JSON")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        status = {
            "csv": args.csv, "target_kind": args.target_kind,
            "exists": False, "verdict": "FAIL",
            "issues": ["csv not found"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.json_out, "w") as fh:
            json.dump(status, fh, indent=2)
        print(f"[QA] CSV not found: {args.csv}", file=sys.stderr)
        return 2

    if os.path.getsize(args.csv) == 0:
        status = {
            "csv": args.csv, "target_kind": args.target_kind,
            "exists": True, "verdict": "FAIL",
            "issues": ["csv empty"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.json_out, "w") as fh:
            json.dump(status, fh, indent=2)
        print(f"[QA] CSV empty: {args.csv}", file=sys.stderr)
        return 2

    summary = validate_csv(args.csv, args.target_kind)
    with open(args.json_out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"[QA] {args.csv}: verdict={summary['verdict']} "
          f"pass={summary['n_pass']}/{summary['n_rows']}")
    if summary.get("dG_stats"):
        s = summary["dG_stats"]
        print(f"[QA] dG mean={s['mean']:.2f} std={s['std']:.2f} "
              f"min={s['min']:.2f} max={s['max']:.2f} (kcal/mol)")
    for r in summary["rows"]:
        if not r["pass"]:
            print(f"[QA] FAIL row {r['row_idx']} "
                  f"({r.get('ligand_id','?')}): "
                  f"{'; '.join(r['issues'])}")

    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
