"""PROPKA3 Cys346 pKa calculation for a single Boltz cofold PDB.

Per Olsson 2011 / Søndergaard 2011, PROPKA gives a per-pose,
ligand-environment-aware estimate of the nucleophile pKa.
Together with xTB warhead electrophilicity, this completes the
covalent-rate story (nucleophile activation + electrophile).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def _parse_pka_summary_for_cys346(pka_path: Path) -> tuple[float | None, str | None]:
    """Parse a PROPKA .pka file and return (pKa_Cys346, model_version).

    PROPKA SUMMARY block format:
       Group      pKa  model-pKa   ligand atom-type
       CYS 346 A     9.26       9.00
    """
    text = pka_path.read_text(errors="replace")
    # Version is the first non-empty line, e.g. "propka3.5.1   2026-05-26"
    model_version = None
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("propka"):
            model_version = s.split()[0]
            break

    # Extract SUMMARY block
    summary_match = re.search(
        r"SUMMARY OF THIS PREDICTION(.*?)(?:\n\s*\n|\Z)",
        text,
        re.S,
    )
    if not summary_match:
        return None, model_version
    summary_block = summary_match.group(1)

    # Match "CYS 346 <chain>  <pKa>  <model_pKa>" — accept any chain id
    cys_re = re.compile(
        r"^\s*CYS\s+346\s+([A-Za-z0-9])\s+([\-\d\.]+)\s+([\-\d\.]+)",
        re.M,
    )
    matches = cys_re.findall(summary_block)
    if not matches:
        return None, model_version

    # Prefer chain "A"; otherwise pick the first
    chosen = None
    for chain, pka, _mod in matches:
        if chain.upper() == "A":
            chosen = pka
            break
    if chosen is None:
        chosen = matches[0][1]

    try:
        return float(chosen), model_version
    except ValueError:
        return None, model_version


def compute_cys346_pka(pdb_path: str | Path) -> dict:
    """Run PROPKA3 on a PDB file and return Cys346 pKa info.

    Returns dict with keys: pKa_Cys346, pKa_model, success_flag, error.
    """
    pdb_path = Path(pdb_path)
    out = {
        "pKa_Cys346": None,
        "pKa_model": None,
        "success_flag": 0,
        "error": "",
    }

    if not pdb_path.exists():
        out["error"] = f"PDB not found: {pdb_path}"
        return out

    # PROPKA writes the .pka into the CWD. Use a temp dir so concurrent workers
    # don't clobber each other and to keep the tree clean.
    tmpdir = Path(tempfile.mkdtemp(prefix="propka_"))
    try:
        local_pdb = tmpdir / pdb_path.name
        shutil.copy2(pdb_path, local_pdb)
        try:
            proc = subprocess.run(
                ["propka3", local_pdb.name],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            out["error"] = "propka3 timeout (180s)"
            return out
        except FileNotFoundError:
            out["error"] = "propka3 binary not found in PATH"
            return out

        pka_path = tmpdir / (local_pdb.stem + ".pka")
        if not pka_path.exists():
            # Some inputs cause propka to exit non-zero; capture stderr tail.
            err = (proc.stderr or "").strip().splitlines()
            tail = " | ".join(err[-3:]) if err else f"rc={proc.returncode}"
            out["error"] = f"no .pka produced: {tail}"
            return out

        pka_val, version = _parse_pka_summary_for_cys346(pka_path)
        out["pKa_model"] = version
        if pka_val is None:
            out["error"] = "Cys346 not found in SUMMARY"
            return out

        out["pKa_Cys346"] = pka_val
        out["success_flag"] = 1
        return out
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("pdb", help="Path to PDB file")
    args = ap.parse_args()
    print(json.dumps(compute_cys346_pka(args.pdb), indent=2))
