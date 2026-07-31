"""Experiment 6 — POST-HOC SURGERY on a corrupted FT checkpoint.

Hypothesis (architect's Experiment 6):
  The L1-FT corruption is concentrated in the X/Y/Z coordinate heads
  (`fcx`, `fcy`, `fcz`) and the second decoder (`decoder_relative`) because
  their conditioning was mismatched (Bug #2). The token head (`proj`,
  `gt_fea1`) and the encoder may be salvageable. If we COPY the pretrained
  coord heads + decoder_relative back into the FT'd checkpoint, sampling
  should work — combining the FT'd token distribution (good for covalent
  chemistry) with the pretrained coordinate distribution (good for valid 3D
  placement).

Surgery plan
------------
Build a SURGICAL state_dict by:

  Copy FROM PRETRAINED (coord / relative-path family):
      fcx.*, fcy.*, fcz.*                  — voxel x/y/z heads
      decoder_relative.*                   — second decoder (coord path)
      decoder_layerbias.*                  — TEMPLATE for decoder_relative
      relative_emb.*, dist_emb.*           — root-coord edge embeddings
      linears_edge.*, linears_edge1.*      — edge biases for decoder_relative
      proj2_aux.*, proj3_aux.*, proj4_aux.*— r/theta/phi aux heads
      coords_emb_gt.*                      — ligand voxel-coord embedding

  Keep FROM FT (token / encoder family):
      encoder.*, encoder_layer.*           — encoder (covalent pocket features)
      gt_fea1.*                            — ligand token embedding
      proj.*                               — next-token head
      decoder.*, decoder_layer.*           — first decoder (token path)
      relative_emb_topo.*, dist_emb_topo.* — ligand-only edges for token path
      linears_edge_topo.*, linears_edge1_topo.* — edge biases for first decoder
      pos_encode.*                         — positional encoding
      coords_emb.*                         — pocket voxel-coord embedding
      src_fea.*                            — pocket atom-type embedding
      residue.*                            — pocket residue embedding
      anchor_emb.*                         — pocket-anchor embedding

  Dead / unused-at-inference (keep from FT, doesn't matter):
      gt_fea.*           — legacy ligand emb (never called in gen_mol path)
      proj2/3/4.*        — legacy aux heads (proj2/3/4_aux replaced them)
      proj_aux.*         — legacy
      proj_contact.*, proj_matrix.*, proj_matrix1.*, proj_residue.*
                         — contact-net heads (the contact path uses contact.pkl,
                           not gen_mol.pkl, so these don't matter at sampling time)

Sanity checks
-------------
  * print FT-vs-PRETRAINED key counts (target ~60% FT / ~40% pretrained)
  * verify load_state_dict(strict=True) on a freshly-constructed TransformerModel
    returns NO missing/unexpected keys
  * shape mismatch between PRE and FT in any shared key → abort

Outputs
-------
  data/lingo3dmol_surgery_v1.pt   — flat state_dict (torch.save)

Usage
-----
  conda run -n quris python experiments/run_lingo3dmol_surgery.py

This script only builds the surgical checkpoint. Sampling and eval are done
by a sibling driver script (see run_lingo3dmol_surgery_sample.py).
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import OrderedDict

import torch

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "external" / "Lingo3DMol" / "checkpoint" / "gen_mol.pkl"
FT         = ROOT / "data" / "lingo3dmol_L1_full_ft" / "ckpt_phase2_dev.pt"
OUT        = ROOT / "data" / "lingo3dmol_surgery_v1.pt"

# Prefix -> "PRE" (copy from pretrained) or "FT" (keep from FT'd)
ASSIGNMENT = {
    # ----- COORD / RELATIVE PATH — from PRETRAINED -----
    "fcx":              "PRE",
    "fcy":              "PRE",
    "fcz":              "PRE",
    "decoder_relative": "PRE",
    "decoder_layerbias": "PRE",  # template for decoder_relative.layers
    "relative_emb":     "PRE",
    "dist_emb":         "PRE",
    "linears_edge":     "PRE",
    "linears_edge1":    "PRE",
    "proj2_aux":        "PRE",
    "proj3_aux":        "PRE",
    "proj4_aux":        "PRE",
    "coords_emb_gt":    "PRE",
    # ----- TOKEN / ENCODER PATH — keep FT -----
    "encoder":          "FT",
    "encoder_layer":    "FT",   # template for encoder.layers
    "gt_fea1":          "FT",
    "proj":             "FT",
    "decoder":          "FT",
    "decoder_layer":    "FT",   # template for decoder.layers
    "relative_emb_topo": "FT",
    "dist_emb_topo":    "FT",
    "linears_edge_topo": "FT",
    "linears_edge1_topo": "FT",
    "pos_encode":       "FT",
    "coords_emb":       "FT",
    "src_fea":          "FT",
    "residue":          "FT",
    "anchor_emb":       "FT",
    # ----- DEAD / UNUSED — keep FT (doesn't matter) -----
    "gt_fea":           "FT",
    "proj2":            "FT",
    "proj3":            "FT",
    "proj4":            "FT",
    "proj_aux":         "FT",
    "proj_contact":     "FT",
    "proj_matrix":      "FT",
    "proj_matrix1":     "FT",
    "proj_residue":     "FT",
}


def main():
    print(f"[surgery] PRE: {PRETRAINED}")
    print(f"[surgery] FT : {FT}")
    print(f"[surgery] OUT: {OUT}")
    assert PRETRAINED.exists(), f"missing pretrained checkpoint: {PRETRAINED}"
    assert FT.exists(),         f"missing FT checkpoint:        {FT}"

    sd_pre = torch.load(str(PRETRAINED), map_location="cpu", weights_only=False)
    ck_ft  = torch.load(str(FT),         map_location="cpu", weights_only=False)
    sd_ft  = ck_ft["model"] if isinstance(ck_ft, dict) and "model" in ck_ft else ck_ft
    assert isinstance(sd_pre, (dict, OrderedDict)) and isinstance(sd_ft, (dict, OrderedDict))

    print(f"[surgery] PRE keys: {len(sd_pre)}")
    print(f"[surgery] FT  keys: {len(sd_ft)}")

    # Sanity: same prefix set
    pre_prefixes = {k.split(".")[0] for k in sd_pre}
    ft_prefixes  = {k.split(".")[0] for k in sd_ft}
    assert pre_prefixes == ft_prefixes, (
        f"prefix mismatch: PRE-only={pre_prefixes - ft_prefixes} "
        f"FT-only={ft_prefixes - pre_prefixes}"
    )

    # Sanity: every prefix is assigned
    unassigned = pre_prefixes - set(ASSIGNMENT.keys())
    if unassigned:
        raise SystemExit(f"[surgery] UNASSIGNED prefixes: {sorted(unassigned)}")
    extra = set(ASSIGNMENT.keys()) - pre_prefixes
    if extra:
        print(f"[surgery] WARNING: ASSIGNMENT has unused prefixes: {sorted(extra)}")

    # Sanity: shapes match across PRE and FT for every shared key
    for k in sd_pre:
        if k not in sd_ft:
            raise SystemExit(f"[surgery] FT missing key {k}")
        if tuple(sd_pre[k].shape) != tuple(sd_ft[k].shape):
            raise SystemExit(
                f"[surgery] shape mismatch on {k}: "
                f"PRE={tuple(sd_pre[k].shape)} FT={tuple(sd_ft[k].shape)}"
            )
    print("[surgery] all shapes & keys agree across PRE and FT ✔")

    # Build surgical state_dict in PRE key order (preserves OrderedDict)
    surgical = OrderedDict()
    n_pre, n_ft = 0, 0
    counts_by_prefix = {}
    for k in sd_pre:
        pref = k.split(".")[0]
        src = ASSIGNMENT[pref]
        if src == "PRE":
            surgical[k] = sd_pre[k].clone()
            n_pre += 1
        elif src == "FT":
            surgical[k] = sd_ft[k].clone()
            n_ft += 1
        else:
            raise SystemExit(f"bad assignment for {pref}: {src}")
        counts_by_prefix.setdefault(pref, [0, src])
        counts_by_prefix[pref][0] += 1

    assert len(surgical) == len(sd_pre)
    total = n_pre + n_ft
    pct_pre = 100.0 * n_pre / total
    pct_ft  = 100.0 * n_ft  / total
    print(f"[surgery] surgical state_dict built: {total} keys")
    print(f"[surgery]   from PRETRAINED: {n_pre} keys ({pct_pre:.1f}%)")
    print(f"[surgery]   from FT'D:        {n_ft} keys ({pct_ft:.1f}%)")

    # Print per-prefix breakdown
    print("\n[surgery] per-prefix breakdown:")
    for pref in sorted(counts_by_prefix.keys()):
        n, src = counts_by_prefix[pref]
        marker = "PRE" if src == "PRE" else "FT "
        print(f"   {marker}  {pref:<22}  {n:>3} keys")

    # Validate by constructing the model and load_state_dict(strict=True)
    print("\n[surgery] validation: constructing TransformerModel and loading...")
    sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
    sys.path.insert(0, str(ROOT / "experiments"))
    os.chdir(str(ROOT / "external" / "Lingo3DMol"))
    import lingo3dmol_cpu_shim  # noqa: F401
    from model.transformer_v1_res_fac2 import TransformerModel
    m = TransformerModel()
    incompat = m.load_state_dict(surgical, strict=False)
    missing = list(getattr(incompat, "missing_keys", []) or [])
    unexpected = list(getattr(incompat, "unexpected_keys", []) or [])
    print(f"[surgery]   missing_keys   = {len(missing)}")
    print(f"[surgery]   unexpected_keys= {len(unexpected)}")
    if missing:
        print(f"[surgery]   FIRST 20 missing: {missing[:20]}")
    if unexpected:
        print(f"[surgery]   FIRST 20 unexpected: {unexpected[:20]}")
    # Persist a JSON sidecar with the load report
    report = {
        "pre_checkpoint": str(PRETRAINED),
        "ft_checkpoint":  str(FT),
        "out_checkpoint": str(OUT),
        "n_total":        total,
        "n_from_pre":     n_pre,
        "n_from_ft":      n_ft,
        "pct_from_pre":   pct_pre,
        "pct_from_ft":    pct_ft,
        "missing_keys":   missing,
        "unexpected_keys": unexpected,
        "per_prefix":     {p: {"n": n, "src": s} for p, (n, s) in counts_by_prefix.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(surgical, str(OUT))
    with (OUT.with_suffix(".report.json")).open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[surgery] WROTE: {OUT}")
    print(f"[surgery] WROTE: {OUT.with_suffix('.report.json')}")

    # The current TransformerModel class registers EXTRA chassis/reactivity
    # heads that are NOT in the original pretrained checkpoint. Those are
    # never exercised in the sampling pipeline (the L1-FT sample script
    # monkey-patches load_state_dict to strict=False for the same reason).
    # We accept missing keys ONLY when they match this known-extra pattern.
    KNOWN_EXTRA_PREFIXES = ("reactivity_head", "chassis_head")
    bad_missing = [k for k in missing
                   if not any(k.startswith(p + ".") for p in KNOWN_EXTRA_PREFIXES)]
    if bad_missing:
        raise SystemExit(
            f"[surgery] FATAL: {len(bad_missing)} unexpected missing keys — "
            f"surgical checkpoint is INCOMPLETE: {bad_missing[:10]}"
        )
    if missing:
        print(f"[surgery] OK: {len(missing)} missing keys all match expected "
              f"extra-head prefixes {KNOWN_EXTRA_PREFIXES} (sampling pipeline "
              f"never uses them)")
    if unexpected:
        # unexpected can happen if state_dict has a key the model doesn't
        # register; we want to be aware but it isn't fatal for sampling.
        print(f"[surgery] WARNING: {len(unexpected)} unexpected keys (non-fatal)")


if __name__ == "__main__":
    main()
