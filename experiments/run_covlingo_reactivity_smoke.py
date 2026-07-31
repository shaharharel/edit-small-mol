"""CovalentLingo Full — reactivity head smoke test (Phase C).

Verifies that the new `reactivity_head` (predicting xTB log_k2_GSH from the
LIGAND decoder representation) trains end-to-end:

  - dataloader returns `log_k2_target: (B,)` float tensor with NaN where
    the SMILES has no acrylamide warhead / xTB label
  - model `forward_train(..., log_k2_target=...)` returns a finite
    `loss_reactivity` and a `reactivity_n_valid` count
  - over 100 SGD iters on a small acrylamide subset, `loss_reactivity`
    DROPS monotonically (modulo SGD noise)
  - the BASE 4 losses (token, x, y, z) DO NOT blow up — the head is
    additive, weighted at `reactivity_weight=0.2`
  - on a held-out subset, Spearman(pred, true) > random — gradient flows
    from the regression target through the decoder into the head

CPU only. ~1 hr wall budget.

Usage:
    python experiments/run_covlingo_reactivity_smoke.py \
        --reactivity_csv data/covindb_xtb_reactivity.csv \
        --complex_csv ~/Downloads/CovInDB2/CovInDB/Covalent_Complex_Records.csv \
        --pdb_dir ~/Downloads/CovInDB2/CovInDB/PDB \
        --out_dir data/covlingo_reactivity_smoke

We deliberately restrict the dataset to rows whose SMILES has a non-NaN
label so the smoke isn't dominated by NaN-masked batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# CPU shim must be installed BEFORE any external/Lingo3DMol import.
import lingo3dmol_cpu_shim  # noqa: F401, E402

sys.path.insert(0, str(HERE.parent / "external" / "Lingo3DMol"))

from lingo3dmol_l1_dataloader import (  # noqa: E402
    CovalentInDB2Dataset,
    _canon_smiles,
    collate_fn,
)
from model.transformer_v1_res_fac2 import TransformerModel  # noqa: E402


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Returns NaN if either array constant."""
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _filter_labelled_rows(complex_csv: str, reactivity_csv: str) -> pd.DataFrame:
    """Return rows of the complex CSV whose SMILES has a non-NaN log_k2 label."""
    cdf = pd.read_csv(complex_csv)
    rdf = pd.read_csv(reactivity_csv)
    # Build canonical-SMILES -> log_k2 lookup (non-NaN only).
    smi_to_k2: dict[str, float] = {}
    for _, r in rdf.iterrows():
        if pd.isna(r["log_k2_GSH"]):
            continue
        s = str(r["smiles"]).strip()
        c = _canon_smiles(s)
        if c is not None:
            smi_to_k2.setdefault(c, float(r["log_k2_GSH"]))
        smi_to_k2.setdefault(s, float(r["log_k2_GSH"]))
    # Mark each row labelled / not.
    keep = []
    for s in cdf["SMILES"].fillna("").astype(str):
        s = s.strip()
        c = _canon_smiles(s)
        if (c is not None and c in smi_to_k2) or s in smi_to_k2:
            keep.append(True)
        else:
            keep.append(False)
    out = cdf[keep].reset_index(drop=True)
    print(f"[filter] {len(out)}/{len(cdf)} complex rows have a reactivity label")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reactivity_csv",
        default="data/covindb_xtb_reactivity.csv",
    )
    ap.add_argument(
        "--complex_csv",
        default=str(Path.home()
                    / "Downloads/CovInDB2/CovInDB/Covalent_Complex_Records.csv"),
    )
    ap.add_argument(
        "--pdb_dir",
        default=str(Path.home() / "Downloads/CovInDB2/CovInDB/PDB"),
    )
    ap.add_argument(
        "--cache_dir",
        default="data/lingo3dmol_L1_cache",
    )
    ap.add_argument("--out_dir", default="data/covlingo_reactivity_smoke")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reactivity_weight", type=float, default=0.2)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--n_train_complexes", type=int, default=32)
    ap.add_argument("--n_eval_complexes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] out_dir={out_dir}")

    # ----------------------------------------------------------------
    # 1) Build a complex CSV restricted to LABELLED rows. We need both
    #    the smoke-train and held-out-eval samples to have valid labels.
    # ----------------------------------------------------------------
    labelled_df = _filter_labelled_rows(args.complex_csv, args.reactivity_csv)
    # Shuffle for diverse train/eval. Use deterministic seed.
    labelled_df = labelled_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_total = args.n_train_complexes + args.n_eval_complexes
    if len(labelled_df) < n_total:
        n_total = len(labelled_df)
        args.n_train_complexes = max(1, int(0.8 * n_total))
        args.n_eval_complexes = n_total - args.n_train_complexes
        print(f"[smoke] WARNING: only {len(labelled_df)} labelled rows; "
              f"shrinking to train={args.n_train_complexes} eval={args.n_eval_complexes}")
    sub_csv = out_dir / "_labelled_subset.csv"
    labelled_df.head(n_total).to_csv(sub_csv, index=False)

    # ----------------------------------------------------------------
    # 2) Build train / eval datasets (separate slices of the labelled csv).
    # ----------------------------------------------------------------
    print(f"[smoke] building TRAIN dataset (n={args.n_train_complexes})")
    train_ds = CovalentInDB2Dataset(
        complex_csv_path=str(sub_csv),
        pdb_dir=args.pdb_dir,
        T=args.T,
        max_complexes=args.n_train_complexes,
        cache_dir=args.cache_dir,
        pocket_radius=15.0,
        verbose=False,
        reactivity_csv_path=args.reactivity_csv,
    )

    # Eval dataset: same labelled-subset CSV but with offset selection.
    # We sniff the offset by writing a second CSV with the held-out tail.
    eval_csv = out_dir / "_labelled_eval.csv"
    labelled_df.iloc[args.n_train_complexes:n_total].to_csv(eval_csv, index=False)
    print(f"[smoke] building EVAL dataset (n={args.n_eval_complexes})")
    eval_ds = CovalentInDB2Dataset(
        complex_csv_path=str(eval_csv),
        pdb_dir=args.pdb_dir,
        T=args.T,
        max_complexes=args.n_eval_complexes,
        cache_dir=args.cache_dir,
        pocket_radius=15.0,
        verbose=False,
        reactivity_csv_path=args.reactivity_csv,
    )

    # ----------------------------------------------------------------
    # 3) Materialise a static smoke-train batch and an eval batch.
    #    (DataLoader workers would just slow CPU runs down.)
    # ----------------------------------------------------------------
    def _materialise(ds, max_items: int) -> list[dict]:
        items: list[dict] = []
        attempted = 0
        while len(items) < max_items and attempted < len(ds) * 2:
            try:
                items.append(ds[attempted % len(ds)])
            except Exception as e:
                print(f"[smoke] ds[{attempted}] failed: {e}")
            attempted += 1
        return items

    train_items = _materialise(train_ds, args.n_train_complexes)
    eval_items = _materialise(eval_ds, args.n_eval_complexes)
    print(f"[smoke] materialised train={len(train_items)} eval={len(eval_items)}")
    if len(train_items) < args.bs:
        raise RuntimeError(
            f"Not enough train items ({len(train_items)}) for bs={args.bs}"
        )

    # Verify labels are present and non-NaN in the materialised items.
    train_labels = np.array([float(it["log_k2_target"].item()) for it in train_items])
    eval_labels = np.array([float(it["log_k2_target"].item()) for it in eval_items])
    n_train_nan = int(np.isnan(train_labels).sum())
    n_eval_nan = int(np.isnan(eval_labels).sum())
    print(f"[smoke] train labels: n={len(train_labels)} "
          f"min={np.nanmin(train_labels):.3f} max={np.nanmax(train_labels):.3f} "
          f"nan={n_train_nan}")
    print(f"[smoke] eval labels:  n={len(eval_labels)}  "
          f"min={np.nanmin(eval_labels):.3f} max={np.nanmax(eval_labels):.3f} "
          f"nan={n_eval_nan}")

    # ----------------------------------------------------------------
    # 4) Build model. FRESH init (no pretrained checkpoint loading).
    # ----------------------------------------------------------------
    print("[smoke] building TransformerModel (fresh init)")
    model = TransformerModel()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ----------------------------------------------------------------
    # 5) Training loop. We repeat the SAME train_items by drawing random
    #    bs-sized mini-batches without replacement per pass.
    # ----------------------------------------------------------------
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "wall_sec",
            "loss_total", "loss_token", "loss_x", "loss_y", "loss_z",
            "loss_reactivity", "reactivity_n_valid",
        ])

    t0 = time.time()
    history: list[dict] = []
    rng = np.random.default_rng(args.seed)
    for step in range(args.iters):
        # Random minibatch of size bs from train_items.
        idx = rng.choice(len(train_items), size=args.bs, replace=False)
        batch = collate_fn([train_items[int(i)] for i in idx])
        optimizer.zero_grad()
        out = model.forward_train(**batch, reactivity_weight=args.reactivity_weight)
        loss = out["loss_total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        rec = {
            "step": step,
            "wall_sec": time.time() - t0,
            "loss_total":      float(out["loss_total"].detach()),
            "loss_token":      float(out["loss_token"]),
            "loss_x":          float(out["loss_x"]),
            "loss_y":          float(out["loss_y"]),
            "loss_z":          float(out["loss_z"]),
            "loss_reactivity": float(out["loss_reactivity"]),
            "reactivity_n_valid": int(out["reactivity_n_valid"]),
        }
        history.append(rec)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                rec["step"], rec["wall_sec"],
                rec["loss_total"], rec["loss_token"],
                rec["loss_x"], rec["loss_y"], rec["loss_z"],
                rec["loss_reactivity"], rec["reactivity_n_valid"],
            ])
        if step % max(1, args.iters // 10) == 0 or step == args.iters - 1:
            print(
                f"  step={step:3d} t={rec['wall_sec']:6.1f}s  "
                f"tot={rec['loss_total']:.3f} "
                f"tok={rec['loss_token']:.3f} "
                f"x={rec['loss_x']:.3f} y={rec['loss_y']:.3f} z={rec['loss_z']:.3f} "
                f"react={rec['loss_reactivity']:.3f} (n={rec['reactivity_n_valid']})"
            )

    # ----------------------------------------------------------------
    # 6) Held-out eval: run the model on eval_items and Spearman-rank
    #    predicted vs true log_k2.
    # ----------------------------------------------------------------
    print("[smoke] held-out evaluation")
    model.eval()
    preds: list[float] = []
    trues: list[float] = []
    with torch.no_grad():
        # Evaluate one-by-one to keep memory low.
        for it in eval_items:
            batch = collate_fn([it])
            out = model.forward_train(**batch, reactivity_weight=args.reactivity_weight)
            pred = float(out["log_k2_pred"].squeeze().item())
            tgt = float(it["log_k2_target"].item())
            if not np.isnan(tgt):
                preds.append(pred)
                trues.append(tgt)
    preds_a = np.asarray(preds)
    trues_a = np.asarray(trues)
    spr = _spearman(preds_a, trues_a)
    pea = _pearson(preds_a, trues_a)
    mae = float(np.mean(np.abs(preds_a - trues_a))) if len(preds_a) else float("nan")
    print(f"[eval] n={len(preds_a)} Spearman={spr:.3f} Pearson={pea:.3f} MAE={mae:.3f}")

    # ----------------------------------------------------------------
    # 7) Stability check — early vs late base losses (mean of first 10
    #    vs last 10 steps).
    # ----------------------------------------------------------------
    hist_df = pd.DataFrame(history)
    early = hist_df.head(10).mean(numeric_only=True)
    late = hist_df.tail(10).mean(numeric_only=True)
    stability = {
        "loss_token_early": float(early["loss_token"]),
        "loss_token_late":  float(late["loss_token"]),
        "loss_x_early":     float(early["loss_x"]),
        "loss_x_late":      float(late["loss_x"]),
        "loss_y_early":     float(early["loss_y"]),
        "loss_y_late":      float(late["loss_y"]),
        "loss_z_early":     float(early["loss_z"]),
        "loss_z_late":      float(late["loss_z"]),
        "loss_reactivity_early": float(early["loss_reactivity"]),
        "loss_reactivity_late":  float(late["loss_reactivity"]),
    }
    print("[stability]")
    for k, v in stability.items():
        print(f"  {k}: {v:.3f}")

    # Persist everything for the report.
    eval_rows = pd.DataFrame({
        "true_log_k2": trues_a,
        "pred_log_k2": preds_a,
    })
    eval_rows.to_csv(out_dir / "eval_pairs.csv", index=False)
    with open(out_dir / "smoke_results.json", "w") as f:
        json.dump({
            "args": vars(args),
            "stability": stability,
            "eval": {
                "n": len(preds_a),
                "spearman": spr,
                "pearson": pea,
                "mae": mae,
            },
            "history_first_step": history[0] if history else None,
            "history_last_step":  history[-1] if history else None,
        }, f, indent=2)
    print(f"[done] train_log={log_path}")
    print(f"[done] eval_pairs={out_dir / 'eval_pairs.csv'}")
    print(f"[done] smoke_results={out_dir / 'smoke_results.json'}")


if __name__ == "__main__":
    main()
