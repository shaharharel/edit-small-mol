"""Phase C — Chassis-prior head smoke test.

Goal:
  Validate that the chassis-prior head (added in Phase B to
  TransformerModel) can be trained on real CovInDB cocrystals and that
  loss_chassis decreases over 100 iters on a small balanced train set.

Setup:
  - 20 (default) train cocrystals + 5 val cocrystals, balanced across the
    chassis families produced by Phase A clustering
    (`data/covindb_chassis_labels.csv`).
  - bs=4, 100 iters, AdamW lr=1e-4 on the chassis_head ONLY by default
    (other params frozen — keeps the smoke fast and isolates the head).
  - CPU only. Wall budget ~30 min on Mac CPU.

Verifies:
  1. chassis_logits are NOT all-zero after training.
  2. loss_chassis decreases over 100 iters (linear regression slope < 0).
  3. Train-set argmax accuracy > random (1/N).
  4. Val-set argmax accuracy (held-out) for diagnostic.

Outputs:
  data/chassis_head_smoke/
    smoke_log.csv         — per-step loss
    smoke_summary.json    — final accuracies + slope + verdict
    smoke_loss_curve.png  — loss vs step (matplotlib, if installed)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# CPU shim must be installed before any Lingo3DMol import.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))

import lingo3dmol_cpu_shim  # noqa: F401

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model.transformer_v1_res_fac2 import TransformerModel
from lingo3dmol_l1_dataloader import CovalentInDB2Dataset, collate_fn


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    return str((ROOT / p).resolve())


def select_smoke_pdbs(
    labels_csv: str,
    pdb_dir: str,
    n_per_family_train: int = 2,
    n_per_family_val: int = 1,
    n_classes_cap: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick a balanced train/val split across chassis families.

    Returns (train_df, val_df) — both have the columns that
    CovalentInDB2Dataset's lower-case schema branch expects:
      pdb_id, ligand_chain, ligand_resi, ligand_resname,
      cys_chain, cys_resi, smiles  (+ Resi_name synthesized).
    Source columns come from the raw Covalent_Complex_Records.csv joined
    on pdb_id + ligand_smiles.
    """
    labels = pd.read_csv(labels_csv)
    labels["pdb_id"] = labels["pdb_id"].str.upper()
    complex_csv = str(Path.home() / "Downloads/CovInDB2/CovInDB/Covalent_Complex_Records.csv")
    cx = pd.read_csv(complex_csv)
    cx["PDB"] = cx["PDB"].astype(str).str.upper()
    cx = cx.dropna(subset=["PDB", "SMILES", "Ligand_chain",
                            "Ligand_position", "Resi_chain", "Resi_posi"])

    # Map labels -> complex rows by (pdb_id, smiles). One label can match many
    # complex rows (multiple ligand chains/positions per PDB); pick the first.
    merged_rows = []
    for _, lab in labels.iterrows():
        if lab["chassis_family_id"] < 0:
            continue
        match = cx[(cx["PDB"] == lab["pdb_id"]) &
                   (cx["SMILES"] == lab["ligand_smiles"])]
        if match.empty:
            continue
        row = match.iloc[0]
        if not (Path(pdb_dir) / f"{lab['pdb_id']}.pdb").exists():
            continue
        merged_rows.append({
            "PDB":             row["PDB"],
            "Ligand_chain":    row["Ligand_chain"],
            "Ligand_position": row["Ligand_position"],
            "Ligand_name":     row["Ligand_name"],
            "Resi_chain":      row["Resi_chain"],
            "Resi_posi":       row["Resi_posi"],
            "Resi_name":       row.get("Resi_name", "CYS"),
            "SMILES":          row["SMILES"],
            "chassis_family_id": int(lab["chassis_family_id"]),
            "warhead_class":     lab.get("warhead_class", ""),
        })
    merged = pd.DataFrame(merged_rows)
    print(f"[smoke-pick] merged labels+complex rows: {len(merged)}")
    if n_classes_cap is not None:
        merged = merged[merged["chassis_family_id"] < n_classes_cap]
        print(f"[smoke-pick] after n_classes_cap={n_classes_cap}: {len(merged)}")

    train_rows: list[pd.DataFrame] = []
    val_rows: list[pd.DataFrame] = []
    for fid, sub in merged.groupby("chassis_family_id"):
        sub = sub.reset_index(drop=True).head(n_per_family_train + n_per_family_val)
        train_rows.append(sub.iloc[:n_per_family_train])
        if len(sub) > n_per_family_train:
            val_rows.append(sub.iloc[n_per_family_train:n_per_family_train + n_per_family_val])

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    val_df   = pd.concat(val_rows,   ignore_index=True) if val_rows   else pd.DataFrame()
    print(f"[smoke-pick] train: {len(train_df)} rows, "
          f"{train_df['chassis_family_id'].nunique()} families")
    print(f"[smoke-pick] val:   {len(val_df)} rows, "
          f"{val_df['chassis_family_id'].nunique()} families")
    return train_df, val_df


def write_dataset_csv(df: pd.DataFrame, path: str):
    """CovalentInDB2Dataset accepts both upper-case (raw CovInDB) and
    lower-case (curated) schemas. We write the upper-case columns directly
    so the dataset's rename_map is a no-op.
    """
    cols = ["PDB", "Ligand_chain", "Ligand_position", "Ligand_name",
            "Resi_chain", "Resi_posi", "Resi_name", "SMILES"]
    df[cols].to_csv(path, index=False)


def freeze_all_except_chassis(model: torch.nn.Module) -> tuple[int, int]:
    frozen = 0
    trainable = 0
    for name, p in model.named_parameters():
        if name.startswith("chassis_head"):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    return frozen, trainable


def main(args):
    out_dir = Path(_abs(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Either use override CSVs (fast path for re-runs over cached pockets)
    # OR pick balanced train/val cocrystals from the full labeled set.
    if args.train_csv_override:
        train_csv = Path(_abs(args.train_csv_override))
        val_csv   = (Path(_abs(args.val_csv_override))
                     if args.val_csv_override else None)
        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv) if val_csv and val_csv.exists() else pd.DataFrame()
        print(f"[smoke] using override CSVs: train={train_csv} ({len(train_df)} rows), "
              f"val={val_csv if val_csv else 'NONE'} ({len(val_df)} rows)")
    else:
        train_df, val_df = select_smoke_pdbs(
            labels_csv=_abs(args.chassis_labels_csv),
            pdb_dir=_abs(args.pdb_dir),
            n_per_family_train=args.n_per_family_train,
            n_per_family_val=args.n_per_family_val,
            n_classes_cap=args.n_chassis_classes,
        )
        if len(train_df) == 0:
            raise RuntimeError("Smoke train set is empty — chassis labels or PDBs missing.")

        train_csv = out_dir / "smoke_train.csv"
        val_csv   = out_dir / "smoke_val.csv"
        write_dataset_csv(train_df, str(train_csv))
        if len(val_df):
            write_dataset_csv(val_df, str(val_csv))
        print(f"[smoke] wrote {train_csv} and {val_csv}")

    # 2) Build datasets / loaders.
    train_ds = CovalentInDB2Dataset(
        complex_csv_path=str(train_csv),
        pdb_dir=_abs(args.pdb_dir),
        T=args.seq_len,
        cache_dir=str(cache_dir),
        pocket_radius=args.pocket_radius,
        chassis_labels_csv=_abs(args.chassis_labels_csv),
        verbose=False,
    )
    val_ds = None
    if len(val_df) and val_csv is not None:
        val_ds = CovalentInDB2Dataset(
            complex_csv_path=str(val_csv),
            pdb_dir=_abs(args.pdb_dir),
            T=args.seq_len,
            cache_dir=str(cache_dir),
            pocket_radius=args.pocket_radius,
            chassis_labels_csv=_abs(args.chassis_labels_csv),
            verbose=False,
        )
    print(f"[smoke] train_ds={len(train_ds)}  val_ds={len(val_ds) if val_ds else 0}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    # 3) Build model. n_chassis_classes from the families JSON if available.
    fam_json_path = Path(args.chassis_labels_csv).with_name("covindb_chassis_families.json")
    n_classes = args.n_chassis_classes
    if fam_json_path.exists():
        fj = json.loads(open(fam_json_path).read())
        n_classes = int(fj.get("n_classes", n_classes))
        print(f"[smoke] using n_chassis_classes={n_classes} from {fam_json_path}")

    model = TransformerModel(n_chassis_classes=n_classes)
    model = model.to(args.device)

    # Optionally load pretrained encoder.
    if args.pretrained_ckpt and Path(_abs(args.pretrained_ckpt)).exists():
        sd = torch.load(_abs(args.pretrained_ckpt), map_location=args.device, weights_only=False)
        info = model.load_state_dict(sd, strict=False)
        print(f"[smoke] loaded pretrained: missing={len(info.missing_keys)} "
              f"unexpected={len(info.unexpected_keys)}")
        # chassis_head.* will be in missing; that's expected.
    else:
        print(f"[smoke] no pretrained ckpt — random init")

    # 4) Freeze everything except chassis_head (smoke focus: just the head).
    if args.head_only:
        frozen, trainable = freeze_all_except_chassis(model)
        print(f"[smoke] head-only mode: frozen={frozen/1e6:.2f}M  "
              f"trainable={trainable/1e3:.1f}K params")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)

    # 5) Snapshot the head's pre-training output to detect "all-zero" head.
    model.eval()
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        out_init = model.forward_train(**first_batch)
    chassis_logits_init = out_init["chassis_logits"]
    is_all_zero_init = bool((chassis_logits_init.abs() < 1e-8).all().item())
    print(f"[smoke] init chassis_logits shape={tuple(chassis_logits_init.shape)} "
          f"all_zero={is_all_zero_init}  std={chassis_logits_init.std().item():.4f}")

    # 6) Train loop — 100 iters, sample from the train loader on repeat.
    csv_path = out_dir / "smoke_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "wall_sec", "loss_total", "loss_chassis",
                    "loss_token", "loss_x", "loss_y", "loss_z"])

    model.train()
    t0 = time.time()
    history: list[dict] = []
    step = 0
    while step < args.n_iters:
        for batch in train_loader:
            if step >= args.n_iters:
                break
            opt.zero_grad()
            out = model.forward_train(**batch)
            loss = out["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            opt.step()
            row = {
                "step":         step,
                "wall_sec":     time.time() - t0,
                "loss_total":   float(out["loss_total"].detach()),
                "loss_chassis": float(out.get("loss_chassis",
                                              torch.tensor(float("nan"))).detach())
                                if "loss_chassis" in out else float("nan"),
                "loss_token":   float(out["loss_token"]),
                "loss_x":       float(out["loss_x"]),
                "loss_y":       float(out["loss_y"]),
                "loss_z":       float(out["loss_z"]),
            }
            history.append(row)
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row["step"], row["wall_sec"],
                                         row["loss_total"], row["loss_chassis"],
                                         row["loss_token"], row["loss_x"],
                                         row["loss_y"], row["loss_z"]])
            if step % max(1, args.n_iters // 10) == 0:
                print(f"  step={step:3d} t={row['wall_sec']:6.1f}s "
                      f"loss_total={row['loss_total']:.4f} "
                      f"loss_chassis={row['loss_chassis']:.4f}")
            step += 1
    print(f"[smoke] training done in {time.time() - t0:.1f}s, {step} steps")

    # 7) Compute final train + val accuracies on the chassis head.
    model.eval()

    def _accuracy(ds, name):
        if ds is None or len(ds) == 0:
            return None
        total = 0
        hit = 0
        per_family = defaultdict(lambda: [0, 0])
        ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, drop_last=False)
        with torch.no_grad():
            for batch in ld:
                out = model.forward_train(**batch)
                logits = out["chassis_logits"]
                preds = logits.argmax(dim=-1).cpu().numpy()
                labels = batch["chassis_label"].cpu().numpy()
                for p, l in zip(preds, labels):
                    if l == -1:
                        continue
                    total += 1
                    per_family[int(l)][0] += 1
                    if int(p) == int(l):
                        hit += 1
                        per_family[int(l)][1] += 1
        acc = hit / max(total, 1)
        print(f"[smoke] {name} accuracy: {hit}/{total} = {acc:.3f}")
        for fid, (t, h) in sorted(per_family.items()):
            print(f"  fam {fid}: {h}/{t} = {h/max(t,1):.2f}")
        return {"accuracy": acc, "hits": hit, "total": total,
                "per_family": dict(per_family)}

    train_metrics = _accuracy(train_ds, "TRAIN")
    val_metrics   = _accuracy(val_ds,   "VAL  ") if val_ds else None

    # 8) Slope on loss_chassis to flag learning vs flat.
    losses = np.array([r["loss_chassis"] for r in history], dtype=np.float64)
    valid = np.isfinite(losses)
    if valid.sum() >= 3:
        x = np.arange(len(losses))[valid]
        y = losses[valid]
        slope, intercept = np.polyfit(x, y, 1)
        n_avg = min(10, max(1, len(y) // 3))
        first10 = float(y[:n_avg].mean())
        last10  = float(y[-n_avg:].mean())
    else:
        slope = float("nan"); intercept = float("nan")
        first10 = float("nan"); last10 = float("nan")
    print(f"[smoke] loss_chassis  slope={slope:+.4e}  "
          f"first10_mean={first10:.4f}  last10_mean={last10:.4f}")

    # 9) Final logits diagnostic.
    with torch.no_grad():
        out_final = model.forward_train(**first_batch)
    chassis_logits_final = out_final["chassis_logits"]
    is_all_zero_final = bool((chassis_logits_final.abs() < 1e-8).all().item())
    std_change = (float(chassis_logits_final.std().item()) -
                  float(chassis_logits_init.std().item()))

    # 10) Verdict.
    chance = 1.0 / max(n_classes, 1)
    learning_decrease = (last10 < first10 - 0.05) and (slope < 0)
    learning_acc = (train_metrics is not None and
                    train_metrics["accuracy"] > chance + 0.05)
    verdict = "OK"
    if is_all_zero_final:
        verdict = "BLOCKER: chassis_logits remain all-zero after training"
    elif not learning_decrease and not learning_acc:
        verdict = "BLOCKER: loss did not decrease AND accuracy at chance"
    elif not learning_decrease:
        verdict = "WARN: loss did not decrease, but accuracy > chance"
    elif not learning_acc:
        verdict = "WARN: loss decreased but accuracy at chance"

    summary = {
        "n_chassis_classes": n_classes,
        "n_iters":           args.n_iters,
        "batch_size":        args.batch_size,
        "lr":                args.lr,
        "head_only":         args.head_only,
        "n_train_complexes": len(train_ds),
        "n_val_complexes":   len(val_ds) if val_ds else 0,
        "chassis_logits_init_all_zero":  is_all_zero_init,
        "chassis_logits_final_all_zero": is_all_zero_final,
        "chassis_logits_std_init":       float(chassis_logits_init.std().item()),
        "chassis_logits_std_final":      float(chassis_logits_final.std().item()),
        "chassis_logits_std_change":     std_change,
        "loss_chassis_slope":            float(slope),
        "loss_chassis_first10_mean":     first10,
        "loss_chassis_last10_mean":      last10,
        "loss_chassis_drop":             float(first10 - last10),
        "chance_accuracy":               chance,
        "train_accuracy":                train_metrics["accuracy"] if train_metrics else None,
        "train_hits":                    train_metrics["hits"] if train_metrics else None,
        "train_total":                   train_metrics["total"] if train_metrics else None,
        "val_accuracy":                  val_metrics["accuracy"] if val_metrics else None,
        "val_hits":                      val_metrics["hits"] if val_metrics else None,
        "val_total":                     val_metrics["total"] if val_metrics else None,
        "verdict":                       verdict,
    }
    with open(out_dir / "smoke_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 11) Plot loss curve if matplotlib available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = [r["step"] for r in history]
        ax.plot(steps, [r["loss_chassis"] for r in history], label="loss_chassis", color="tab:red")
        ax.set_xlabel("step")
        ax.set_ylabel("loss_chassis (CE)")
        ax.set_title(f"Chassis head smoke ({n_classes} classes, "
                     f"chance={chance:.2f})")
        ax.axhline(np.log(n_classes), linestyle=":", color="gray",
                   label=f"random chance = log({n_classes}) = {np.log(n_classes):.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "smoke_loss_curve.png", dpi=120)
        plt.close(fig)
        print(f"[smoke] plot: {out_dir / 'smoke_loss_curve.png'}")
    except Exception as e:
        print(f"[smoke] plot failed: {e}")

    print(f"\n[smoke] VERDICT: {verdict}")
    print(f"[smoke] summary: {out_dir / 'smoke_summary.json'}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chassis_labels_csv",
                   default="data/covindb_chassis_labels.csv")
    p.add_argument("--pdb_dir",
                   default=str(Path.home() / "Downloads/CovInDB2/CovInDB/PDB"))
    p.add_argument("--pretrained_ckpt",
                   default="checkpoint/gen_mol.pkl")
    p.add_argument("--out_dir", default="data/chassis_head_smoke")
    p.add_argument("--n_iters", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_chassis_classes", type=int, default=15)
    p.add_argument("--n_per_family_train", type=int, default=2)
    p.add_argument("--n_per_family_val", type=int, default=1)
    p.add_argument("--seq_len", type=int, default=80)
    p.add_argument("--pocket_radius", type=float, default=15.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--head_only", action="store_true", default=True,
                   help="Freeze everything except chassis_head.")
    p.add_argument("--full_finetune", dest="head_only", action="store_false")
    p.add_argument("--train_csv_override", default=None,
                   help="Skip per-family balanced selection and use this CSV "
                   "as the training set (must be in upper-case schema).")
    p.add_argument("--val_csv_override", default=None,
                   help="Companion to --train_csv_override.")
    args = p.parse_args()
    main(args)
