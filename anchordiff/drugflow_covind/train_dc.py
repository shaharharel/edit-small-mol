"""DrugFlow C+D finetune on CovBinder.

C-arm: 290-d covalent token (warhead Morgan FP + mechanism + residue counts
       + canonical d/θ) → small MLP → bias on pocket['one_hot']. Same adapter
       as `anchordiff/covind/cov_adapter_v2_5.py` (it's a Linear→ReLU→Linear,
       so it doesn't care whether the pocket is CrossDocked or DrugFlow-CA+).

D-arm: at training time, add Gaussian jitter (σ_d=0.1 Å) to warhead atom
       positions in the LIGAND ligand['x'] (before the flow-matching loss
       computes z_t). The model thus learns to denoise toward a *distribution*
       of warhead poses near canonical, not a delta function.
       Optionally add λ * ||pred_warhead_pos - canonical||² loss term that
       uses the model's predicted z1 positions for warhead atoms.

Adapter parameters: ~19k (290→64→13 with a learnable scale).
Backbone parameters: ~3M (DrugFlow). Adapter LR = 10×backbone LR.

Smoke test: --smoke runs 5 forward/backward steps, prints grad norms, exits.
Full run: --epochs 10 --batch_size 4 --lr 1e-5 (≈ 30 min on V100).
"""
from __future__ import annotations
import sys, argparse, time, json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# DrugFlow sys path
DRUGFLOW_ROOT = None
for cand in [Path.home() / "DrugFlow", Path("/home/shaharh_quris_ai/DrugFlow")]:
    if cand.exists():
        sys.path.insert(0, str(cand))
        DRUGFLOW_ROOT = cand
        break

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from anchordiff.covind.covalent_token_v2_5 import (
    build_token_v2_5, TOKEN_DIM_V2_5,
)
from anchordiff.covind.cov_adapter_v2_5 import (
    CovalentConditioningAdapterV25,
)
from anchordiff.covind.local_frame import compute_frame


# --------------------------------------------------------------------------
# CovBinder adapter dataset — uses DrugFlow's process_raw_pair under the hood
# --------------------------------------------------------------------------
class CovBinderDrugFlowDataset:
    """Lazy dataset: on __getitem__, loads PDB → splits ligand+pocket via
    DrugFlow's `process_raw_pair`, builds the covalent token, returns a dict
    DrugFlow's `compute_loss` understands. Skips bad rows with a counter.
    """
    def __init__(self, csv_path: str, raw_pdb_dir: str,
                 pocket_representation: str = "CA+",
                 dist_cutoff: float = 8.0,
                 max_ligand_atoms: int = 40,
                 warhead_jitter_sigma: float = 0.0):
        self.df = pd.read_csv(csv_path)
        self.raw_pdb_dir = Path(raw_pdb_dir)
        self.pocket_repr = pocket_representation
        self.cutoff = dist_cutoff
        self.max_lig = max_ligand_atoms
        self.warhead_jitter = warhead_jitter_sigma
        self.drop = {}
        # Resolve PDB paths up-front so smoke can see what exists
        self.df["resolved_pdb"] = self.df.apply(self._resolve_pdb, axis=1)
        n_have = int(self.df["resolved_pdb"].notna().sum())
        print(f"CovBinder DrugFlow dataset: {len(self.df)} rows, {n_have} with resolvable PDB")

    def _resolve_pdb(self, r):
        """CSV provides pdb_path (relative to PROJECT_ROOT) and pdb_id;
        try both."""
        pdb_id = str(r.get("pdb_id", "")).upper()
        candidates = [
            PROJECT_ROOT / str(r.get("pdb_path", "")),
            self.raw_pdb_dir / f"{pdb_id}.pdb",
            self.raw_pdb_dir / f"{pdb_id.lower()}.pdb",
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return None

    def _drop(self, why):
        self.drop[why] = self.drop.get(why, 0) + 1
        return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from Bio.PDB import PDBParser
        from rdkit import Chem
        from src.data.data_utils import process_raw_pair
        r = self.df.iloc[idx]
        pdbf = r["resolved_pdb"]
        if pdbf is None:
            return self._drop("no_pdb")
        try:
            structure = PDBParser(QUIET=True).get_structure("", pdbf)[0]
        except Exception:
            return self._drop("pdb_parse_fail")
        # Locate the ligand: HETATM residue matching ligand_chain + ligand_resi
        lig_res = None
        for chain in structure:
            if chain.id != r["ligand_chain"]:
                continue
            for res in chain:
                if res.id[1] == int(r["ligand_resi"]):
                    lig_res = res; break
            if lig_res is not None: break
        if lig_res is None:
            return self._drop("ligand_not_found")
        # Build an RDKit mol from the ligand residue (heavy atoms only)
        rd_mol = _residue_to_rdmol(lig_res)
        if rd_mol is None:
            return self._drop("rdkit_from_residue_fail")
        if rd_mol.GetNumAtoms() > self.max_lig:
            return self._drop(f"too_large_>{self.max_lig}")
        # Reduce to a single-model biopython object
        try:
            ligand, pocket = process_raw_pair(
                structure, rd_mol,
                dist_cutoff=self.cutoff,
                pocket_representation=self.pocket_repr,
                compute_nerf_params=True,
                nma_input=None,
            )
        except Exception as e:
            return self._drop(f"process_raw_pair_{type(e).__name__}")
        if ligand["x"].numel() == 0 or pocket["x"].numel() == 0:
            return self._drop("empty_after_process")
        # Build C-arm token
        ctx_res = [a.get_parent().get_resname() for a in structure.get_atoms()
                   if a.get_id() == "CA"]
        token = build_token_v2_5(
            d_canonical=float(r.get("warhead_canonical_d", 1.85)),
            theta_canonical=float(r.get("warhead_canonical_angle", 107.0)),
            warhead_class=str(r.get("warhead_class", "Michael Acceptor")),
            cys_context_residues=ctx_res,
            reaction_mechanism=str(r.get("reaction_type", "Michael Addition")),
        )
        # Compute canonical warhead position in GLOBAL frame (for D-arm)
        try:
            cys = structure[r["cys_chain"]][int(r["cys_resi"])]
            sg = np.array(cys["SG"].get_coord(), dtype=float)
            cb = np.array(cys["CB"].get_coord(), dtype=float)
            ca = np.array(cys["CA"].get_coord(), dtype=float)
            R, t = compute_frame(sg, cb, ca)
            canonical_global = t + R @ np.array([0.0, 0.0, 1.85])  # Cβ canonical
        except Exception:
            canonical_global = None
        # Warhead jitter on the warhead atom (anchor_atom_idx_in_ligand)
        warhead_idx = int(r.get("anchor_atom_idx_in_ligand", 0))
        if self.warhead_jitter > 0.0 and warhead_idx < ligand["x"].shape[0]:
            noise = torch.randn(3) * self.warhead_jitter
            ligand["x"][warhead_idx] = ligand["x"][warhead_idx] + noise
        ligand["name"] = str(r.get("record_id", f"row_{idx}"))
        # Per-sample 'size' fields (collate expects them)
        ligand["size"] = torch.tensor([ligand["x"].shape[0]])
        pocket["size"] = torch.tensor([pocket["x"].shape[0]])
        ligand["n_bonds"] = torch.tensor([ligand["bond_one_hot"].shape[0]])
        pocket["n_bonds"] = torch.tensor([pocket["bond_one_hot"].shape[0]])
        return {
            "ligand": ligand,
            "pocket": pocket,
            "cov_token": torch.from_numpy(token).float(),
            "warhead_atom_idx": warhead_idx,
            "canonical_global": (torch.from_numpy(canonical_global).float()
                                 if canonical_global is not None else None),
        }


def _residue_to_rdmol(residue):
    """Build an RDKit mol from a biopython residue (heavy atoms, no bond info).
    DrugFlow's `prepare_ligand` infers bonds from coords + RDKit valence —
    we just need a flat connection-table-free atom block.

    Returns None on failure.
    """
    from rdkit import Chem
    try:
        block_lines = []
        atoms = [a for a in residue.get_atoms() if a.element != "H"]
        if not atoms:
            return None
        block_lines.append("")  # title
        block_lines.append("  CovBinder")  # producer
        block_lines.append("")
        block_lines.append(f"{len(atoms):>3}  0  0  0  0  0  0  0  0  0999 V2000")
        for a in atoms:
            x, y, z = a.get_coord()
            sym = a.element if a.element != " " else "C"
            block_lines.append(
                f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3}0  0  0  0  0  0  0  0  0  0  0  0"
            )
        block_lines.append("M  END")
        block = "\n".join(block_lines)
        mol = Chem.MolFromMolBlock(block, removeHs=True, sanitize=False)
        if mol is None:
            return None
        # Don't sanitize — covalent ligands have weird valences
        return mol
    except Exception:
        return None


# --------------------------------------------------------------------------
# Forward + loss with C-arm injection and D-arm pose loss
# --------------------------------------------------------------------------
def forward_one(model, adapter, batch, device,
                d_arm_lambda: float = 0.0,
                disable_adapter: bool = False):
    """Run model.compute_loss on a single example, with C-arm bias on
    pocket['one_hot'] and an optional D-arm pose penalty.

    Returns (loss, info_dict).
    """
    from src.data.data_utils import TensorDict
    from src.data.data_utils import Residues
    ligand = TensorDict(**batch["ligand"]).to(device)
    pocket_raw = TensorDict(**batch["pocket"]).to(device)
    pocket_oh_orig = pocket_raw["one_hot"].float()
    # C-arm: bias pocket one_hot
    cov_token = batch["cov_token"].to(device)
    # Adapter outputs a feat_dim=20 bias for the CA-aa one-hot space (matches
    # DrugFlow's aa_decoder, len 20). We instantiate adapter with the right
    # feat_dim externally.
    if disable_adapter:
        # Ablation C: D-arm only — adapter is still instantiated (and gets
        # zero grad) but its output is multiplied by zero so it can't
        # influence pocket features. We still call adapter() so its params
        # appear in the graph (avoids opt.step() errors on missing grads).
        bias = adapter(cov_token) * 0.0
    else:
        bias = adapter(cov_token)            # (20,)
    pocket_oh_aug = pocket_oh_orig + bias.unsqueeze(0)
    pocket_raw["one_hot"] = pocket_oh_aug
    pocket = pocket_raw  # process_raw_pair already wraps as a dict-like
    # Per-sample sizes / masks needed by compute_loss
    if "mask" not in ligand:
        ligand["mask"] = torch.zeros(ligand["x"].shape[0], device=device, dtype=torch.long)
    if "mask" not in pocket:
        pocket["mask"] = torch.zeros(pocket["x"].shape[0], device=device, dtype=torch.long)
    if "bond_mask" not in ligand:
        ligand["bond_mask"] = torch.zeros(ligand["bond_one_hot"].shape[0],
                                          device=device, dtype=torch.long)
    if "bond_mask" not in pocket:
        pocket["bond_mask"] = torch.zeros(pocket["bond_one_hot"].shape[0],
                                          device=device, dtype=torch.long)
    if "size" not in ligand or ligand["size"].dim() == 0:
        ligand["size"] = torch.tensor([ligand["x"].shape[0]], device=device)
    if "size" not in pocket or pocket["size"].dim() == 0:
        pocket["size"] = torch.tensor([pocket["x"].shape[0]], device=device)

    loss, info = model.compute_loss(ligand, pocket, return_info=True)
    # D-arm soft pose loss: we don't have access to predicted positions
    # without re-running the dynamics call; defer (set d_arm_lambda=0 for
    # smoke test). The training-time data-side jitter is the primary D-arm.
    info_out = dict(info)
    info_out["loss_total"] = float(loss)
    return loss, info_out


def smoke_test(args):
    """Run 5 forward-backward steps. Print drop counts, grad norms, losses.
    Exit early on any failure mode."""
    print("=== SMOKE TEST ===")
    if DRUGFLOW_ROOT is None:
        print("DrugFlow repo not found"); return False
    from src.model.lightning import DrugFlow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    print(f"  loading DrugFlow ckpt: {args.checkpoint}")
    model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
    if args.datadir:
        model.datadir = Path(args.datadir)
    model.setup(stage="generation")  # avoid downloading datasets we don't have
    model.train().to(device)

    # Adapter: feat_dim must match pocket one-hot dim (20 = len(aa_decoder))
    feat_dim = model.module_h.dim  # not the right dim; for pocket CA+ it's len(aa_encoder)=20
    # Look at one batch to read actual pocket one_hot dim
    ds = CovBinderDrugFlowDataset(
        csv_path=args.csv,
        raw_pdb_dir=args.raw_pdb_dir,
        pocket_representation=model.pocket_representation,
        warhead_jitter_sigma=args.jitter_sigma,
    )
    ex = None
    for i in range(min(50, len(ds))):
        ex = ds[i]
        if ex is not None:
            break
    if ex is None:
        print(f"  SMOKE FAIL: 0/50 examples loaded. drop_counts: {ds.drop}")
        return False
    pocket_oh_dim = ex["pocket"]["one_hot"].shape[-1]
    print(f"  pocket one_hot dim: {pocket_oh_dim}")
    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=pocket_oh_dim, hidden_dim=64,
        init_scale=0.5, token_dropout_p=0.1,
    ).to(device)
    print(f"  adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    opt = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": adapter.parameters(), "lr": args.lr * 10.0},
        ],
        weight_decay=1e-5,
    )

    n_steps_done = 0
    n_skip = 0
    for i in range(min(50, len(ds))):
        if n_steps_done >= args.smoke_steps:
            break
        ex = ds[i]
        if ex is None:
            n_skip += 1; continue
        try:
            loss, info = forward_one(model, adapter, ex, device, d_arm_lambda=0.0)
            opt.zero_grad()
            loss.backward()
            adapter_grad = sum(p.grad.norm().item()**2 for p in adapter.parameters()
                               if p.grad is not None)**0.5
            model_grad = sum(p.grad.norm().item()**2 for p in model.parameters()
                             if p.grad is not None and p.grad.numel() > 0)**0.5
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), 10.0)
            opt.step()
            n_steps_done += 1
            print(f"  step {n_steps_done}: loss={float(loss):.4f}  "
                  f"adapter_grad={adapter_grad:.3e}  model_grad={model_grad:.3e}")
            if not (adapter_grad > 1e-12 and model_grad > 1e-8):
                print(f"  WARN: dead gradient path")
        except Exception as e:
            n_skip += 1
            import traceback
            print(f"  step error ({type(e).__name__}): {e}")
            traceback.print_exc()
    print(f"  smoke done: {n_steps_done}/{args.smoke_steps} steps  drops: {ds.drop}")
    return n_steps_done >= 3 and n_steps_done > n_skip


def run_train(args):
    """Full finetune."""
    if DRUGFLOW_ROOT is None:
        raise RuntimeError("DrugFlow repo not found")
    from src.model.lightning import DrugFlow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAIN  device={device}  epochs={args.epochs}  lr={args.lr} ===")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir: {out_dir}")

    model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
    if args.datadir:
        model.datadir = Path(args.datadir)
    model.setup(stage="generation")
    model.train().to(device)

    ds = CovBinderDrugFlowDataset(
        csv_path=args.csv,
        raw_pdb_dir=args.raw_pdb_dir,
        pocket_representation=model.pocket_representation,
        warhead_jitter_sigma=args.jitter_sigma,
    )
    # Sniff pocket one-hot dim
    ex0 = None
    for i in range(min(50, len(ds))):
        ex0 = ds[i]
        if ex0 is not None:
            break
    if ex0 is None:
        raise RuntimeError(f"0/50 examples loaded: drops={ds.drop}")
    pocket_oh_dim = ex0["pocket"]["one_hot"].shape[-1]
    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=pocket_oh_dim, hidden_dim=64,
        init_scale=0.5, token_dropout_p=0.1,
    ).to(device)

    # Optionally freeze the bond/edge decoder — this was identified as the
    # source of the dust-mol collapse (edge bias shifts toward NONE class).
    n_frozen = 0
    if args.freeze_bond_head:
        for n, p in model.named_parameters():
            if "edge_decoder" in n or "edge_out" in n:
                p.requires_grad = False
                n_frozen += 1
        print(f"[ablation F] froze {n_frozen} edge-decoder parameters")

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": adapter.parameters(), "lr": args.lr * 10.0},
        ],
        weight_decay=1e-5,
    )

    log_path = out_dir / "train.log"
    flog = open(log_path, "a", buffering=1)
    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True); flog.write(line + "\n")
    log(f"START epochs={args.epochs} lr={args.lr} bs={args.batch_size} jitter={args.jitter_sigma}")

    t0 = time.time()
    step = 0
    best_train = float("inf")
    for epoch in range(args.epochs):
        running = 0.0; n_ok = 0; n_skip = 0; accum = 0
        opt.zero_grad()
        for i in range(len(ds)):
            ex = ds[i]
            if ex is None:
                n_skip += 1; continue
            try:
                loss, info = forward_one(model, adapter, ex, device,
                                         d_arm_lambda=args.d_arm_lambda,
                                         disable_adapter=args.disable_adapter)
                if not torch.isfinite(loss):
                    n_skip += 1
                    continue
                # Cap loss magnitude — pathological examples produce 1e6 spikes
                # that propagate to NaN through grad clipping interaction
                if float(loss) > 50.0:
                    n_skip += 1
                    continue
                (loss / args.batch_size).backward()
                running += float(loss); n_ok += 1; accum += 1
                if accum >= args.batch_size:
                    # Check for NaN grads before stepping
                    has_nan_grad = False
                    for p in list(model.parameters()) + list(adapter.parameters()):
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_nan_grad = True
                            break
                    if has_nan_grad:
                        opt.zero_grad(); accum = 0; n_skip += 1
                        continue
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(adapter.parameters()), 1.0)
                    opt.step(); opt.zero_grad(); accum = 0
                    step += 1
                    if step % 25 == 0:
                        elapsed = time.time() - t0
                        log(f"epoch {epoch+1}  step {step}  loss={running/max(n_ok,1):.4f}  "
                            f"rate={step/max(elapsed,1):.2f} steps/s  n_skip={n_skip}")
            except Exception as e:
                n_skip += 1
                if n_skip < 5:
                    log(f"  step error ({type(e).__name__}): {str(e)[:200]}")
                opt.zero_grad(); accum = 0
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), 10.0)
            opt.step(); opt.zero_grad()
        train_loss = running / max(n_ok, 1)
        log(f"EPOCH {epoch+1} done: train_loss={train_loss:.4f} n_ok={n_ok} n_skip={n_skip}")
        ckpt = out_dir / f"epoch_{epoch+1:02d}.pt"
        torch.save({
            "epoch": epoch + 1, "step": step,
            "model_state": model.state_dict(),
            "adapter_state": adapter.state_dict(),
            "args": vars(args),
            "train_loss": train_loss,
        }, ckpt)
        log(f"  saved {ckpt.name}")
        if train_loss < best_train:
            best_train = train_loss
            torch.save({
                "epoch": epoch + 1, "step": step,
                "model_state": model.state_dict(),
                "adapter_state": adapter.state_dict(),
                "args": vars(args),
                "train_loss": train_loss,
            }, out_dir / "best.pt")
            log(f"  saved best.pt (loss={train_loss:.4f})")
    final = out_dir / "final.pt"
    torch.save({"model_state": model.state_dict(),
                "adapter_state": adapter.state_dict(),
                "args": vars(args), "best_train": best_train}, final)
    log(f"DONE  total_time={time.time()-t0:.0f}s  best_train={best_train:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(Path.home()/ "DrugFlow"/"checkpoints"/"drugflow.ckpt"))
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "data/covbinder/covind_training_set.csv"))
    ap.add_argument("--raw_pdb_dir", default=str(PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB"))
    ap.add_argument("--out_dir", default=str(Path.home() / "runs/drugflow_dc"))
    ap.add_argument("--datadir", default=str(Path.home() / "DrugFlow" / "src" / "default"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4, help="grad accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--jitter_sigma", type=float, default=0.1)
    ap.add_argument("--no_jitter", action="store_true",
                    help="Ablation B: set jitter_sigma=0 (overrides --jitter_sigma)")
    ap.add_argument("--disable_adapter", action="store_true",
                    help="Ablation C: zero out adapter output (D-arm only)")
    ap.add_argument("--freeze_bond_head", action="store_true",
                    help="Ablation F: freeze edge_decoder (bond head) — hypothesized fix")
    ap.add_argument("--d_arm_lambda", type=float, default=0.0,
                    help="weight on the predicted-warhead-pos penalty (deferred)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke_steps", type=int, default=5)
    args = ap.parse_args()

    if args.no_jitter:
        args.jitter_sigma = 0.0
        print(f"[ablation B] --no_jitter: jitter_sigma set to 0.0")
    if args.disable_adapter:
        print(f"[ablation C] --disable_adapter: adapter output zeroed (D-arm only)")

    if args.smoke:
        ok = smoke_test(args)
        print(f"SMOKE STATUS: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    run_train(args)


if __name__ == "__main__":
    main()
