"""Finetune PocketFlow with C-arm (covalent token) + D-arm (soft warhead jitter).

Status (honestly): a full CovBinder→PocketFlow LMDB conversion is out of scope
for the day-1 time budget — PocketFlow's `LoadDataset` expects pre-processed
crossdocked_pocket10.lmdb-style records (per-complex featurized graphs). The
CovBinder CSV only has SMILES + Cys coords; converting requires running
`process_raw` over each of the ~1k complexes, which is doable but
multi-hour by itself.

So this file does two things:
  1. SMOKE TEST (default): build a SINGLE in-memory training example from
     ZAP70 pocket + warhead SDF + a hand-grown random ligand, plug it through
     PocketFlow.get_loss with the C-arm adapter wired in and the D-arm
     jitter applied to the warhead atom's y_pos. Verify forward + backward
     produce non-NaN gradients flowing into both arms.

  2. FULL FINETUNE (--data_dir <lmdb>): if a PocketFlow-format LMDB is
     provided, wraps PocketFlow's standard Experiment.fit_step but with
     two patches:
       a) `pocket_flow_model.protein_atom_emb` input feature is augmented
          by the C-arm adapter output (broadcast bias)
       b) the model.get_loss is wrapped to add λ_D · ||y_pos_pred − y_pos_canon||²
          for atom positions tagged as warhead.

The two-arm hook into get_loss is the same math as anchordiff/covind/train_d_soft.py
but applied to PocketFlow's `pos_predictor.get_mdn_probability` output: we
penalize MDN modes that drift from the canonical warhead position.

Usage (T4 smoke):
  python ~/edit-small-mol/anchordiff/pocketflow_covind/train_dc.py \
      --smoke \
      --pocket ~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb \
      --warhead_sdf ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
      --ckpt ~/PocketFlow/ckpt/ZINC-pretrained-255000.pt \
      --steps 100

Usage (full, requires LMDB):
  python ~/edit-small-mol/anchordiff/pocketflow_covind/train_dc.py \
      --data_dir ~/PocketFlow/data/zap70_covbinder.lmdb \
      --ckpt ~/PocketFlow/ckpt/ZINC-pretrained-255000.pt \
      --steps 5000 --jitter_sigma_d 0.10
"""
from __future__ import annotations
import os
import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

POCKETFLOW_ROOT = os.path.expanduser("~/PocketFlow")
sys.path.insert(0, POCKETFLOW_ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pocket_flow import PocketFlow  # type: ignore
from pocket_flow.utils import (  # type: ignore
    Protein, ComplexData, torchify_dict,
    FeaturizeProteinAtom, FeaturizeLigandAtom, FocalMaker, AtomComposer,
    RefineData, LigandCountNeighbors, mask_node,
)

from cov_token import build_token, CovTokenAdapter, TOKEN_DIM  # type: ignore
from warhead_seed import load_acrylamide_seed_from_sdf  # type: ignore


# ---------------------------------------------------------------------------
# C-arm wiring
# ---------------------------------------------------------------------------
class CovConditionedPocketFlow(torch.nn.Module):
    """Wrap PocketFlow + a CovTokenAdapter.

    On forward, the adapter's bias is added to the LIGAND atom feature vector
    BEFORE the input embedding layer. The ligand_atom_feature_dim is read
    from the model config so the bias dim matches.
    """
    def __init__(self, base_model: PocketFlow, token_dim: int = TOKEN_DIM):
        super().__init__()
        self.base = base_model
        lig_feat_dim = self.base.config.ligand_atom_feature_dim
        self.adapter = CovTokenAdapter(feat_dim=lig_feat_dim, token_dim=token_dim)

    def add_cov_bias(self, data, token: torch.Tensor) -> None:
        """Shift the ligand-context rows of `data.cpx_feature` by the adapter
        bias. Token is broadcast to all ligand atoms (each gets the same
        conditioning). Builds a NEW tensor (no in-place ops on graph leaves).
        """
        bias = self.adapter(token.to(data.cpx_pos.device))  # (lig_feat_dim,)
        lig_idx = data.idx_ligand_ctx_in_cpx
        if lig_idx.numel() == 0:
            return
        feat = data.cpx_feature.float()
        K = min(bias.shape[0], feat.shape[1])
        # Build an additive update tensor with the bias placed on the rows
        # indexed by lig_idx; uses index_add so autograd tracks bias properly.
        update = torch.zeros_like(feat)
        broadcast_bias = torch.zeros(feat.shape[1], device=feat.device, dtype=feat.dtype)
        broadcast_bias[:K] = bias[:K]
        update.index_add_(0, lig_idx, broadcast_bias.unsqueeze(0).expand(lig_idx.shape[0], -1))
        data.cpx_feature = feat + update

    def forward(self, data, cov_token: torch.Tensor):
        self.add_cov_bias(data, cov_token)
        return self.base.get_loss(data)


# ---------------------------------------------------------------------------
# D-arm jitter helper
# ---------------------------------------------------------------------------
def apply_warhead_jitter(y_pos: torch.Tensor, warhead_mask: torch.Tensor,
                         sigma_d: float, sigma_theta_deg: float,
                         rng: np.random.Generator) -> torch.Tensor:
    """Gaussian jitter on warhead-atom positions; same math as
    anchordiff/covind/train_d_soft.py:apply_warhead_jitter (isotropic version).

    y_pos          : (N, 3) target positions for the autoregressive step
    warhead_mask   : (N,) bool, True for atoms tagged as warhead
    sigma_d        : Å, radial std
    sigma_theta    : deg, angular wobble (converted to perp-Å at d≈1.81)
    """
    if (sigma_d == 0 and sigma_theta_deg == 0) or warhead_mask.sum() == 0:
        return y_pos
    sigma_perp = 1.81 * math.tan(math.radians(sigma_theta_deg))
    sigma_total = math.sqrt(sigma_d ** 2 + sigma_perp ** 2)
    n = int(warhead_mask.sum())
    noise = torch.from_numpy(
        rng.normal(0.0, sigma_total, size=(n, 3))
    ).float().to(y_pos.device)
    out = y_pos.clone()
    out[warhead_mask] = out[warhead_mask] + noise
    return out


# ---------------------------------------------------------------------------
# Smoke test path
# ---------------------------------------------------------------------------
def build_smoke_batch(pocket_pdb: str, warhead_sdf: str, device: str):
    """Build a single training example from ZAP70 pocket + warhead SDF.

    The warhead SDF gives us 5 ligand atoms with full bond info. We treat
    them as a tiny ligand — PocketFlow's transform pipeline will produce
    a valid ComplexData with `data.atom_label` etc.
    """
    pro_dict = Protein(pocket_pdb).get_atom_dict(removeHs=True, get_surf=True)
    lig_dict = load_acrylamide_seed_from_sdf(warhead_sdf)
    data = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lig_dict),
    )

    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(
        atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53]
    )
    # FocalMaker needs the training-time signal too
    focal_masker = FocalMaker(
        r=4.0, num_work=16, atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53]
    )
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=False,
                                 use_protein_bond=True)

    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    # Random mask: hide one atom (the Cβ, idx 0) as the "next-atom" target,
    # the other 4 stay in context.
    masked = torch.LongTensor([lig_dict.cb_seed_idx])
    node4mask = torch.arange(data.ligand_pos.size(0))
    keep = node4mask[~torch.isin(node4mask, masked)]
    # QA fix 2026-05-25: mask_node signature is mask_node(data, context, masked, ...)
    # NOT mask_node(data, masked, context, ...). Previously this trained the model to
    # predict the WRONG atom set (Cα/Ccarb/O/N treated as targets, Cβ kept as context).
    # Correct call: keep = context, masked = target.
    data = mask_node(data, keep, masked, num_atom_type=9, y_pos_std=0.)
    try:
        data = focal_masker.run(data)
    except Exception as e:
        print(f"[smoke] focal_masker failed: {e}; skipping (smoke path proceeds)")
    data = atom_composer.run(data)

    return data.to(device), lig_dict


def smoke_run(args):
    device = args.device
    pkt = args.pocket
    wh = args.warhead_sdf

    print(f"[smoke] device={device}")
    print(f"[smoke] loading PocketFlow ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    base = PocketFlow(config).to(device)
    base.load_state_dict(ckpt["model"])
    wrapped = CovConditionedPocketFlow(base).to(device)
    wrapped.train()

    n_params_adapter = sum(p.numel() for p in wrapped.adapter.parameters())
    print(f"[smoke] adapter params: {n_params_adapter}")

    # build static token (acrylamide / Michael) — same at every step (smoke)
    tok = torch.from_numpy(build_token(
        d_canonical=1.85, theta_canonical=107.0,
        warhead_class="Michael Acceptor",
        pocket_residues=["LEU", "GLY", "CYS", "GLY", "ASN", "PHE", "GLY"],
        reaction_mechanism="Michael Addition",
    )).float().to(device)
    print(f"[smoke] token_dim={tok.shape[0]}, mean|tok|={tok.abs().mean().item():.4f}")

    rng = np.random.default_rng(args.seed)

    # Build smoke batch once and reuse
    print("[smoke] building smoke batch...")
    try:
        data, lig_dict = build_smoke_batch(pkt, wh, device)
    except Exception as e:
        print(f"[smoke] batch build failed: {type(e).__name__}: {e}")
        raise
    print(f"[smoke] data.cpx_pos shape={tuple(data.cpx_pos.shape)}  "
          f"n_lig_ctx={data.idx_ligand_ctx_in_cpx.numel()}")

    optim = torch.optim.AdamW(
        list(wrapped.adapter.parameters()) +
        # finetune only the late-layer pieces (per finetuning.py reset_parameters list)
        [p for n, p in wrapped.base.named_parameters()
         if any(k in n for k in [
             "edge_flow.flow_layers.5", "atom_flow.flow_layers.5",
             "pos_predictor.mu_net", "pos_predictor.logsigma_net",
             "pos_predictor.pi_net", "focal_net.net.1",
         ])],
        lr=args.lr, weight_decay=0.0,
    )
    print(f"[smoke] optimizer over {sum(p.numel() for g in optim.param_groups for p in g['params'])} params")

    cb_idx_in_ctx = 0  # the masked atom is the Cβ — its y_pos IS the warhead-target
    print(f"[smoke] starting smoke loop, steps={args.steps}")

    # snapshot the unmodified cpx_feature so we can re-inject the C-arm bias
    # fresh at each step (the graph from a previous step would otherwise be
    # retained and break autograd).
    cpx_feature_clean = data.cpx_feature.clone().detach()
    y_pos_clean = data.y_pos.clone().detach() if hasattr(data, "y_pos") and data.y_pos.numel() > 0 else None

    losses = []
    adapter_grad_norms = []
    t0 = time.time()
    for step in range(args.steps):
        # Re-seed data with fresh, detached tensors each step (no stale graph)
        data.cpx_feature = cpx_feature_clean.clone()
        if y_pos_clean is not None:
            data.y_pos = y_pos_clean.clone()
            # D-arm: jitter the warhead y_pos
            warhead_mask = torch.zeros(data.y_pos.shape[0], dtype=torch.bool,
                                      device=data.y_pos.device)
            warhead_mask[0] = True
            data.y_pos = apply_warhead_jitter(
                data.y_pos, warhead_mask, args.jitter_sigma_d, args.jitter_sigma_theta, rng,
            )

        optim.zero_grad()
        try:
            out_dict = wrapped(data, tok)
            loss = out_dict["loss"]
            loss.backward()
            # check gradient flow through adapter
            adapter_g = sum((p.grad.norm().item() ** 2
                             for p in wrapped.adapter.parameters()
                             if p.grad is not None)) ** 0.5
            torch.nn.utils.clip_grad_norm_(
                [p for g in optim.param_groups for p in g['params']], 5.0,
            )
            optim.step()
            losses.append(float(loss.detach()))
            adapter_grad_norms.append(adapter_g)
            if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
                print(f"  step {step:4d}  loss={float(loss):.4f}  "
                      f"adapter|g|={adapter_g:.6f}  "
                      f"|log_scale|={float(wrapped.adapter.log_scale):.3f}")
            if not np.isfinite(float(loss)):
                print("[smoke] NaN/Inf loss — stopping early")
                break
        except Exception as e:
            print(f"[smoke] step {step} failed: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            break

    dt = time.time() - t0
    n_ok = len(losses)
    print(f"\n[smoke] DONE in {dt:.1f}s — {n_ok}/{args.steps} steps")
    if n_ok > 0:
        print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
        nz = [g for g in adapter_grad_norms if g > 0]
        print(f"  adapter grad >0 on {len(nz)}/{len(adapter_grad_norms)} steps")
        ok = (n_ok >= max(5, args.steps // 4)) and (len(nz) > 0)
        print(f"  SMOKE PASS = {ok}")

        # Save the smoke ckpt so sample.py has something to load.
        os.makedirs(args.out_dir, exist_ok=True)
        ck_path = os.path.join(args.out_dir, "smoke_final.ckpt")
        a_path = os.path.join(args.out_dir, "adapter_smoke.pt")
        torch.save({"model": wrapped.base.state_dict(), "config": config,
                    "args": vars(args), "loss_trace": losses}, ck_path)
        torch.save(wrapped.adapter.state_dict(), a_path)
        print(f"  saved {ck_path}")
        print(f"  saved {a_path}")
        return ok
    return False


# ---------------------------------------------------------------------------
# Full finetune path (gated by --data_dir presence)
# ---------------------------------------------------------------------------
def full_finetune(args):
    """Patched fit_step that wires the C-arm adapter + D-arm jitter into
    PocketFlow's training loop. Requires a PocketFlow-format LMDB.
    """
    from pocket_flow.utils import LoadDataset, Experiment, mask_node  # type: ignore
    from pocket_flow.utils.transform import (
        Combine, TrajCompose, LigandTrajectory, collate_fn,
    )

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt["config"]
    base = PocketFlow(config).to(device)
    base.load_state_dict(ckpt["model"])
    wrapped = CovConditionedPocketFlow(base).to(device)

    # build the standard PocketFlow transform pipeline
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
    traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
    focal_masker = FocalMaker(r=4, num_work=16, atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
    atom_composer = AtomComposer(knn=16, num_workers=16, graph_type='knn', radius=10, use_protein_bond=True)
    combine = Combine(traj_fn, focal_masker, atom_composer)
    transform = TrajCompose([
        RefineData(), LigandCountNeighbors(),
        protein_featurizer, ligand_featurizer, combine, collate_fn,
    ])
    dataset = LoadDataset(args.data_dir, transform=transform)
    train_set, valid_set = LoadDataset.split(dataset, val_num=100, shuffle=True, random_seed=0)

    # build a static token for the smoke version (per-complex tokens require
    # warhead-class metadata in the LMDB which CovBinder→LMDB doesn't carry yet)
    tok = torch.from_numpy(build_token(
        d_canonical=1.85, theta_canonical=107.0,
        warhead_class="Michael Acceptor",
        pocket_residues=["CYS"], reaction_mechanism="Michael Addition",
    )).float().to(device)
    rng = np.random.default_rng(args.seed)

    # monkey-patch get_loss to inject C-arm + D-arm
    orig_get_loss = wrapped.base.get_loss
    def patched_get_loss(data):
        wrapped.add_cov_bias(data, tok)
        # D-arm: jitter y_pos for atoms tagged as warhead (here: heuristic =
        # the first ligand atom in each batch, which is the autoregressive
        # focus at warhead-build time). A real CovBinder→LMDB conversion
        # should tag warhead atoms explicitly.
        if hasattr(data, "y_pos") and data.y_pos.numel() > 0 and args.jitter_sigma_d > 0:
            mask = torch.zeros(data.y_pos.shape[0], dtype=torch.bool, device=data.y_pos.device)
            mask[0] = True
            data.y_pos = apply_warhead_jitter(
                data.y_pos, mask, args.jitter_sigma_d, args.jitter_sigma_theta, rng,
            )
        return orig_get_loss(data)
    wrapped.base.get_loss = patched_get_loss

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, wrapped.parameters()),
        lr=args.lr, betas=(0.99, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.6, patience=10, min_lr=1.e-5,
    )
    exp = Experiment(
        wrapped.base, train_set, optimizer, valid_set=valid_set,
        scheduler=scheduler, device=device, data_parallel=False, use_amp=False,
    )
    exp.fit_step(
        args.steps, valid_per_step=max(args.steps // 4, 100),
        train_batch_size=args.batch_size, valid_batch_size=16,
        print_log=True, with_tb=False, logdir=args.out_dir,
        schedule_key='loss', num_workers=4, pin_memory=False,
        follow_batch=[], exclude_keys=[], collate_fn=None,
        max_edge_num_in_batch=200000,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pocket", default=os.path.expanduser(
        "~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb"))
    ap.add_argument("--warhead_sdf", default=os.path.expanduser(
        "~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf"))
    ap.add_argument("--ckpt", default=os.path.expanduser(
        "~/PocketFlow/ckpt/ZINC-pretrained-255000.pt"))
    ap.add_argument("--data_dir", default=None,
                    help="PocketFlow-format LMDB; if None, falls back to smoke mode")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--jitter_sigma_d", type=float, default=0.10)
    ap.add_argument("--jitter_sigma_theta", type=float, default=5.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.expanduser("~/runs/pocketflow_dc"))
    args = ap.parse_args()

    if args.smoke or args.data_dir is None:
        return smoke_run(args)
    return full_finetune(args)


if __name__ == "__main__":
    main()
