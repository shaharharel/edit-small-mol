"""DrugFlow C+D finetune on CovBinder — v2 with parent-SMILES bond reconstruction.

Diff vs train_dc.py
-------------------
1. `_residue_to_rdmol` is replaced by `_residue_to_rdmol_from_template`, which:
     - Parses the parent SMILES from `covind_training_set.csv`
     - Builds a flat atom block (positions + elements, NO bonds) from the PDB residue
       — same as the old code
     - Uses RDKit `AllChem.AssignBondOrdersFromTemplate` to lift the template's
       bond graph onto the 3D atom block. This propagates the substructure
       atom-index permutation back so `anchor_atom_idx_in_ligand` is remapped.
2. Anything that fails template-matching (SMILES has different #heavy atoms
   than the PDB residue, or no substructure match) FALLS BACK to old
   coord-only mode (still inserted into the dataset, but logged separately).
3. Per-batch counter for `bond_recon_ok` / `bond_recon_fail` written to log.
4. Adds `--no_jitter` and `--freeze_bond_head` are inherited from train_dc.py.

Rationale (from ablation report)
-------------------------------
With train_dc.py the edge-decoder bias drifts toward class 0 (NOBOND) because
the training target — `ligand['bond_one_hot']` from `prepare_ligand` — sees
RDKit's coord+valence-only bond inference produce sparse/wrong bonds on
covalent inhibitors. By assigning a known correct bond template, the bond
classes become a balanced distribution of SINGLE/DOUBLE/AROMATIC etc., so
NOBOND is no longer the dominant class.

CLI args (extras vs train_dc.py)
--------------------------------
--no_smiles_bonds : disable bond reconstruction (revert to train_dc.py behavior)
                    — useful for A/B ablation
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
# Bond reconstruction from parent SMILES
# --------------------------------------------------------------------------
def _build_3d_mol_no_bonds(residue):
    """V2000 mol block from biopython residue: positions + element symbols,
    no bonds. Returns (rd_mol, pdb_atoms_kept_in_order) or (None, None)."""
    from rdkit import Chem
    try:
        atoms = [a for a in residue.get_atoms() if a.element != "H"]
        if not atoms:
            return None, None
        block_lines = []
        block_lines.append("")  # title
        block_lines.append("  CovBinder")
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
            return None, None
        return mol, atoms
    except Exception:
        return None, None


def _smiles_to_template(smiles):
    """Parse SMILES, drop Hs, return RDKit mol with bonds — or None."""
    from rdkit import Chem
    try:
        tpl = Chem.MolFromSmiles(smiles)
        if tpl is None:
            return None
        tpl = Chem.RemoveHs(tpl)
        return tpl
    except Exception:
        return None


def _assign_bonds_from_template(raw_3d_mol, template, timeout_sec: int = 8):
    """Strategy: ALWAYS run DetermineConnectivity first (gives all-SINGLE bonds
    from coords). Then, if atom counts match, attempt AssignBondOrdersFromTemplate
    on the *connected* mol — this can now succeed because the substructure match
    sees the inferred connectivity graph.

    Returns:
      (bonded_mol, perm) — perm[bonded_i] = raw_i; or (connected_mol, None) if
      template fails (caller treats as fallback path with all-SINGLE bonds).

      Or (None, None) on connectivity failure (very rare).
    """
    import signal
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds

    class _Timeout(Exception): pass
    def _h(*a): raise _Timeout()
    old_handler = signal.signal(signal.SIGALRM, _h)
    try:
        signal.alarm(timeout_sec)
        try:
            mw = Chem.RWMol(raw_3d_mol)
            rdDetermineBonds.DetermineConnectivity(mw)
            connected = mw.GetMol()
        except Exception:
            signal.alarm(0)
            return None, None
        finally:
            signal.alarm(0)

        # If atom counts don't match the template, return connected mol
        # (all-SINGLE bonds) without trying template assign.
        if template is None or connected.GetNumAtoms() != template.GetNumAtoms():
            return connected, None  # signal "no remap, fallback bonds"

        # Try AssignBondOrdersFromTemplate on the connected mol
        signal.alarm(timeout_sec)
        try:
            bonded = AllChem.AssignBondOrdersFromTemplate(template, connected)
            if bonded is None:
                return connected, None
        except Exception:
            return connected, None
        finally:
            signal.alarm(0)

        # Recover permutation: bonded uses template atom indices, with coords
        # pulled from connected. We need raw_i -> bonded_i mapping so we can
        # remap anchor_atom_idx_in_ligand (which is in raw_3d_mol order, which
        # equals connected order — DetermineConnectivity preserves order).
        #
        # Direct GetSubstructMatch(template) fails because connected has only
        # SINGLE bonds and the template has DOUBLE/AROMATIC; substructure match
        # is bond-type-aware. Workaround: build a "skeleton" template (all
        # SINGLE bonds, no aromaticity) and match that.
        try:
            tpl_skel = Chem.RWMol(template)
            for b in tpl_skel.GetBonds():
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
            for a in tpl_skel.GetAtoms():
                a.SetIsAromatic(False)
            tpl_skel_mol = tpl_skel.GetMol()
            match = connected.GetSubstructMatch(tpl_skel_mol)
        except Exception:
            match = ()
        if not match:
            # Try the other direction
            try:
                match = template.GetSubstructMatch(connected)
            except Exception:
                match = ()
            if not match:
                return bonded, None
            # template.GetSubstructMatch(connected): match[i] = j means
            # connected_i -> template_j. perm[template_j] = raw_i (== connected_i).
            perm = [0] * len(match)
            for connected_i, template_j in enumerate(match):
                perm[template_j] = connected_i
            return bonded, perm
        # connected.GetSubstructMatch(tpl_skel): match[i] = j means
        # template_i -> connected_j. perm[bonded_i (== template_i)] = raw_i (== connected_j).
        perm = list(match)
        return bonded, perm
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _residue_to_rdmol_from_template(residue, smiles, allow_fallback=True):
    """Build RDKit mol with bond topology from parent SMILES + PDB coords.

    Strategy (decreasing preference):
      A. SMILES atom count matches PDB → DetermineConnectivity → AssignBondOrdersFromTemplate
         (full bond orders: SINGLE/DOUBLE/AROMATIC/TRIPLE)
      B. SMILES present but atom count mismatch → DetermineConnectivity only
         (all-SINGLE bonds — still vastly better than 0 bonds because the bond
         head sees a non-NOBOND signal)
      C. No SMILES → DetermineConnectivity only (same as B)
      D. DetermineConnectivity fails → return None

    Returns (mol, anchor_idx_remap, status) where:
      mol — RDKit mol with bonds
      anchor_idx_remap — function (old_idx -> new_idx) for indices in the
        PDB atom order. None if no remap needed.
      status — "bonded_full" | "bonded_single_only" | "fail"
    """
    raw_mol, _atoms = _build_3d_mol_no_bonds(residue)
    if raw_mol is None:
        return None, None, "fail"
    template = None
    if smiles is not None and not (isinstance(smiles, float) and np.isnan(smiles)):
        template = _smiles_to_template(str(smiles))
    bonded, perm = _assign_bonds_from_template(raw_mol, template)
    if bonded is None:
        return None, None, "fail"
    if perm is None:
        # Connected-only path: bond orders are all SINGLE, atom order matches
        # raw_3d_mol (no remap needed).
        return bonded, None, "bonded_single_only"
    # perm[bonded_i] = raw_i.  Build inverse: raw_to_bonded[raw_i] = bonded_i
    raw_to_bonded = {raw_i: bonded_i for bonded_i, raw_i in enumerate(perm)}

    def remap(old_idx: int) -> int:
        return raw_to_bonded.get(old_idx, 0)

    return bonded, remap, "bonded_full"


# --------------------------------------------------------------------------
# CovBinder adapter dataset — v2 with SMILES-based bond reconstruction
# --------------------------------------------------------------------------
class CovBinderDrugFlowDatasetV2:
    def __init__(self, csv_path: str, raw_pdb_dir: str,
                 pocket_representation: str = "CA+",
                 dist_cutoff: float = 8.0,
                 max_ligand_atoms: int = 40,
                 warhead_jitter_sigma: float = 0.0,
                 use_smiles_bonds: bool = True):
        self.df = pd.read_csv(csv_path)
        self.raw_pdb_dir = Path(raw_pdb_dir)
        self.pocket_repr = pocket_representation
        self.cutoff = dist_cutoff
        self.max_lig = max_ligand_atoms
        self.warhead_jitter = warhead_jitter_sigma
        self.use_smiles_bonds = use_smiles_bonds
        self.drop = {}
        # bond-reconstruction telemetry
        self.bond_status = {"bonded_full": 0, "bonded_single_only": 0, "fail": 0}
        self.df["resolved_pdb"] = self.df.apply(self._resolve_pdb, axis=1)
        n_have = int(self.df["resolved_pdb"].notna().sum())
        print(f"CovBinder DrugFlow dataset V2: {len(self.df)} rows, "
              f"{n_have} with resolvable PDB  use_smiles_bonds={use_smiles_bonds}")

    def _resolve_pdb(self, r):
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

        # Bond reconstruction
        smiles = r.get("smiles", None) if self.use_smiles_bonds else None
        rd_mol, anchor_remap_fn, bond_status = _residue_to_rdmol_from_template(
            lig_res, smiles, allow_fallback=True
        )
        self.bond_status[bond_status] = self.bond_status.get(bond_status, 0) + 1
        if rd_mol is None:
            return self._drop("rdkit_from_residue_fail")
        if rd_mol.GetNumAtoms() > self.max_lig:
            return self._drop(f"too_large_>{self.max_lig}")

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

        # Canonical warhead pos
        try:
            cys = structure[r["cys_chain"]][int(r["cys_resi"])]
            sg = np.array(cys["SG"].get_coord(), dtype=float)
            cb = np.array(cys["CB"].get_coord(), dtype=float)
            ca = np.array(cys["CA"].get_coord(), dtype=float)
            R, t = compute_frame(sg, cb, ca)
            canonical_global = t + R @ np.array([0.0, 0.0, 1.85])
        except Exception:
            canonical_global = None

        # Warhead atom: REMAP from PDB-order to bonded-mol-order
        warhead_idx_pdb = int(r.get("anchor_atom_idx_in_ligand", 0))
        if anchor_remap_fn is not None:
            warhead_idx = anchor_remap_fn(warhead_idx_pdb)
        else:
            warhead_idx = warhead_idx_pdb
        if warhead_idx >= ligand["x"].shape[0]:
            warhead_idx = 0

        if self.warhead_jitter > 0.0 and warhead_idx < ligand["x"].shape[0]:
            noise = torch.randn(3) * self.warhead_jitter
            ligand["x"][warhead_idx] = ligand["x"][warhead_idx] + noise
        ligand["name"] = str(r.get("record_id", f"row_{idx}"))
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
            "bond_status": bond_status,
        }


# --------------------------------------------------------------------------
# Forward + loss (identical to train_dc.py — re-exported here so the script
# stands alone)
# --------------------------------------------------------------------------
def forward_one(model, adapter, batch, device,
                d_arm_lambda: float = 0.0,
                disable_adapter: bool = False):
    from src.data.data_utils import TensorDict
    ligand = TensorDict(**batch["ligand"]).to(device)
    pocket_raw = TensorDict(**batch["pocket"]).to(device)
    pocket_oh_orig = pocket_raw["one_hot"].float()
    cov_token = batch["cov_token"].to(device)
    if disable_adapter:
        bias = adapter(cov_token) * 0.0
    else:
        bias = adapter(cov_token)
    pocket_raw["one_hot"] = pocket_oh_orig + bias.unsqueeze(0)
    pocket = pocket_raw
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
    info_out = dict(info)
    info_out["loss_total"] = float(loss)
    return loss, info_out


def smoke_test(args):
    print("=== SMOKE TEST (v2 SMILES-bonds) ===")
    if DRUGFLOW_ROOT is None:
        print("DrugFlow repo not found"); return False
    from src.model.lightning import DrugFlow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    print(f"  loading DrugFlow ckpt: {args.checkpoint}")
    model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
    if args.datadir:
        model.datadir = Path(args.datadir)
    model.setup(stage="generation")
    model.train().to(device)

    ds = CovBinderDrugFlowDatasetV2(
        csv_path=args.csv,
        raw_pdb_dir=args.raw_pdb_dir,
        pocket_representation=model.pocket_representation,
        warhead_jitter_sigma=args.jitter_sigma,
        use_smiles_bonds=(not args.no_smiles_bonds),
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
    print(f"  pocket one_hot dim: {pocket_oh_dim}  bond_status: {ds.bond_status}")
    adapter = CovalentConditioningAdapterV25(
        token_dim=TOKEN_DIM_V2_5, feat_dim=pocket_oh_dim, hidden_dim=64,
        init_scale=0.5, token_dropout_p=0.1,
    ).to(device)
    print(f"  adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # Replicate train-time freezing for smoke
    n_frozen = 0
    if args.freeze_bond_head:
        for n, p in model.named_parameters():
            if "edge_decoder" in n or "edge_out" in n:
                p.requires_grad = False
                n_frozen += 1
        print(f"  [freeze_bond_head] froze {n_frozen} edge-decoder parameters")

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": adapter.parameters(), "lr": args.lr * 10.0},
        ],
        weight_decay=1e-5,
    )

    # Pre-compute current edge_decoder.2.bias for drift sanity check
    bond_bias_path = None
    for n, p in model.named_parameters():
        if "edge_decoder" in n and n.endswith(".bias") and p.numel() == 5:
            bond_bias_path = n
            initial_bond_bias = p.detach().clone()
            print(f"  initial {n} = {initial_bond_bias.cpu().numpy()}")
            break

    n_steps_done = 0
    n_skip = 0
    for i in range(min(args.smoke_steps * 5, len(ds))):
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
            # Bond-head grad specifically
            bond_grad = 0.0
            for n, p in model.named_parameters():
                if ("edge_decoder" in n or "edge_out" in n) and p.grad is not None:
                    bond_grad += p.grad.norm().item()**2
            bond_grad = bond_grad**0.5

            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(adapter.parameters()) if p.requires_grad], 10.0)
            opt.step()
            n_steps_done += 1
            print(f"  step {n_steps_done}: loss={float(loss):.4f}  "
                  f"bond_status={ex['bond_status']}  "
                  f"adapter_grad={adapter_grad:.3e}  model_grad={model_grad:.3e}  "
                  f"bond_head_grad={bond_grad:.3e}")
            if args.freeze_bond_head and bond_grad > 1e-6:
                print(f"  WARN: bond head should be frozen but bond_grad={bond_grad:.3e}")
        except Exception as e:
            n_skip += 1
            import traceback
            print(f"  step error ({type(e).__name__}): {e}")
            traceback.print_exc()
    # Drift check on bond_bias
    if bond_bias_path is not None:
        for n, p in model.named_parameters():
            if n == bond_bias_path:
                drift = (p.detach() - initial_bond_bias).abs()
                print(f"  bond_bias drift after {n_steps_done} steps: max={drift.max().item():.4e}  "
                      f"argmax_class={drift.argmax().item()}  new={p.detach().cpu().numpy()}")
                break
    print(f"  smoke done: {n_steps_done}/{args.smoke_steps} steps  drops: {ds.drop}  bond_status: {ds.bond_status}")
    return n_steps_done >= 3 and n_steps_done > n_skip


def run_train(args):
    if DRUGFLOW_ROOT is None:
        raise RuntimeError("DrugFlow repo not found")
    from src.model.lightning import DrugFlow

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAIN v2  device={device}  epochs={args.epochs}  lr={args.lr} ===")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir: {out_dir}")

    model = DrugFlow.load_from_checkpoint(args.checkpoint, map_location=device, strict=False)
    if args.datadir:
        model.datadir = Path(args.datadir)
    model.setup(stage="generation")
    model.train().to(device)

    ds = CovBinderDrugFlowDatasetV2(
        csv_path=args.csv,
        raw_pdb_dir=args.raw_pdb_dir,
        pocket_representation=model.pocket_representation,
        warhead_jitter_sigma=args.jitter_sigma,
        use_smiles_bonds=(not args.no_smiles_bonds),
    )
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
    log(f"START v2 epochs={args.epochs} lr={args.lr} bs={args.batch_size} "
        f"jitter={args.jitter_sigma}  freeze_bond_head={args.freeze_bond_head}  "
        f"use_smiles_bonds={not args.no_smiles_bonds}")

    # Sanity check: track edge_decoder.2.bias evolution
    edge_bias_param = None
    for n, p in model.named_parameters():
        if "edge_decoder" in n and n.endswith(".bias") and p.numel() == 5:
            edge_bias_param = (n, p)
            log(f"  tracking {n} (initial = {p.detach().cpu().numpy()})")
            break

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
                if float(loss) > 50.0:
                    n_skip += 1
                    continue
                (loss / args.batch_size).backward()
                running += float(loss); n_ok += 1; accum += 1
                if accum >= args.batch_size:
                    has_nan_grad = False
                    for p in list(model.parameters()) + list(adapter.parameters()):
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            has_nan_grad = True
                            break
                    if has_nan_grad:
                        opt.zero_grad(); accum = 0; n_skip += 1
                        continue
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in list(model.parameters()) + list(adapter.parameters())
                         if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(); accum = 0
                    step += 1
                    if step % 25 == 0:
                        elapsed = time.time() - t0
                        msg = (f"epoch {epoch+1}  step {step}  loss={running/max(n_ok,1):.4f}  "
                               f"rate={step/max(elapsed,1):.2f} steps/s  n_skip={n_skip}  "
                               f"bond_status={ds.bond_status}")
                        if edge_bias_param is not None:
                            n, p = edge_bias_param
                            msg += f"  bond_bias={p.detach().cpu().numpy()}"
                        log(msg)
            except Exception as e:
                n_skip += 1
                if n_skip < 5:
                    log(f"  step error ({type(e).__name__}): {str(e)[:200]}")
                opt.zero_grad(); accum = 0
        if accum > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(model.parameters()) + list(adapter.parameters())
                 if p.requires_grad], 10.0)
            opt.step(); opt.zero_grad()
        train_loss = running / max(n_ok, 1)
        log(f"EPOCH {epoch+1} done: train_loss={train_loss:.4f} n_ok={n_ok} n_skip={n_skip}  "
            f"bond_status={ds.bond_status}")
        if edge_bias_param is not None:
            n, p = edge_bias_param
            log(f"  end-of-epoch bond_bias={p.detach().cpu().numpy()}")
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
    ap.add_argument("--out_dir", default=str(Path.home() / "runs/drugflow_dc_v2"))
    ap.add_argument("--datadir", default=str(Path.home() / "DrugFlow" / "src" / "default"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--jitter_sigma", type=float, default=0.0)
    ap.add_argument("--no_jitter", action="store_true")
    ap.add_argument("--disable_adapter", action="store_true")
    ap.add_argument("--freeze_bond_head", action="store_true")
    ap.add_argument("--no_smiles_bonds", action="store_true",
                    help="Disable SMILES-based bond reconstruction (A/B ablation)")
    ap.add_argument("--d_arm_lambda", type=float, default=0.0)
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
