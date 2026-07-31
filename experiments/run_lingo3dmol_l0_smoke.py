"""L0 SMOKE — Vanilla Lingo3DMol generation on ZAP70 pocket (CPU, 10 mols).

This is a CPU port of inference_avoid_clash.py, scoped to a single pocket
and a tiny gennums target so we can verify the end-to-end pipeline on a
Mac without CUDA. Outputs an SDF file of generated mols.
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path

# Install CPU shim FIRST, before any Lingo3DMol model import
ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    cand = (ORIG_CWD / p).resolve()
    if cand.exists():
        return str(cand)
    cand2 = (ROOT / p).resolve()
    if cand2.exists():
        return str(cand2)
    return str(cand)


import lingo3dmol_cpu_shim  # noqa: F401

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdGeometry
RDLogger.DisableLog("rdApp.*")

from util.fragmol_frag_zyh import FragmolUtil
from model.transformer_v1_res_mp1 import TransformerModel as TransformerModel_contact, topkp_random
from model.transformer_v1_res_fac2 import TransformerModel
from dataloader.dataloader_case_nci_res_merge import mydataset as testdataset
from torch.utils.data import DataLoader
from inference.cube_collision_check import CollisionCheck


def changepos2(mol, center):
    mol = Chem.RWMol(mol)
    conf = mol.GetConformer()
    resolution = 0.1
    length = int(24 / resolution)
    grid_c = (length - 1) / 2
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        pos1 = (np.array([pos]) - grid_c) * resolution + center
        pos1 = pos1[0]
        p = rdGeometry.Point3D(float(pos1[0]), float(pos1[1]), float(pos1[2]))
        conf.SetAtomPosition(i, p)
    for atom in list(mol.GetAtoms()):
        if atom.GetSymbol() == "*":
            mol.ReplaceAtom(atom.GetIdx(), Chem.Atom("H"))
    return mol


def bonusVSpenalty2(coc, pattern, semi_product):
    atom_pos = coc.get_atom_xyz(semi_product)
    # FIX 2026-06-01: atom_pos comes as float64 from RDKit conformer.x/y/z;
    # `pattern[float, float, float]` is undefined indexing — silently returns
    # zero/invalid on some torch backends. Cast to int64 and bounds-check.
    atom_pos_i = atom_pos.astype(np.int64)
    n = atom_pos_i.shape[0]
    in_bounds = (
        (atom_pos_i[:, 0] >= 0) & (atom_pos_i[:, 0] < pattern.shape[0]) &
        (atom_pos_i[:, 1] >= 0) & (atom_pos_i[:, 1] < pattern.shape[1]) &
        (atom_pos_i[:, 2] >= 0) & (atom_pos_i[:, 2] < pattern.shape[2])
    )
    if not in_bounds.all():
        if os.environ.get("LINGO_DEBUG_BONUS"):
            print(f"  [DBG_BONUS] OOB atoms: {n - in_bounds.sum()}/{n}  shape={pattern.shape}  pos_range x={atom_pos_i[:,0].min()}..{atom_pos_i[:,0].max()} y={atom_pos_i[:,1].min()}..{atom_pos_i[:,1].max()} z={atom_pos_i[:,2].min()}..{atom_pos_i[:,2].max()}")
        return False
    hits_per_atom = pattern[atom_pos_i[:, 0], atom_pos_i[:, 1], atom_pos_i[:, 2]]
    score = hits_per_atom.sum()
    if os.environ.get("LINGO_DEBUG_BONUS"):
        try:
            _s = score.item() if hasattr(score, 'item') else float(score)
        except Exception:
            _s = float(score)
        # find which atoms are in occupied voxels
        if hasattr(hits_per_atom, 'cpu'):
            _hpa = hits_per_atom.cpu().numpy()
        else:
            _hpa = np.array(hits_per_atom)
        miss = np.where(_hpa < 0.5)[0]
        miss_pos = [(int(i), tuple(atom_pos_i[i].tolist())) for i in miss[:6]]
        print(f"  [DBG_BONUS2] n={n} hits={_s} miss_idx_pos={miss_pos}  pos_dtype={atom_pos.dtype}")
    if hasattr(score, 'item'):
        score = score.item()
    return float(score) == n


class Args:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def go_factory(factory_args, gudiecap, gudiepos, args):
    coords, residue, mask, atom_type, center, caption, contact_idx, contact_prob1, pattern, _ = factory_args
    sample_num = args.gen_frag_set
    recon, pred_pos, cap_mask, Value_prob = caption(
        coords=coords, type=atom_type, residue=residue, src_mask=mask,
        critical_anchor=contact_prob1, contact_idx=contact_idx,
        sample_num=sample_num, isTrain=args.isTrain,
        isMultiSample=args.isMultiSample, USE_THRESHOLD=args.USE_THRESHOLD,
        isGuideSample=args.isGuideSample,
        guidepath=[np.tile(gudiecap, (args.gen_frag_set, 1)),
                   np.tile(gudiepos, (args.gen_frag_set, 1, 1))],
        start_this_step=(gudiecap > 0).sum(),
        OnceMolGen=args.OnceMolGen, frag_len=args.frag_len_add, tempture=args.tempture)
    captions = recon.cpu().data.numpy()
    pred_pos = pred_pos.cpu().data.numpy()
    Value_prob = Value_prob.cpu().data.numpy()
    center = center.cpu().data.numpy()
    captions = captions * cap_mask
    pred_pos = pred_pos * cap_mask[:, :, np.newaxis]
    Value_prob = Value_prob * cap_mask
    return captions, pred_pos, center, Value_prob


def get_partial_to_warehouse(frag_level, factory_args, partial_product, args):
    coc = factory_args[8]
    pattern = coc.maskcoc
    fragUtil = FragmolUtil()
    all_captions = np.empty((0, 100))
    all_pos = np.empty((0, 100, 3))
    all_center = np.empty((0, 3))
    value_scores, anchor_scores_list = [], []
    prod_time = 0
    uni_smi = set()

    one_line_product = []
    for frag in partial_product[0]:
        one_line_product.extend(frag)
    gudiecap = np.zeros((1, 100))
    gudiecap[:, :len(one_line_product)] = np.array(one_line_product)[np.newaxis, :] if one_line_product else 0
    gudiepos = np.zeros((1, 100, 3))
    one_line_coords = []
    for coo in partial_product[1]:
        one_line_coords.extend(coo)
    if len(one_line_coords) != 0:
        gudiepos[:, :len(one_line_coords), :] = np.array(one_line_coords)[np.newaxis, :, :]

    factory_time = args.prod_time
    anchors = factory_args[0][factory_args[-1] == 1]

    while all_captions.shape[0] < 100 and prod_time < factory_time:
        print(f"  level={frag_level} prod={prod_time} frag_num={all_captions.shape[0]}")
        captions, pred_pos, center, Value_prob = go_factory(factory_args, gudiecap, gudiepos, args)
        smiles, ss, moleculars = fragUtil.decode3d(captions, pred_pos)
        valid_index = []
        # DEBUG: log decode and bonus stats
        if os.environ.get("LINGO_DEBUG_BONUS"):
            _n_decoded = sum(1 for s in smiles if s is not None)
            _n_valid_mol = sum(1 for m in moleculars if m is not None)
            print(f"  [DBG_BONUS] decoded={_n_decoded}/{len(smiles)} valid_mol={_n_valid_mol}")
        for j, smi in enumerate(ss):
            if smiles[j] is not None and moleculars[j] is not None:
                if smiles[j] not in uni_smi:
                    score = bonusVSpenalty2(coc, pattern, moleculars[j])
                    if os.environ.get("LINGO_DEBUG_BONUS"):
                        print(f"  [DBG_BONUS] j={j} smi={smiles[j]!r} score={score}")
                    if not score:
                        continue
                    value = Value_prob[j].sum() / ((Value_prob[j] > 0).sum() + 1e-5)
                    value_scores.append(value)
                    distance = np.linalg.norm(
                        anchors.cpu().numpy()[:, np.newaxis, :] - pred_pos[j][1:(captions[j] != 0).sum()],
                        axis=-1)
                    anchor_score = (distance.min(axis=-1) < 40).sum() / max(anchors.shape[0], 1)
                    anchor_scores_list.append(anchor_score)
                    uni_smi.add(smiles[j])
                    valid_index.append(j)
        all_captions = np.append(all_captions, captions[valid_index], axis=0)
        all_pos = np.append(all_pos, pred_pos[valid_index], axis=0)
        all_center = np.append(all_center, center[valid_index], axis=0)
        prod_time += 1

    if len(value_scores) == 0:
        return [], []
    value_scores = np.array(value_scores)
    value_scores = (value_scores - value_scores.min()) / (value_scores.max() - value_scores.min() + 1e-5)
    anchor_scores_list = np.array(anchor_scores_list)
    anchor_scores_list = (anchor_scores_list - anchor_scores_list.min()) / (anchor_scores_list.max() - anchor_scores_list.min() + 1e-5)
    all_pos_indices = (value_scores + anchor_scores_list).argsort()[::-1]
    all_captions = all_captions[all_pos_indices]
    all_pos = all_pos[all_pos_indices]
    all_captions = all_captions[:max(args.gen_frag_set // 5, 1)]
    all_pos = all_pos[:max(args.gen_frag_set // 5, 1)]

    frag_new_cap, frag_new_pos = [], []
    o_index = np.where(gudiecap == 3)[-1]
    o_index = -1 if len(o_index) == 0 else o_index[-1]
    for i in range(all_captions.shape[0]):
        cap = all_captions[i]
        pos = all_pos[i]
        if not args.OnceMolGen:
            index = np.argwhere(cap == 2)
            if len(index):
                index = index[-1][-1]
            else:
                idx3 = np.argwhere(cap == 3)
                if len(idx3) == 0:
                    continue
                index = idx3[-1][-1]
        else:
            try:
                index = np.argwhere(cap == 2)[-1][-1]
            except Exception:
                continue
        frag_new_cap.append(cap[o_index + 1:index + 1])
        frag_new_pos.append(pos[o_index + 1:index + 1, :])
    return frag_new_cap, frag_new_pos


def write_product(partial_product):
    one_line_product = []
    for frag in partial_product[0]:
        one_line_product.extend(frag)
    gudiecap = np.zeros((1, 100))
    gudiecap[:, :len(one_line_product)] = np.array(one_line_product)[np.newaxis, :]
    one_line_coords = []
    for coords in partial_product[1]:
        one_line_coords.extend(coords)
    gudiepos = np.zeros((1, 100, 3))
    gudiepos[:, :len(one_line_coords), :] = np.array(one_line_coords)[np.newaxis, :]
    fragUtil = FragmolUtil()
    smiles, ss, moleculars = fragUtil.decode3d(gudiecap, gudiepos)
    return smiles, ss, moleculars, gudiecap


def molecular_workflow(frag_level, warehouse, partial_product, factory_args, results, args,
                       writer=None, summary_path=None, summary=None, start_time=None):
    if len(partial_product[0]) and (2 in partial_product[0][-1]):
        try:
            smiles, ss, moleculars, gudiecap = write_product(partial_product)
        except Exception as e:
            print(f"  ! write_product fail: {e}")
            return
        if not moleculars or moleculars[0] is None:
            print("  ! moleculars[0] is None — skipping")
            return
        center = factory_args[4][0]
        try:
            mol = changepos2(moleculars[0], center.cpu().numpy())
            mol_h_removed = Chem.RemoveHs(mol)
            results.append(mol_h_removed)
            smi = Chem.MolToSmiles(mol_h_removed)
            print(f"  + mol #{len(results)} smiles={smi}")
            # Incremental SDF write
            if writer is not None:
                try:
                    mol_h_removed.SetProp("_Name", f"L0_{len(results)-1}")
                    writer.write(mol_h_removed)
                    writer.flush() if hasattr(writer, "flush") else None
                except Exception as e:
                    print(f"  ! incremental write fail: {e}")
            # Incremental summary flush
            if summary is not None and summary_path is not None:
                summary["n_sampled"] = len(results)
                summary["last_smiles"] = smi
                summary["elapsed_sec"] = (time.time() - start_time) if start_time else None
                _flush_summary(summary_path, summary)
        except Exception as e:
            print(f"  ! failed to add mol: {e}")
        return
    if len(results) >= args.gennums:
        return
    if frag_level + 1 > len(warehouse[0]):
        try:
            captions, pred_pos = get_partial_to_warehouse(frag_level, factory_args, partial_product, args)
        except Exception as e:
            import traceback as _tb
            print(f"  ! get_partial_to_warehouse fail: {e}")
            _tb.print_exc()
            return
        if len(captions) == 0:
            return
        warehouse[0].append(captions)
        warehouse[1].append(pred_pos)
    for index in range(len(warehouse[0][frag_level])):
        if len(results) >= args.gennums:
            break
        frag = warehouse[0][frag_level][index]
        coords = warehouse[1][frag_level][index]
        partial_product[0].append(frag)
        partial_product[1].append(coords)
        molecular_workflow(frag_level + 1, warehouse, partial_product, factory_args, results, args,
                           writer=writer, summary_path=summary_path, summary=summary, start_time=start_time)
        partial_product[0].pop(-1)
        partial_product[1].pop(-1)
    warehouse[0].pop(-1)
    warehouse[1].pop(-1)


def _open_incremental_writer(output_path):
    """Open SDWriter and prep summary path for incremental writes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = Chem.SDWriter(output_path)
    summary_path = output_path.replace(".sdf", "_summary.json")
    return writer, summary_path


def _flush_summary(summary_path, summary):
    try:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"  ! flush summary failed: {e}")


def run(args):
    # Normalize all paths to absolute since cwd is external/Lingo3DMol
    args.pocket_pdb = _abs(args.pocket_pdb)
    args.output = _abs(args.output)
    print(f"[L0 smoke] pocket={args.pocket_pdb}")
    print(f"[L0 smoke] caption_path={args.caption_path}")
    print(f"[L0 smoke] contact_path={args.contact_path}")
    print(f"[L0 smoke] gennums={args.gennums} min_acceptable={args.min_acceptable}")

    # Load models on CPU (shim makes .cuda() no-op)
    caption_contact = TransformerModel_contact()
    dict_ = torch.load(args.contact_path, map_location="cpu", weights_only=False)
    caption_contact.load_state_dict(dict_, strict=False)
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.eval()

    caption = TransformerModel()
    caption.load_state_dict(torch.load(args.caption_path, map_location="cpu", weights_only=False))
    caption = nn.DataParallel(caption)
    caption.eval()

    # Input line: lgdPath,ncipath,pocketpath
    line = f",,{args.pocket_pdb}"
    testset = testdataset([line])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0)

    # Get the first batch to set up CollisionCheck
    for batch in testloader:
        coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
        coc = CollisionCheck(args.pocket_pdb, args.coc_dis, center=center)
        break

    results = []
    start = time.time()
    sample_num = args.gen_frag_set
    warehouse = [[], []]

    # Open incremental SDF writer + summary
    writer, summary_path = _open_incremental_writer(args.output)
    summary = {
        "n_sampled": 0,
        "elapsed_sec": 0,
        "output_sdf": args.output,
        "pocket_pdb": args.pocket_pdb,
        "status": "running",
        "gennums_target": args.gennums,
        "min_acceptable": args.min_acceptable,
    }
    _flush_summary(summary_path, summary)

    try:
        while len(results) < args.gennums and (time.time() - start) < args.max_run_seconds:
            for batch in testloader:
                coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
                with torch.no_grad():
                    index = index.repeat(sample_num)
                    center = center.repeat(sample_num, 1)
                    if contact_prob.shape[-1] == 0 or contact_scaffold_prob.shape[-1] == 0:
                        model_cp, model_csp = caption_contact(
                            coords=coords, residue=residue, atom_type=atom_type,
                            src_mask=mask, isTrain=args.isTrain)
                    if contact_prob.shape[-1] == 0:
                        contact_prob = model_cp
                    if contact_scaffold_prob.shape[-1] == 0:
                        contact_scaffold_prob = model_csp

                    contact_prob0 = torch.where(contact_prob > args.nci_thrs, 2, 0)
                    contact_scaffold_prob1 = torch.where(contact_scaffold_prob > 0.9, 1, 0)
                    contact_prob1 = contact_prob0 + contact_scaffold_prob1
                    contact_prob = contact_prob.repeat(sample_num, 1)
                    residue_use = residue.repeat(sample_num, 1)
                    residue_mask = torch.where(residue_use != 5, 1.0, 0.0)
                    src_mask_repeat = mask.squeeze(1).repeat(sample_num, 1)
                    contact_prob = contact_prob.masked_fill(src_mask_repeat == 0, 0)
                    contact_prob = contact_prob.masked_fill(residue_mask == 0, 0)
                    contact_prob = torch.softmax(contact_prob * 5, dim=-1)
                    contact_idx = topkp_random(contact_prob, top_k=args.topk, top_p=0.9, thred=0.0)
                    factory_args = [coords, residue, mask, atom_type, center, caption,
                                    contact_idx, contact_prob1, coc, contact_scaffold_prob1]
                    molecular_workflow(0, warehouse, [[], []], factory_args, results, args,
                                       writer=writer, summary_path=summary_path,
                                       summary=summary, start_time=start)
                    if len(results) >= args.gennums:
                        break
            if len(results) >= args.gennums:
                break
            # Early-exit on min_acceptable when we've spent at least 60% of budget
            elapsed_so_far = time.time() - start
            if (len(results) >= args.min_acceptable
                    and elapsed_so_far >= 0.6 * args.max_run_seconds):
                print(f"[L0 smoke] min_acceptable={args.min_acceptable} met with "
                      f"n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[L0 smoke] generated {len(results)} mols in {elapsed:.1f}s status={status}")
    print(f"[L0 smoke] wrote SDF: {args.output}")

    summary.update({
        "n_sampled": len(results),
        "elapsed_sec": elapsed,
        "status": status,
    })
    _flush_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--caption_path", default="checkpoint/gen_mol.pkl")
    p.add_argument("--gennums", type=int, default=3)
    p.add_argument("--min_acceptable", type=int, default=1,
                   help="Smoke PASSES if at least this many valid mols emitted.")
    p.add_argument("--gen_frag_set", type=int, default=20)
    p.add_argument("--prod_time", type=int, default=1)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=2.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=1800)
    p.add_argument("--isTrain", action="store_true")
    p.add_argument("--USE_THRESHOLD", action="store_true", default=True)
    p.add_argument("--isMultiSample", action="store_true", default=True)
    p.add_argument("--isGuideSample", action="store_true", default=True)
    p.add_argument("--OnceMolGen", action="store_true")
    args = p.parse_args()
    run(args)
