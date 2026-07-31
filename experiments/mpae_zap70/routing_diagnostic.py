#!/usr/bin/env python3
"""Track D — Cross-attention weight + logit-KL diagnostic on v4_composite ckpt.

Verdict logic:
  - If mean cross-attn on POSE token > 5% AND per-step KL(A||C) > 0.05 nats:
      routing works, sampling noise wall — try low temperature to expose signal
  - If cross-attn near 0:
      routing failure (pose token effectively unused by decoder)
  - If cross-attn > 5% but KL ≈ 0:
      pose reaches h_pooled (contrastive/reg heads happy) but decoder ignores it

Memory layout (built in M1aV2ConditionedModel._build_conditioned_memory):
  position 0 : pocket_vec
  position 1 : pose_vec  <-- our target
  positions 2..L+1 : encoded source tokens

For each of N_ANCHORS Mol1-randomized anchors we run TWO forward passes
(A and C clamps). At every decoding position we read:
  - cross-attention weights on memory position 1, averaged over heads
  - full log-prob distribution over vocab (teacher-forced on Mol1 target)
  - per-position KL(logits_A || logits_C)

Then we also sample 100 mols at T=0.5 per clamp and measure emitted planar
dihedral to see if the signal shows up in output when sampling noise is reduced.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT
REINVENT4_ROOT = Path("/home/shaharh_quris_ai/REINVENT4") if not LOCAL_ROOT.exists() else Path("/Users/shaharharel/Documents/github/REINVENT4")
sys.path.insert(0, str(REINVENT4_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2/fixes"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments/mpae_zap70"))

from m1a_v2_model import load_m1a_v2  # noqa: E402
from train_zap70_target import RegressionHead, randomize_smi, _make_std_mask  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[N]")

N_ANCHORS = 100
POSE_MEM_POS = 1  # position of pose_vec in memory (0=pocket, 1=pose, 2..=src)
N_SAMPLES_T05 = 100
N_KL_STEPS = 20  # first N decoder positions for KL averaging


def encode_mol1_batch(model, n_anchors, device, seed=0):
    """Return src_t, src_mask, trg_t, trg_mask for n Mol1-randomized anchors."""
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    rng = np.random.default_rng(seed)
    # Use randomized SMILES like sampling script does
    anchors = []
    for i in range(n_anchors):
        s = randomize_smi(MOL1_SMI)
        anchors.append(s)
    seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
    L = max(len(s) for s in seqs)
    src = np.zeros((n_anchors, L), dtype=np.int64)
    for j, s in enumerate(seqs):
        src[j, :len(s)] = s
    # For target (teacher forcing) use the canonical Mol1 - same for all anchors
    # We want deterministic teacher forcing to compare A vs C
    canon_seq = np.array(vocab.encode(tok.tokenize(MOL1_SMI)), dtype=np.int64)
    trg = np.tile(canon_seq[None], (n_anchors, 1)).astype(np.int64)
    src_t = torch.from_numpy(src).to(device)
    trg_t = torch.from_numpy(trg).to(device)
    src_mask = (src_t != 0).unsqueeze(-2).long()
    trg_mask = _make_std_mask(trg_t[:, :-1], 0)
    return src_t, src_mask, trg_t, trg_mask


@torch.no_grad()
def run_forward_capture(model, src_t, src_mask, trg_t, trg_mask,
                        res_emb, res_mask, pose_norm):
    """Run full forward, capture per-layer cross-attn on POSE token and logits.

    Returns:
        attn_pose_by_layer: (n_layers, B, Ldec) mean over heads
        attn_pocket_by_layer: (n_layers, B, Ldec) mean over heads (for scale)
        attn_total_first_two_by_layer: (n_layers, B, Ldec) — attn on positions 0+1
        log_prob: (B, Ldec, V)
    """
    net = model.base.network
    device = model.device
    # Build conditioned memory using model's helper (matches training/sampling)
    memory_ext, src_mask_ext = model._build_conditioned_memory(
        src_t, src_mask, res_emb, res_mask, pose_norm)
    trg_in = trg_t[:, :-1]
    # Forward decoder — src_attn stores its .attn as a buffer on each layer
    tgt_embed = net.tgt_embed(trg_in)
    x = tgt_embed
    attn_pose_by_layer = []
    attn_pocket_by_layer = []
    for layer in net.decoder.layers:
        x = layer(x, memory_ext, src_mask_ext, trg_mask)
        # layer.src_attn.attn: (B, H, Ldec, Lmem)
        a = layer.src_attn.attn  # softmax weights
        # Mean over heads
        a_mean = a.mean(dim=1)  # (B, Ldec, Lmem)
        attn_pose_by_layer.append(a_mean[:, :, POSE_MEM_POS].detach().cpu())
        attn_pocket_by_layer.append(a_mean[:, :, 0].detach().cpu())
    x = net.decoder.norm(x)
    log_prob = net.generator(x, model.base.temperature)  # (B, Ldec, V)
    return (torch.stack(attn_pose_by_layer),  # (n_layers, B, Ldec)
            torch.stack(attn_pocket_by_layer),
            log_prob.detach().cpu())


def per_step_kl(log_prob_A, log_prob_C):
    """KL(P_A || P_C) per (batch, position). log_prob = log softmax outputs."""
    # log_prob shape (B, Ldec, V)
    p_A = torch.exp(log_prob_A)
    kl = (p_A * (log_prob_A - log_prob_C)).sum(dim=-1)  # (B, Ldec)
    return kl


@torch.no_grad()
def sample_at_temperature(model, res_emb, res_mask, pose_norm, n=100,
                          temperature=0.5, batch_size=25, seed=42):
    """Sample n Mol1-anchored SMILES at given temperature."""
    device = model.device
    vocab = model.base.vocabulary
    tok = model.base.tokenizer
    torch.manual_seed(seed)
    np.random.seed(seed)
    smis = []
    n_done = 0
    while n_done < n:
        bn = min(batch_size, n - n_done)
        anchors = [randomize_smi(MOL1_SMI) for _ in range(bn)]
        seqs = [np.array(vocab.encode(tok.tokenize(s)), dtype=np.int64) for s in anchors]
        L = max(len(s) for s in seqs)
        src = np.zeros((bn, L), dtype=np.int64)
        for j, s in enumerate(seqs):
            src[j, :len(s)] = s
        src_t = torch.from_numpy(src).to(device)
        src_mask = (src_t != 0).unsqueeze(-2).long()
        re = torch.from_numpy(np.tile(res_emb[None], (bn, 1, 1))).to(device)
        rm = torch.from_numpy(np.tile(res_mask[None], (bn, 1))).to(device)
        po = torch.from_numpy(np.tile(pose_norm[None], (bn, 1))).to(device)
        out_smiles, _ = model.sample_multinomial(
            src_t, src_mask, re, rm, po, max_length=128, temperature=temperature)
        smis.extend(out_smiles)
        n_done += bn
    return smis


def measure_planar_dihedral(smi_list, seed=42):
    """Return list of |planar_dihedral| medians (deg) per SMILES."""
    vals = []
    for smi in smi_list:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        matches = m.GetSubstructMatches(ACRYL_SMARTS)
        if not matches:
            continue
        a1, a2, a3, a_o, a_n = matches[0]
        try:
            mH = Chem.AddHs(m)
            p = AllChem.ETKDGv3(); p.randomSeed = seed
            cid = AllChem.EmbedMolecule(mH, p)
            if cid < 0:
                continue
            conf = mH.GetConformer(cid)
            phi = float(AllChem.GetDihedralDeg(conf, a1, a2, a3, a_n))
            phi_wrap = ((phi + 180.0) % 360.0) - 180.0
            vals.append(min(abs(phi_wrap), abs(180.0 - abs(phi_wrap))))
        except Exception:
            continue
    return vals


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)

    ckpt_path = PROJECT_ROOT / "models/m1a_v2_v4_composite.ckpt"
    prior_path = PROJECT_ROOT / "models/reinvent4_mol2mol_covalent_ft.prior"
    pose_stats_path = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/pose_stats_zap70.json"
    esm_path = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/esm_cache_zap70.npz"
    labeled_path = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/labeled_mols_zap70.npz"
    out_dir = PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/routing_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    ps = json.loads(pose_stats_path.read_text())
    pose_mean = np.array(ps["pose_mean"], dtype=np.float32)
    pose_std = np.array(ps["pose_std"], dtype=np.float32)

    print(f"[load] {ckpt_path}", flush=True)
    model = load_m1a_v2(str(prior_path), device, ckpt_path=None,
                        pose_mean=torch.from_numpy(pose_mean),
                        pose_std=torch.from_numpy(pose_std))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    n_layers = len(model.base.network.decoder.layers)
    print(f"[model] decoder layers = {n_layers}", flush=True)

    # Load ESM pocket cache (pocket 0 as canonical, matches sampling script)
    E = np.load(esm_path, allow_pickle=True)
    res_emb = E["residues_emb"][0].astype(np.float32)
    res_mask = E["residues_mask"][0].astype(bool)

    # Compute clamps A and C (best / worst quartile pooling) — target=composite
    L = np.load(labeled_path, allow_pickle=True)
    pose_all = L["pose_boltz"].astype(np.float32)
    tgt_all = L["target_composite"].astype(np.float32)
    q75 = float(np.quantile(tgt_all, 0.75))
    clamp_A = pose_all.mean(axis=0)
    clamp_C = pose_all[tgt_all >= q75].mean(axis=0)
    print(f"[clamps] A={clamp_A}", flush=True)
    print(f"[clamps] C={clamp_C}", flush=True)

    def norm(p):
        return (p - pose_mean) / np.clip(pose_std, 1e-6, None)
    pose_norm_A = torch.from_numpy(norm(clamp_A)).to(device).float().unsqueeze(0)
    pose_norm_C = torch.from_numpy(norm(clamp_C)).to(device).float().unsqueeze(0)

    # Build batch of anchors (Mol1-randomized) — one batch of N_ANCHORS
    print(f"[encode] building {N_ANCHORS} anchors", flush=True)
    src_t, src_mask, trg_t, trg_mask = encode_mol1_batch(model, N_ANCHORS, device, seed=0)
    B = src_t.shape[0]

    # Tile ESM pocket for whole batch
    re = torch.from_numpy(np.tile(res_emb[None], (B, 1, 1))).to(device)
    rm = torch.from_numpy(np.tile(res_mask[None], (B, 1))).to(device)
    po_A = pose_norm_A.expand(B, -1)
    po_C = pose_norm_C.expand(B, -1)

    # Run captures
    print(f"[forward] Clamp A", flush=True)
    t0 = time.time()
    attn_pose_A, attn_pocket_A, log_prob_A = run_forward_capture(
        model, src_t, src_mask, trg_t, trg_mask, re, rm, po_A)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print(f"[forward] Clamp C", flush=True)
    t0 = time.time()
    attn_pose_C, attn_pocket_C, log_prob_C = run_forward_capture(
        model, src_t, src_mask, trg_t, trg_mask, re, rm, po_C)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    # ==== attn_weights.json ====
    # Aggregate over anchors, keep layer × decode step
    # attn_pose_* shape: (n_layers, B, Ldec)
    attn_pose_mean_A = attn_pose_A.mean(dim=1).numpy()  # (n_layers, Ldec)
    attn_pose_mean_C = attn_pose_C.mean(dim=1).numpy()
    attn_pocket_mean_A = attn_pocket_A.mean(dim=1).numpy()
    attn_pocket_mean_C = attn_pocket_C.mean(dim=1).numpy()

    # Overall means (over layers × dec positions × anchors)
    overall_pose_A = float(attn_pose_A.mean())
    overall_pose_C = float(attn_pose_C.mean())
    overall_pocket_A = float(attn_pocket_A.mean())
    overall_pocket_C = float(attn_pocket_C.mean())

    # Reference: uniform attention over memory would give 1/L
    L_mem = src_t.shape[1] + 2  # +2 conditioning tokens
    uniform_ref = 1.0 / float(L_mem)

    attn_out = {
        "notes": (
            "Cross-attention (mean over heads) on conditioning tokens. "
            "Position 0 = pocket_vec, position 1 = pose_vec. "
            f"Uniform ref = 1/{L_mem} = {uniform_ref:.4f}. "
            "Values are means over 100 anchors."
        ),
        "n_layers": n_layers,
        "L_mem": int(L_mem),
        "uniform_ref": uniform_ref,
        "overall_mean_attn_on_pose_A": overall_pose_A,
        "overall_mean_attn_on_pose_C": overall_pose_C,
        "overall_mean_attn_on_pocket_A": overall_pocket_A,
        "overall_mean_attn_on_pocket_C": overall_pocket_C,
        "attn_pose_by_layer_A_avg_over_decode": [float(x) for x in attn_pose_mean_A.mean(axis=1)],
        "attn_pose_by_layer_C_avg_over_decode": [float(x) for x in attn_pose_mean_C.mean(axis=1)],
        "attn_pocket_by_layer_A_avg_over_decode": [float(x) for x in attn_pocket_mean_A.mean(axis=1)],
        "attn_pocket_by_layer_C_avg_over_decode": [float(x) for x in attn_pocket_mean_C.mean(axis=1)],
        "attn_pose_first_20_dec_steps_layer_avg_A": [float(x) for x in attn_pose_mean_A[:, :20].mean(axis=0)],
        "attn_pose_first_20_dec_steps_layer_avg_C": [float(x) for x in attn_pose_mean_C[:, :20].mean(axis=0)],
    }
    (out_dir / "attn_weights.json").write_text(json.dumps(attn_out, indent=2))
    print(f"[write] attn_weights.json", flush=True)

    # ==== logit_kl.json ====
    kl_AC = per_step_kl(log_prob_A, log_prob_C)  # (B, Ldec)
    kl_mean_per_step = kl_AC.mean(dim=0).numpy()  # (Ldec,)
    kl_mean_first20 = float(kl_AC[:, :N_KL_STEPS].mean())
    kl_mean_full = float(kl_AC.mean())

    # Sanity: KL(A||A) should be ~0
    kl_AA = per_step_kl(log_prob_A, log_prob_A)
    kl_AA_first20 = float(kl_AA[:, :N_KL_STEPS].mean())

    kl_out = {
        "notes": (
            "KL(P_A || P_C) in nats, teacher-forced on canonical Mol1 SMILES. "
            "Averaged over 100 Mol1-randomized anchors. First 20 decode steps."
        ),
        "n_anchors": int(B),
        "mean_kl_first_20_steps": kl_mean_first20,
        "mean_kl_full_seq": kl_mean_full,
        "mean_kl_AA_sanity": kl_AA_first20,
        "per_step_kl_first_20": [float(x) for x in kl_mean_per_step[:N_KL_STEPS]],
    }
    (out_dir / "logit_kl.json").write_text(json.dumps(kl_out, indent=2))
    print(f"[write] logit_kl.json  kl_first20={kl_mean_first20:.4f} nats", flush=True)

    # ==== T=0.5 sampling ====
    print(f"[sample T=0.5] Clamp A ({N_SAMPLES_T05} mols)", flush=True)
    t0 = time.time()
    smis_A_t05 = sample_at_temperature(
        model, res_emb, res_mask, norm(clamp_A).astype(np.float32),
        n=N_SAMPLES_T05, temperature=0.5, seed=42)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print(f"[sample T=0.5] Clamp C ({N_SAMPLES_T05} mols)", flush=True)
    t0 = time.time()
    smis_C_t05 = sample_at_temperature(
        model, res_emb, res_mask, norm(clamp_C).astype(np.float32),
        n=N_SAMPLES_T05, temperature=0.5, seed=42)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    phi_A = measure_planar_dihedral(smis_A_t05)
    phi_C = measure_planar_dihedral(smis_C_t05)
    med_A = float(np.median(phi_A)) if phi_A else float("nan")
    med_C = float(np.median(phi_C)) if phi_C else float("nan")
    delta_AC_T05 = med_C - med_A

    # Also do a quick T=1.0 reference for comparison
    print(f"[sample T=1.0] Clamp A ({N_SAMPLES_T05} mols)", flush=True)
    smis_A_t10 = sample_at_temperature(
        model, res_emb, res_mask, norm(clamp_A).astype(np.float32),
        n=N_SAMPLES_T05, temperature=1.0, seed=42)
    smis_C_t10 = sample_at_temperature(
        model, res_emb, res_mask, norm(clamp_C).astype(np.float32),
        n=N_SAMPLES_T05, temperature=1.0, seed=42)
    phi_A_t10 = measure_planar_dihedral(smis_A_t10)
    phi_C_t10 = measure_planar_dihedral(smis_C_t10)
    med_A_t10 = float(np.median(phi_A_t10)) if phi_A_t10 else float("nan")
    med_C_t10 = float(np.median(phi_C_t10)) if phi_C_t10 else float("nan")
    delta_AC_T10 = med_C_t10 - med_A_t10

    pose_shift_out = {
        "notes": (
            "Emitted planar dihedral median at T=0.5 vs T=1.0. "
            f"Clamps derived from target_composite (q75={q75:.4f}). "
            "Delta_AC = median(C) - median(A) in deg."
        ),
        "n_per_clamp": int(N_SAMPLES_T05),
        "T05_clamp_A_planar_dihedral_median_deg": med_A,
        "T05_clamp_C_planar_dihedral_median_deg": med_C,
        "T05_delta_AC_deg": delta_AC_T05,
        "T05_n_valid_A": len(phi_A),
        "T05_n_valid_C": len(phi_C),
        "T10_clamp_A_planar_dihedral_median_deg": med_A_t10,
        "T10_clamp_C_planar_dihedral_median_deg": med_C_t10,
        "T10_delta_AC_deg": delta_AC_T10,
        "T10_n_valid_A": len(phi_A_t10),
        "T10_n_valid_C": len(phi_C_t10),
    }
    (out_dir / "T05_pose_shift.json").write_text(json.dumps(pose_shift_out, indent=2))
    print(f"[write] T05_pose_shift.json  delta_AC(T=0.5)={delta_AC_T05:.2f}deg", flush=True)

    # ==== Verdict ====
    lines = []
    lines.append("# Track D — Cross-Attention + Logit-KL Diagnostic on v4_composite\n")
    lines.append(f"**Ckpt**: `{ckpt_path.name}`\n")
    lines.append(f"**N anchors**: {int(B)}  •  **Decoder layers**: {n_layers}  •  **Memory positions**: {L_mem} (uniform ref = {uniform_ref:.4f})\n")

    lines.append("## 1) Cross-attention weight on POSE token (memory position 1)\n")
    lines.append(f"- Mean over layers × dec steps × anchors, Clamp A: **{overall_pose_A:.4f}** ({100*overall_pose_A:.2f}%)\n")
    lines.append(f"- Mean over layers × dec steps × anchors, Clamp C: **{overall_pose_C:.4f}** ({100*overall_pose_C:.2f}%)\n")
    lines.append(f"- Reference: uniform 1/L = {uniform_ref:.4f} ({100*uniform_ref:.2f}%)\n")
    lines.append(f"- Pocket-token attention for scale, A: {overall_pocket_A:.4f}, C: {overall_pocket_C:.4f}\n\n")
    lines.append("Per-layer average attn on pose (A):\n")
    for i, v in enumerate(attn_out["attn_pose_by_layer_A_avg_over_decode"]):
        lines.append(f"  - layer {i}: {v:.4f} ({100*v:.2f}%)\n")
    lines.append("\n")

    lines.append("## 2) Per-token logit KL(A || C)\n")
    lines.append(f"- Mean KL over first {N_KL_STEPS} decoder steps × {int(B)} anchors: **{kl_mean_first20:.4f} nats**\n")
    lines.append(f"- Full-seq mean KL: {kl_mean_full:.4f} nats\n")
    lines.append(f"- Sanity KL(A||A): {kl_AA_first20:.6f} nats (should be ~0)\n\n")

    lines.append("## 3) Emitted pose shift at T=0.5 vs T=1.0\n")
    lines.append(f"- T=1.0 Δ_AC (planar dihedral median): {delta_AC_T10:.2f}°  (A={med_A_t10:.2f}, C={med_C_t10:.2f})\n")
    lines.append(f"- T=0.5 Δ_AC: **{delta_AC_T05:.2f}°**  (A={med_A:.2f}, C={med_C:.2f})\n\n")

    # Threshold-based verdict
    attn_ok = max(overall_pose_A, overall_pose_C) > 0.05
    kl_ok = kl_mean_first20 > 0.05

    lines.append("## Verdict\n")
    if attn_ok and kl_ok:
        verdict = "ROUTING WORKS — sampling-noise wall"
        interpretation = (
            f"Cross-attn on POSE token is {100*max(overall_pose_A, overall_pose_C):.1f}% "
            f"(> 5% threshold) and per-step KL is {kl_mean_first20:.3f} nats "
            f"(> 0.05 threshold). The decoder DOES route pose info AND its logits shift "
            f"between clamps. The failure to move emitted geometry at T=1.0 is a "
            f"**sampling-noise wall**: multinomial sampling averages out the small logit "
            f"deltas. Lower T (or beam/nucleus filtering) should expose the signal — "
            f"observed T=0.5 shift is {delta_AC_T05:.2f}° vs T=1.0 shift {delta_AC_T10:.2f}°."
        )
    elif not attn_ok and not kl_ok:
        verdict = "ROUTING FAILURE — pose token unused"
        interpretation = (
            f"Cross-attn on POSE token is only {100*max(overall_pose_A, overall_pose_C):.2f}% "
            f"(< 5% threshold). Decoder is not attending to the pose conditioning token. "
            f"h_pooled receives pose info (contrastive/regression heads succeed at 0.877/0.488) "
            f"but the decoder's autoregressive path routes it away. Fix requires arch change: "
            f"pose adapters into every decoder layer (FiLM/prefix), higher pose-token weight, "
            f"or pose-conditioned target embedding."
        )
    elif attn_ok and not kl_ok:
        verdict = "ATTN OK, LOGITS FLAT — decoder read-and-ignore"
        interpretation = (
            f"Cross-attn on POSE is {100*max(overall_pose_A, overall_pose_C):.1f}% "
            f"but per-step KL is only {kl_mean_first20:.4f} nats (< 0.05). Decoder "
            f"attends to the pose token but its value contribution collapses into "
            f"the FFN residual. Effectively the pose-conditioned key/value are "
            f"projected to a near-zero delta in output space. Fix: stronger pose "
            f"projection, larger pose-encoder capacity, or auxiliary pose-recon loss "
            f"on early decoder layers."
        )
    else:
        verdict = "AMBIGUOUS — low attn but logits shift"
        interpretation = (
            f"Cross-attn on POSE is only {100*max(overall_pose_A, overall_pose_C):.2f}% "
            f"but logits DO differ by {kl_mean_first20:.3f} nats. The pose signal is "
            f"reaching the decoder through some other path (probably early source-embedding "
            f"cross-talk in encoder), not through direct cross-attention on the pose "
            f"token. Investigate encoder self-attention on the pocket_vec/pose_vec tokens."
        )

    lines.append(f"**{verdict}**\n\n")
    lines.append(interpretation + "\n\n")
    lines.append("### H1/H2 mapping\n")
    lines.append("- H1: contrastive head aligns h_pooled with pose (verified pre-diagnostic: val_pos_wins=0.877)\n")
    lines.append("- H2: pose info survives the decoder cross-attention path\n")
    if attn_ok and kl_ok:
        lines.append("  → H2 SUPPORTED (attention + logit deltas both nonzero). Bottleneck is sampling stochasticity, not routing.\n")
    elif not attn_ok:
        lines.append("  → H2 REJECTED (decoder does not attend to pose token). Fix must add explicit pose injection.\n")
    else:
        lines.append("  → H2 partially: attention exists but doesn't translate to logit change → subtle routing gap.\n")

    (out_dir / "diagnostic_report.md").write_text("".join(lines))
    print(f"[write] diagnostic_report.md", flush=True)
    print(f"\n===== VERDICT: {verdict} =====", flush=True)
    print(f"attn_pose(A) = {overall_pose_A:.4f}  attn_pose(C) = {overall_pose_C:.4f}", flush=True)
    print(f"kl_first20 = {kl_mean_first20:.4f} nats", flush=True)
    print(f"T=0.5 delta_AC = {delta_AC_T05:.2f}deg  vs  T=1.0 delta_AC = {delta_AC_T10:.2f}deg", flush=True)


if __name__ == "__main__":
    main()
