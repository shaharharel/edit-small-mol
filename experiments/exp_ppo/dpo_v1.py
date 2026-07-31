"""DPO-v1: Direct Preference Optimization on FiLMDelta-ranked pairs.

Single supervised loss (no clipping, no entropy, no value head):
  L = -log σ(β · [logπ_θ(chosen) - logπ_ref(chosen)
                 - logπ_θ(rejected) + logπ_ref(rejected)])

Per outer step:
  1. Sample B candidates from π_θ_old anchored on prompts from seed pool
  2. Score via FiLM "film_pIC50_sigmoid · ladder · QED" composite (ladder reward)
  3. Sort by reward; top-K vs bottom-K = K (chosen, rejected) pairs
  4. Compute log-prob under π_θ (grad) and π_ref (no grad) for both
  5. DPO loss; backward; Adam step

Diversity filter (Murcko) is applied to candidate scores BEFORE ranking, so
overrepresented scaffolds get zero-reward → become "rejected" preferentially.
"""
from __future__ import annotations
import argparse, csv, logging, sys, time, warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud

try:
    from reinvent.runmodes.create_adapter import create_adapter
    from reinvent.models.transformer.core.dataset.dataset import Dataset
    from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer
    from reinvent.chemistry import conversions
except Exception as e:
    sys.stderr.write(f"REINVENT4 import failed: {e}\n"); raise

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dpo_v1")


class RESTScorer:
    def __init__(self, url, predictor_id="ladder", predictor_version="dpo_v1",
                 inp_fmt="smiles", timeout=600.0):
        import requests
        self._requests = requests; self.url = url
        self.params = dict(predictor_id=predictor_id, predictor_version=predictor_version, inp_fmt=inp_fmt)
        self.timeout = timeout
    def __call__(self, smiles):
        body = [{"input_string": s if s else "", "query_id": str(i)} for i, s in enumerate(smiles)]
        r = self._requests.post(self.url, params=self.params, json=body, timeout=self.timeout)
        r.raise_for_status()
        out = r.json()["output"]["successes_list"]
        scores = np.zeros(len(smiles), dtype=np.float32)
        for entry in out:
            scores[int(entry["query_id"])] = float(entry["output_value"])
        return scores


def _standardize_smiles_list(smilies, randomize=True, isomeric=True):
    out = []
    for s in smilies:
        try:
            s_std = conversions.convert_to_standardized_smiles(s)
        except Exception:
            s_std = s
        if randomize:
            try:
                mol = conversions.smile_to_mol(s_std)
                if mol is not None:
                    s_std = conversions.mol_to_random_smiles(mol, isomericSmiles=isomeric)
            except Exception:
                pass
        out.append(s_std)
    return out


def _validate_smiles(out_list):
    from rdkit import Chem
    return [(bool(s) and Chem.MolFromSmiles(s) is not None) for s in out_list]


def _warhead_match_rate(smiles_list):
    from rdkit import Chem
    pats = [Chem.MolFromSmarts(p) for p in [
        "C=CC(=O)N1Cc2ccccc2C1", "C(=O)C=C[#7]", "C=CC(=O)N", "[CX3]=[CX3][CX3]=[OX1]"]]
    pats = [p for p in pats if p is not None]
    n_tot = n_match = 0
    for s in smiles_list:
        if not s: continue
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        n_tot += 1
        if any(m.HasSubstructMatch(p) for p in pats): n_match += 1
    return float(n_match) / float(n_tot) if n_tot else 0.0


def _thiq_exact_rate(smiles_list):
    from rdkit import Chem
    pat = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
    if pat is None: return 0.0
    n_tot = n_match = 0
    for s in smiles_list:
        if not s: continue
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        n_tot += 1
        if m.HasSubstructMatch(pat): n_match += 1
    return float(n_match) / float(n_tot) if n_tot else 0.0


def sample_batch(agent_old, input_pool, batch_size, device, randomize=True):
    seed_smilies = list(np.random.choice(input_pool, size=batch_size, replace=True))
    proc = _standardize_smiles_list(seed_smilies, randomize=randomize)
    tokenizer = SMILESTokenizer()
    dataset = Dataset(proc, agent_old.get_vocabulary(), tokenizer)
    loader = tud.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Dataset.collate_fn)
    inputs_all, outputs_all, nlls_all = [], [], []
    agent_old.set_mode("inference")
    for src, src_mask in loader:
        src = src.to(device); src_mask = src_mask.to(device)
        sb = agent_old.sample(src, src_mask, "multinomial")
        inputs_all.extend(list(sb.input))
        outputs_all.extend(list(sb.output))
        nlls = sb.nlls
        if not isinstance(nlls, torch.Tensor):
            nlls = torch.tensor(nlls, dtype=torch.float32)
        nlls_all.append(nlls.float())
    return inputs_all, outputs_all, torch.cat(nlls_all, dim=0).to(device)


def compute_log_probs(agent, inputs, outputs, device, requires_grad=True):
    """Compute log π(out | in) under agent for each pair, with grad if requested.
    Returns tensor of shape (B,) with NaN for token-encoding failures."""
    tokenizer = SMILESTokenizer()
    vocab = agent.get_vocabulary()
    encoded_pairs = []; keep_idx = []
    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        try:
            ei = vocab.encode(tokenizer.tokenize(inp))
            eo = vocab.encode(tokenizer.tokenize(out))
        except KeyError:
            continue
        encoded_pairs.append((torch.tensor(ei).long(), torch.tensor(eo).long(), torch.tensor([0.0]).float()))
        keep_idx.append(i)
    if not encoded_pairs:
        return torch.full((len(inputs),), float("nan"), device=device)
    nll_chunks = []
    MB = 16
    for start in range(0, len(encoded_pairs), MB):
        chunk = encoded_pairs[start:start + MB]
        dto = PairedDataset.collate_fn(chunk)
        src = dto.input.to(device); src_mask = dto.input_mask.to(device)
        trg = dto.output.to(device); trg_mask = dto.output_mask.to(device)
        if requires_grad:
            agent.set_mode("training")
            nll = agent.likelihood(src, src_mask, trg, trg_mask)
        else:
            agent.set_mode("inference")
            with torch.no_grad():
                nll = agent.likelihood(src, src_mask, trg, trg_mask)
        nll_chunks.append(nll)
    nll_all_kept = torch.cat(nll_chunks, dim=0)
    out_lp = torch.full((len(inputs),), float("nan"), device=device)
    out_lp[keep_idx] = -nll_all_kept
    return out_lp


@dataclass
class DPOConfig:
    beta: float = 0.1
    lr: float = 1e-6
    batch_size: int = 64
    k_pairs: int = 16  # top-K vs bottom-K = K (chosen, rejected) pairs
    outer_steps: int = 50


class DPOTrainer:
    def __init__(self, agent, agent_old, prior, scorer, input_pool, device, cfg,
                 div_bucket=25, div_minscore=0.4):
        self.agent = agent; self.agent_old = agent_old; self.prior = prior
        self.scorer = scorer; self.input_pool = input_pool; self.device = device; self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=cfg.lr)
        self.scaffold_counts: Counter = Counter()
        self.div_bucket = int(div_bucket); self.div_minscore = float(div_minscore)

    def _swap_old_to_new(self):
        self.agent_old.network.load_state_dict(self.agent.network.state_dict())
        for p in self.agent_old.get_network_parameters(): p.requires_grad = False

    def _clear_attn_refs(self):
        for model in (self.agent, self.agent_old, self.prior):
            net = getattr(model, "network", None)
            if net is None: continue
            for m in net.modules():
                if hasattr(m, "attn"): m.attn = None

    def _apply_diversity(self, smiles_list, rewards):
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        out = rewards.copy()
        for i, (s, r) in enumerate(zip(smiles_list, rewards)):
            if r < self.div_minscore: continue
            if not s: continue
            try:
                m = Chem.MolFromSmiles(s)
                if m is None: continue
                scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            except Exception:
                continue
            n = self.scaffold_counts[scaf]
            self.scaffold_counts[scaf] = n + 1
            if n >= self.div_bucket:
                out[i] = 0.0
        return out

    def step(self, step_idx):
        # 1. Sample candidates
        inputs, outputs, _old_nlls = sample_batch(self.agent_old, self.input_pool,
                                                   self.cfg.batch_size, self.device, randomize=True)
        valid_mask_list = _validate_smiles(outputs)
        n_valid = int(sum(valid_mask_list))
        rewards = np.zeros(len(outputs), dtype=np.float32)
        if n_valid > 0:
            valid_smiles = [outputs[i] for i, ok in enumerate(valid_mask_list) if ok]
            valid_scores = self.scorer(valid_smiles)
            j = 0
            for i, ok in enumerate(valid_mask_list):
                if ok: rewards[i] = valid_scores[j]; j += 1
        rewards = self._apply_diversity(outputs, rewards)
        warhead_any = _warhead_match_rate(outputs)
        thiq_exact = _thiq_exact_rate(outputs)

        # 2. Rank and build top-K vs bottom-K pairs
        order = np.argsort(rewards)  # ascending
        K = min(self.cfg.k_pairs, len(rewards) // 2)
        if K == 0:
            return {
                "step": step_idx, "n_valid": n_valid, "n_pairs": 0,
                "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
                "mean_chosen_reward": 0.0, "mean_rejected_reward": 0.0,
                "reward_gap": 0.0, "loss": float("nan"),
                "mean_kl_to_prior": 0.0, "mean_entropy": 0.0,
                "warhead_any_rate": warhead_any, "thiq_exact_rate": thiq_exact,
            }
        bot_idx = order[:K]            # rejected (low reward)
        top_idx = order[-K:][::-1]     # chosen (high reward)

        chosen_inputs = [inputs[i] for i in top_idx]
        chosen_outputs = [outputs[i] for i in top_idx]
        rejected_inputs = [inputs[i] for i in bot_idx]
        rejected_outputs = [outputs[i] for i in bot_idx]
        chosen_rewards = rewards[top_idx]
        rejected_rewards = rewards[bot_idx]
        reward_gap = float(np.mean(chosen_rewards - rejected_rewards))

        # 3. log-probs under θ (grad) and ref (no grad) for chosen & rejected
        self.optimizer.zero_grad()
        lp_chosen_theta = compute_log_probs(self.agent, chosen_inputs, chosen_outputs, self.device, requires_grad=True)
        lp_rejected_theta = compute_log_probs(self.agent, rejected_inputs, rejected_outputs, self.device, requires_grad=True)
        with torch.no_grad():
            lp_chosen_ref = compute_log_probs(self.prior, chosen_inputs, chosen_outputs, self.device, requires_grad=False)
            lp_rejected_ref = compute_log_probs(self.prior, rejected_inputs, rejected_outputs, self.device, requires_grad=False)
        finite = (torch.isfinite(lp_chosen_theta) & torch.isfinite(lp_rejected_theta) &
                  torch.isfinite(lp_chosen_ref) & torch.isfinite(lp_rejected_ref))
        if finite.sum().item() == 0:
            logger.warning(f"[step {step_idx}] no finite DPO pairs");
            return {
                "step": step_idx, "n_valid": n_valid, "n_pairs": 0,
                "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
                "mean_chosen_reward": float(np.mean(chosen_rewards)),
                "mean_rejected_reward": float(np.mean(rejected_rewards)),
                "reward_gap": reward_gap, "loss": float("nan"),
                "mean_kl_to_prior": 0.0, "mean_entropy": 0.0,
                "warhead_any_rate": warhead_any, "thiq_exact_rate": thiq_exact,
            }
        lp_c_t = lp_chosen_theta[finite]
        lp_r_t = lp_rejected_theta[finite]
        lp_c_r = lp_chosen_ref[finite]
        lp_r_r = lp_rejected_ref[finite]

        # 4. DPO loss
        logits = self.cfg.beta * ((lp_c_t - lp_c_r) - (lp_r_t - lp_r_r))
        loss = -F.logsigmoid(logits).mean()

        # diagnostics
        with torch.no_grad():
            kl_to_prior = float(((lp_c_t - lp_c_r).mean() + (lp_r_t - lp_r_r).mean()) / 2.0)
            entropy_proxy = float(-((lp_c_t.mean() + lp_r_t.mean()) / 2.0))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.get_network_parameters(), max_norm=1.0)
        self.optimizer.step()

        self._swap_old_to_new(); self._clear_attn_refs()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        return {
            "step": step_idx, "n_valid": n_valid, "n_pairs": int(finite.sum().item()),
            "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
            "mean_chosen_reward": float(np.mean(chosen_rewards)),
            "mean_rejected_reward": float(np.mean(rejected_rewards)),
            "reward_gap": reward_gap, "loss": float(loss.item()),
            "mean_kl_to_prior": kl_to_prior, "mean_entropy": entropy_proxy,
            "warhead_any_rate": warhead_any, "thiq_exact_rate": thiq_exact,
        }

    def train(self, csv_log_path):
        logs = []
        with open(csv_log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "n_valid", "n_pairs", "mean_reward",
                        "mean_chosen_reward", "mean_rejected_reward",
                        "reward_gap", "loss", "mean_kl_to_prior", "mean_entropy",
                        "warhead_any_rate", "thiq_exact_rate", "wall_s"])
            t0 = time.time()
            for step_idx in range(self.cfg.outer_steps):
                step_t0 = time.time()
                log = self.step(step_idx)
                dt = time.time() - step_t0
                logger.info(
                    f"[step {log['step']:03d}] R={log['mean_reward']:+.4f} "
                    f"gap={log['reward_gap']:+.4f} war={log['warhead_any_rate']:.3f} "
                    f"thiq={log['thiq_exact_rate']:.3f} kl={log['mean_kl_to_prior']:+.3f} "
                    f"loss={log['loss']:+.4f} pairs={log['n_pairs']} dt={dt:.1f}s")
                w.writerow([log['step'], log['n_valid'], log['n_pairs'],
                            f"{log['mean_reward']:.6f}",
                            f"{log['mean_chosen_reward']:.6f}", f"{log['mean_rejected_reward']:.6f}",
                            f"{log['reward_gap']:.6f}", f"{log['loss']:.6f}",
                            f"{log['mean_kl_to_prior']:.6f}", f"{log['mean_entropy']:.6f}",
                            f"{log['warhead_any_rate']:.6f}", f"{log['thiq_exact_rate']:.6f}",
                            f"{time.time() - t0:.1f}"])
                fh.flush()
                logs.append(log)
        return logs


def load_seed_smiles(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"): continue
            out.append(line.split()[0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True)
    p.add_argument("--smiles_file", required=True)
    p.add_argument("--reward_url", required=True)
    p.add_argument("--reward_predictor_id", default="ladder")
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--k_pairs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_out", required=True)
    p.add_argument("--div_bucket", type=int, default=25)
    p.add_argument("--div_minscore", type=float, default=0.4)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    logger.info(f"Loading prior+agent from {args.prior} on {device}")
    prior, _, mt = create_adapter(args.prior, "inference", device)
    agent, _, _ = create_adapter(args.prior, "inference", device)
    agent_old, _, _ = create_adapter(args.prior, "inference", device)
    assert mt == "Mol2Mol"
    for net in (prior, agent_old):
        for p_ in net.get_network_parameters(): p_.requires_grad = False
    for p_ in agent.get_network_parameters(): p_.requires_grad = True
    smilies = load_seed_smiles(Path(args.smiles_file))
    logger.info(f"Loaded {len(smilies)} seeds")
    scorer = RESTScorer(args.reward_url, predictor_id=args.reward_predictor_id)
    cfg = DPOConfig(beta=args.beta, lr=args.lr, batch_size=args.batch_size,
                    k_pairs=args.k_pairs, outer_steps=args.steps)
    trainer = DPOTrainer(agent, agent_old, prior, scorer, smilies, device, cfg,
                         div_bucket=args.div_bucket, div_minscore=args.div_minscore)
    output_path = Path(args.output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"DPO cfg: {cfg}")
    trainer.train(output_path)
    ckpt_path = Path(args.checkpoint_out); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_path}")
    agent.save_to_file(str(ckpt_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
