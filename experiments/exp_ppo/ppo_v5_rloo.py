"""PPO-v5: RLOO (REINFORCE Leave-One-Out, Ahmadian 2024).

Like GRPO but the advantage is (R_i - mean of OTHER samples in group), no std
normalization. RLOO uses uniform group baseline.

Diffs vs v3 (GRPO):
  * advantage = R_i - mean(R_{j != i in same prompt group})   (NOT std-normalized)
  * Otherwise identical: SMARTS ladder reward + Murcko diversity + KL=0.04
"""
from __future__ import annotations
import argparse, csv, logging, sys, time, warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
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
logger = logging.getLogger("ppo_v5_rloo")


class RESTScorer:
    def __init__(self, url, predictor_id="ladder", predictor_version="v5", inp_fmt="smiles", timeout=600.0):
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


def sample_grpo_batch(agent_old, input_pool, n_prompts, group_size, device, randomize=True):
    seed_smilies = list(np.random.choice(input_pool, size=n_prompts, replace=True))
    rep_inputs = []; prompt_ids = []
    for pid, sm in enumerate(seed_smilies):
        for _ in range(group_size):
            rep_inputs.append(sm); prompt_ids.append(pid)
    proc = _standardize_smiles_list(rep_inputs, randomize=randomize)
    tokenizer = SMILESTokenizer()
    dataset = Dataset(proc, agent_old.get_vocabulary(), tokenizer)
    loader = tud.DataLoader(dataset, batch_size=n_prompts * group_size,
                            shuffle=False, collate_fn=Dataset.collate_fn)
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
    return inputs_all, outputs_all, torch.cat(nlls_all, dim=0).to(device), np.array(prompt_ids)


def compute_new_log_probs(agent, inputs, outputs, device, requires_grad=True):
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


def compute_rloo_advantages(rewards: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """A_i = R_i - mean(R_{j != i in same group}). For group_size>=2."""
    adv = np.zeros_like(rewards, dtype=np.float32)
    for pid in np.unique(prompt_ids):
        mask = prompt_ids == pid
        idxs = np.where(mask)[0]
        rg = rewards[mask]
        s = float(np.sum(rg))
        n = len(rg)
        for k, i in enumerate(idxs):
            if n > 1:
                loo_mean = (s - rg[k]) / (n - 1)
            else:
                loo_mean = float(rg[k])
            adv[i] = rg[k] - loo_mean
    return adv


@dataclass
class RLOOConfig:
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    alpha_ent: float = 0.001
    ppo_epochs: int = 2
    lr: float = 5e-6
    n_prompts: int = 8
    group_size: int = 8
    outer_steps: int = 50


class RLOOTrainer:
    def __init__(self, agent, agent_old, prior, scorer, input_pool, device, cfg,
                 div_bucket=15, div_minscore=0.4):
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
            if n >= self.div_bucket: out[i] = 0.0
        return out

    def step(self, step_idx):
        inputs, outputs, old_nlls, prompt_ids = sample_grpo_batch(
            self.agent_old, self.input_pool, self.cfg.n_prompts, self.cfg.group_size,
            self.device, randomize=True)
        old_log_probs = (-old_nlls).detach()
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
        advantages = compute_rloo_advantages(rewards, prompt_ids)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            prior_lp = compute_new_log_probs(self.prior, inputs, outputs, self.device, requires_grad=False)
        last_loss = float("nan"); clip_frac_acc = kl_acc = ent_acc = 0.0; n_epochs_run = 0
        for k in range(self.cfg.ppo_epochs):
            self.optimizer.zero_grad()
            new_lp = compute_new_log_probs(self.agent, inputs, outputs, self.device, requires_grad=True)
            finite_mask = torch.isfinite(new_lp) & torch.isfinite(old_log_probs)
            if finite_mask.sum().item() == 0:
                logger.warning(f"[step {step_idx} ep {k}] no finite log-probs"); continue
            new_lp_f = new_lp[finite_mask]; old_lp_f = old_log_probs[finite_mask]
            adv_f = adv_t[finite_mask]; prior_lp_f = prior_lp[finite_mask]
            ratio = torch.exp(new_lp_f - old_lp_f)
            unclipped = ratio * adv_f
            clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * adv_f
            l_clip = -torch.mean(torch.min(unclipped, clipped))
            kl = torch.mean(new_lp_f - prior_lp_f)
            ent = -torch.mean(new_lp_f)
            loss = l_clip + self.cfg.beta_kl * kl - self.cfg.alpha_ent * ent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.get_network_parameters(), max_norm=1.0)
            self.optimizer.step()
            with torch.no_grad():
                clipped_mask = (ratio < 1.0 - self.cfg.clip_eps) | (ratio > 1.0 + self.cfg.clip_eps)
                clip_frac_acc += float(clipped_mask.float().mean().item())
                kl_acc += float(kl.item()); ent_acc += float(ent.item())
            last_loss = float(loss.item()); n_epochs_run += 1
        self._swap_old_to_new(); self._clear_attn_refs()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        denom = max(1, n_epochs_run)
        return {
            "step": step_idx, "n_valid": n_valid,
            "mean_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
            "mean_advantage": float(np.mean(advantages)) if len(advantages) else 0.0,
            "mean_kl_to_prior": kl_acc / denom,
            "mean_entropy": ent_acc / denom,
            "clip_fraction": clip_frac_acc / denom,
            "loss": last_loss,
            "warhead_any_rate": warhead_any,
            "thiq_exact_rate": thiq_exact,
        }

    def train(self, csv_log_path):
        logs = []
        with open(csv_log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "n_valid", "mean_reward", "mean_advantage",
                        "mean_kl_to_prior", "mean_entropy", "clip_fraction",
                        "loss", "warhead_any_rate", "thiq_exact_rate", "wall_s"])
            t0 = time.time()
            for step_idx in range(self.cfg.outer_steps):
                step_t0 = time.time()
                log = self.step(step_idx)
                dt = time.time() - step_t0
                logger.info(
                    f"[step {log['step']:03d}] R={log['mean_reward']:+.4f} war={log['warhead_any_rate']:.3f} "
                    f"thiq={log['thiq_exact_rate']:.3f} kl={log['mean_kl_to_prior']:+.3f} "
                    f"ent={log['mean_entropy']:.3f} loss={log['loss']:+.3f} dt={dt:.1f}s")
                w.writerow([log['step'], log['n_valid'],
                            f"{log['mean_reward']:.6f}", f"{log['mean_advantage']:.6f}",
                            f"{log['mean_kl_to_prior']:.6f}", f"{log['mean_entropy']:.6f}",
                            f"{log['clip_fraction']:.6f}", f"{log['loss']:.6f}",
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
    p.add_argument("--n_prompts", type=int, default=8)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--ppo_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.04)
    p.add_argument("--alpha_ent", type=float, default=0.001)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_out", required=True)
    p.add_argument("--div_bucket", type=int, default=15)
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
    cfg = RLOOConfig(clip_eps=args.clip_eps, beta_kl=args.beta_kl,
                     alpha_ent=args.alpha_ent, ppo_epochs=args.ppo_epochs, lr=args.lr,
                     n_prompts=args.n_prompts, group_size=args.group_size, outer_steps=args.steps)
    trainer = RLOOTrainer(agent, agent_old, prior, scorer, smilies, device, cfg,
                          div_bucket=args.div_bucket, div_minscore=args.div_minscore)
    output_path = Path(args.output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"RLOO cfg: {cfg}")
    trainer.train(output_path)
    ckpt_path = Path(args.checkpoint_out); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_path}")
    agent.save_to_file(str(ckpt_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
