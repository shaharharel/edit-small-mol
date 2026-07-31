"""PPO-v6: combined SOTA candidate.
  GRPO + SMARTS-ladder reward + Murcko diversity + replay buffer (FREED-style)

Key ingredients:
  * GRPO (group-relative advantages, no value head) — Shao 2024
  * SMARTS-ladder reward + Tanimoto tail (server-side)
  * Murcko bucket diversity filter (bucket=15, minscore=0.4)
  * Replay buffer: keep top-256 best-reward warhead-positive samples,
    inject 25% of each prompt-group from replay (anchored to same prompt
    if available — falls back to global replay).
  * Stronger KL anchoring (beta_kl=0.04) than v3 default

Designed for stability + sample efficiency in low-warhead-prevalence regime.
"""
from __future__ import annotations
import argparse, csv, heapq, logging, sys, time, warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

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
logger = logging.getLogger("ppo_v6_combo")


class RESTScorer:
    def __init__(self, url, predictor_id="ladder", predictor_version="v6", inp_fmt="smiles", timeout=600.0):
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


def _has_warhead(smi: str) -> bool:
    from rdkit import Chem
    if not smi: return False
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    for p in [
        Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1"),
        Chem.MolFromSmarts("C(=O)C=C[#7]"),
        Chem.MolFromSmarts("C=CC(=O)N"),
        Chem.MolFromSmarts("[CX3]=[CX3][CX3]=[OX1]"),
    ]:
        if p is not None and m.HasSubstructMatch(p): return True
    return False


def sample_grpo_with_replay(agent_old, input_pool, n_prompts, group_size, device,
                             replay_buffer, replay_frac=0.25, randomize=True):
    """Sample G completions per prompt for n_prompts prompts, with replay injection.

    Replay items are existing (input_smi, output_smi, nll, reward) tuples; we
    splice them into the produced batch. To keep the gradient pipeline simple,
    we recompute NLL under the agent for replay outputs in the inner PPO loop
    (their stored NLL stands as old_log_prob proxy — but we use a stale-aware
    weighting by also re-injecting reward unchanged).

    Returns: inputs, outputs, old_log_probs (Tensor), prompt_ids (np.array), is_replay (np.array bool)
    """
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
    nlls_t = torch.cat(nlls_all, dim=0).to(device)
    is_replay = np.zeros(len(outputs_all), dtype=bool)
    rewards_overlay = np.full(len(outputs_all), np.nan, dtype=np.float32)

    # Inject replay: replace last replay_frac of each group with replay items
    if replay_buffer and replay_frac > 0:
        n_per_group_replay = max(1, int(round(group_size * replay_frac)))
        # pick replay tuples (uniform from buffer); slot them into last entries of each group
        for pid in range(n_prompts):
            grp_idx = [i for i in range(len(prompt_ids)) if prompt_ids[i] == pid]
            slot_idx = grp_idx[-n_per_group_replay:]
            for s in slot_idx:
                rb = replay_buffer[np.random.randint(len(replay_buffer))]
                # rb = (reward, counter, input_smi, output_smi, nll_old) — see V6Trainer._update_replay
                inputs_all[s] = rb[2]
                outputs_all[s] = rb[3]
                nlls_t[s] = torch.tensor(float(rb[4]), device=device)
                is_replay[s] = True
                rewards_overlay[s] = float(rb[0])
    return inputs_all, outputs_all, nlls_t, np.array(prompt_ids), is_replay, rewards_overlay


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


def compute_group_advantages(rewards, prompt_ids, eps=1e-6):
    adv = np.zeros_like(rewards, dtype=np.float32)
    for pid in np.unique(prompt_ids):
        mask = prompt_ids == pid
        rg = rewards[mask]
        mu = float(np.mean(rg)); sd = float(np.std(rg))
        adv[mask] = (rg - mu) / (sd + eps)
    return adv


@dataclass
class V6Config:
    clip_eps: float = 0.2
    beta_kl: float = 0.04
    alpha_ent: float = 0.005
    ppo_epochs: int = 2
    lr: float = 5e-6
    n_prompts: int = 8
    group_size: int = 8
    outer_steps: int = 50
    replay_size: int = 256
    replay_frac: float = 0.25
    replay_min_reward: float = 0.20


class V6Trainer:
    def __init__(self, agent, agent_old, prior, scorer, input_pool, device, cfg,
                 div_bucket=15, div_minscore=0.4):
        self.agent = agent; self.agent_old = agent_old; self.prior = prior
        self.scorer = scorer; self.input_pool = input_pool; self.device = device; self.cfg = cfg
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=cfg.lr)
        self.scaffold_counts: Counter = Counter()
        self.div_bucket = int(div_bucket); self.div_minscore = float(div_minscore)
        # Min-heap of (reward, idx, input_smi, output_smi, nll_old)
        # heap is min-heap so smallest reward at top → replace when full
        self.replay_heap: List[Tuple[float, int, str, str, float]] = []
        self._replay_counter = 0

    def _swap_old_to_new(self):
        self.agent_old.network.load_state_dict(self.agent.network.state_dict())
        for p in self.agent_old.get_network_parameters(): p.requires_grad = False

    def _clear_attn_refs(self):
        for model in (self.agent, self.agent_old, self.prior):
            net = getattr(model, "network", None)
            if net is None: continue
            for m in net.modules():
                if hasattr(m, "attn"): m.attn = None

    def _apply_diversity(self, smiles_list, rewards, is_replay):
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        out = rewards.copy()
        for i, (s, r) in enumerate(zip(smiles_list, rewards)):
            if r < self.div_minscore: continue
            if is_replay[i]: continue  # don't penalize replay (its diversity already paid)
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

    def _update_replay(self, inputs, outputs, rewards, nlls, is_replay):
        """Add warhead-positive, high-reward, non-replay samples to the heap."""
        for i, (inp, out, r, nll) in enumerate(zip(inputs, outputs, rewards, nlls.cpu().numpy())):
            if is_replay[i]: continue
            if r < self.cfg.replay_min_reward: continue
            if not _has_warhead(out): continue
            self._replay_counter += 1
            entry = (float(r), self._replay_counter, inp, out, float(nll))
            if len(self.replay_heap) < self.cfg.replay_size:
                heapq.heappush(self.replay_heap, entry)
            else:
                heapq.heappushpop(self.replay_heap, entry)

    @property
    def replay_buffer(self):
        return self.replay_heap

    def step(self, step_idx):
        inputs, outputs, old_nlls, prompt_ids, is_replay, rewards_overlay = \
            sample_grpo_with_replay(self.agent_old, self.input_pool,
                                    self.cfg.n_prompts, self.cfg.group_size, self.device,
                                    self.replay_buffer, self.cfg.replay_frac, randomize=True)
        old_log_probs = (-old_nlls).detach()
        valid_mask_list = _validate_smiles(outputs)
        n_valid = int(sum(valid_mask_list))
        rewards = np.zeros(len(outputs), dtype=np.float32)
        # Score only non-replay valid; replay uses stored reward
        to_score_idx = [i for i, ok in enumerate(valid_mask_list) if ok and not is_replay[i]]
        if to_score_idx:
            smis = [outputs[i] for i in to_score_idx]
            scores = self.scorer(smis)
            for j, i in enumerate(to_score_idx):
                rewards[i] = scores[j]
        # Fill replay rewards from overlay
        for i in range(len(outputs)):
            if is_replay[i] and not np.isnan(rewards_overlay[i]):
                rewards[i] = rewards_overlay[i]
        rewards = self._apply_diversity(outputs, rewards, is_replay)
        warhead_any = _warhead_match_rate(outputs)
        thiq_exact = _thiq_exact_rate(outputs)
        n_replay = int(is_replay.sum())
        replay_size = len(self.replay_buffer)

        advantages = compute_group_advantages(rewards, prompt_ids)
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
            new_lp_f = new_lp[finite_mask]
            old_lp_f = old_log_probs[finite_mask]
            adv_f = adv_t[finite_mask]
            prior_lp_f = prior_lp[finite_mask]

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

        self._update_replay(inputs, outputs, rewards, old_nlls, is_replay)
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
            "n_replay": n_replay,
            "replay_size": replay_size,
        }

    def train(self, csv_log_path):
        logs = []
        with open(csv_log_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "n_valid", "mean_reward", "mean_advantage",
                        "mean_kl_to_prior", "mean_entropy", "clip_fraction",
                        "loss", "warhead_any_rate", "thiq_exact_rate",
                        "n_replay", "replay_size", "wall_s"])
            t0 = time.time()
            for step_idx in range(self.cfg.outer_steps):
                step_t0 = time.time()
                log = self.step(step_idx)
                dt = time.time() - step_t0
                logger.info(
                    f"[step {log['step']:03d}] R={log['mean_reward']:+.4f} war={log['warhead_any_rate']:.3f} "
                    f"thiq={log['thiq_exact_rate']:.3f} kl={log['mean_kl_to_prior']:+.3f} "
                    f"ent={log['mean_entropy']:.3f} replay={log['n_replay']}/{log['replay_size']} "
                    f"loss={log['loss']:+.3f} dt={dt:.1f}s")
                w.writerow([log['step'], log['n_valid'],
                            f"{log['mean_reward']:.6f}", f"{log['mean_advantage']:.6f}",
                            f"{log['mean_kl_to_prior']:.6f}", f"{log['mean_entropy']:.6f}",
                            f"{log['clip_fraction']:.6f}", f"{log['loss']:.6f}",
                            f"{log['warhead_any_rate']:.6f}", f"{log['thiq_exact_rate']:.6f}",
                            log['n_replay'], log['replay_size'],
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
    p.add_argument("--alpha_ent", type=float, default=0.005)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_out", required=True)
    p.add_argument("--div_bucket", type=int, default=15)
    p.add_argument("--div_minscore", type=float, default=0.4)
    p.add_argument("--replay_size", type=int, default=256)
    p.add_argument("--replay_frac", type=float, default=0.25)
    p.add_argument("--replay_min_reward", type=float, default=0.2)
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
    cfg = V6Config(clip_eps=args.clip_eps, beta_kl=args.beta_kl,
                   alpha_ent=args.alpha_ent, ppo_epochs=args.ppo_epochs, lr=args.lr,
                   n_prompts=args.n_prompts, group_size=args.group_size, outer_steps=args.steps,
                   replay_size=args.replay_size, replay_frac=args.replay_frac,
                   replay_min_reward=args.replay_min_reward)
    trainer = V6Trainer(agent, agent_old, prior, scorer, smilies, device, cfg,
                        div_bucket=args.div_bucket, div_minscore=args.div_minscore)
    output_path = Path(args.output).resolve(); output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"v6 cfg: {cfg}")
    trainer.train(output_path)
    ckpt_path = Path(args.checkpoint_out); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving checkpoint to {ckpt_path}")
    agent.save_to_file(str(ckpt_path))
    logger.info("Done.")


if __name__ == "__main__":
    main()
