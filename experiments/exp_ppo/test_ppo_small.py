"""End-to-end smoke test of the PPO loop.

Uses an in-process dummy scorer (`reward = len(smiles) / 50.0`) so the test
runs without the FiLM REST server. Runs 5 outer PPO batches at batch_size=4
on CPU. Verifies:

  * model loads
  * sampling produces SMILES
  * forward-with-grad through agent works (log-probs are finite)
  * Adam step actually runs (loss is finite, parameters change)
  * CSV log is written with 5 rows
  * checkpoint is saveable + reloadable

Wall time on a 2024 MacBook Pro / M3 Pro: ~3-5 minutes.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Make sibling import work
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ppo_mol2mol as P  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DEFAULT = PROJECT_ROOT / "models" / "reinvent4_mol2mol_covalent_ft.prior"
SEED_DEFAULT = PROJECT_ROOT / "data" / "mol1_rl_seeds" / "seed_zap70_all_plus_mol1.smi"


def dummy_scorer(smiles_list):
    """Toy reward: normalized SMILES length, capped to [0, 1].

    Encourages slightly longer outputs without runaway. Whatever — we just need
    nonzero gradient signal through the PPO loop.
    """
    return np.clip(np.array([len(s) for s in smiles_list], dtype=np.float32) / 50.0, 0.0, 1.0)


def main():
    prior_path = Path(os.environ.get("PRIOR", PRIOR_DEFAULT))
    seed_path = Path(os.environ.get("SMILES_FILE", SEED_DEFAULT))

    assert prior_path.exists(), f"prior not found: {prior_path}"
    assert seed_path.exists(), f"seed SMILES not found: {seed_path}"

    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device("cpu")  # force CPU for smoke test
    print(f"[smoke] device={device} prior={prior_path}")

    prior, _, model_type = P.create_adapter(str(prior_path), "inference", device)
    agent, _, _ = P.create_adapter(str(prior_path), "inference", device)
    agent_old, _, _ = P.create_adapter(str(prior_path), "inference", device)
    assert model_type == "Mol2Mol", f"Expected Mol2Mol, got {model_type}"

    # Make sure prior is frozen and agent is trainable.
    for net in (prior, agent_old):
        for p_ in net.get_network_parameters():
            p_.requires_grad = False
    for p_ in agent.get_network_parameters():
        p_.requires_grad = True

    seeds = P.load_seed_smiles(seed_path)[:200]  # subsample for speed
    assert len(seeds) > 0, "no seeds loaded"
    print(f"[smoke] loaded {len(seeds)} seed SMILES")

    cfg = P.PPOConfig(
        clip_eps=0.2,
        beta_kl=0.01,
        alpha_ent=0.001,
        ppo_epochs=2,
        lr=1e-5,
        batch_size=4,
        outer_steps=5,
        baseline_window=64,
    )

    scorer = P.CallableScorer(dummy_scorer)
    trainer = P.PPOTrainer(agent, agent_old, prior, scorer, seeds, device, cfg)

    # Snapshot a parameter so we can confirm it MOVED after training.
    snap_name, snap_before = None, None
    for name, p_ in agent.network.named_parameters():
        if p_.requires_grad and p_.dim() >= 2:
            snap_name = name
            snap_before = p_.detach().clone()
            break
    assert snap_before is not None, "no trainable parameter found"

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "ppo_smoke_log.csv"
        logs = trainer.train(log_path)
        # log has been written
        assert log_path.exists()
        with open(log_path) as fh:
            lines = fh.readlines()
        assert len(lines) == 1 + cfg.outer_steps, (
            f"expected {1 + cfg.outer_steps} CSV lines, got {len(lines)}"
        )
        print(f"[smoke] CSV log OK: {len(lines)} lines")

        # parameter actually changed
        snap_after = dict(agent.network.named_parameters())[snap_name].detach()
        delta = (snap_after - snap_before).abs().mean().item()
        assert delta > 0.0, f"parameter {snap_name} did not change (delta={delta})"
        print(f"[smoke] parameter {snap_name} changed by mean |delta|={delta:.3e}")

        # checkpoint save + reload
        ckpt_path = Path(tmpdir) / "ppo_smoke.prior"
        agent.save_to_file(str(ckpt_path))
        assert ckpt_path.exists() and ckpt_path.stat().st_size > 1_000_000, (
            "checkpoint too small or missing"
        )
        reloaded, _, _ = P.create_adapter(str(ckpt_path), "inference", device)
        assert reloaded is not None
        print(f"[smoke] checkpoint round-trip OK ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

        # all losses finite
        losses = [lg.loss for lg in logs]
        assert all(np.isfinite(l) for l in losses), f"non-finite loss in {losses}"
        rewards = [lg.mean_reward for lg in logs]
        assert all(np.isfinite(r) for r in rewards), f"non-finite reward in {rewards}"
        print(f"[smoke] losses: {[f'{l:+.3f}' for l in losses]}")
        print(f"[smoke] rewards: {[f'{r:+.3f}' for r in rewards]}")

    print("[smoke] PASS")


if __name__ == "__main__":
    main()
