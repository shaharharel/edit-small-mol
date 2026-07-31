"""QA #3 (2026-06-24): summarize all PPO/DPO RL variants from log CSVs.

Inputs (local copies, scp'd from a100-b earlier or available):
  - data/exp_ppo/ppo_zap70_log.csv (v1 collapsed)
  - data/exp_ppo_v2plus/*.csv (v2, v2b, v3, v4, v5, v6, dpo_v1)  <-- scp from a100-b

This script:
  - If logs don't exist locally, scp them down from a100-b.
  - Parses each log: extract n_steps, final mean_reward, max mean_reward, mean kl,
    final entropy, warhead retention fraction (if present), final FiLM score.
  - Emits a JSON + Markdown comparison table.
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
LOCAL_DIR = PROJECT_ROOT / "data" / "exp_ppo_v2plus"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

REMOTE_HOST = "ai-gpu-a100-b"
REMOTE_ZONE = "us-central1-b"
REMOTE_BASE = "/home/shaharh_quris_ai/edit-small-mol/data/exp_ppo_v2plus"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scp_from_a100b(remote_path, local_path):
    cmd = [
        "gcloud", "compute", "scp",
        "--zone", REMOTE_ZONE,
        "--tunnel-through-iap",
        f"{REMOTE_HOST}:{remote_path}",
        str(local_path),
    ]
    log(f"  scp {remote_path}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        log(f"    scp failed: {r.stderr[:300]}")
        return False
    return True


def fetch_all():
    files = [
        "ppo_v2_log.csv", "ppo_v2b_log.csv", "ppo_v3_log.csv",
        "ppo_v4_log.csv", "ppo_v5_log.csv", "ppo_v6_log.csv",
        "dpo_v1_log.csv",
        "ppo_iteration_log.md",
        "cohort_prior_baseline.csv.summary.json",
        "cohort_v2b.csv.summary.json",
    ]
    for f in files:
        local = LOCAL_DIR / f
        if local.exists() and (time.time() - local.stat().st_mtime) < 600:
            continue  # fresh
        scp_from_a100b(f"{REMOTE_BASE}/{f}", local)


def summarize_log(path: Path) -> dict:
    if not path.exists():
        return {"label": path.stem, "error": "missing"}
    df = pd.read_csv(path)
    s = {"label": path.stem, "n_steps": len(df)}
    # KL / entropy
    if "mean_kl_to_prior" in df.columns:
        s["kl_final"] = float(df["mean_kl_to_prior"].iloc[-1])
        s["kl_max"] = float(df["mean_kl_to_prior"].abs().max())
    if "mean_entropy" in df.columns:
        s["entropy_final"] = float(df["mean_entropy"].iloc[-1])
    # Reward
    if "mean_reward" in df.columns:
        s["reward_mean"] = float(df["mean_reward"].mean())
        s["reward_final"] = float(df["mean_reward"].iloc[-1])
        s["reward_max"] = float(df["mean_reward"].max())
    # Warhead retention (column "warhead_retention" or similar)
    for col in df.columns:
        cl = col.lower()
        if "warhead" in cl or "thiq" in cl:
            v = df[col].astype(float).dropna()
            if len(v):
                s[col + "_mean"] = float(v.mean())
                s[col + "_final"] = float(v.iloc[-1])
                s[col + "_max"] = float(v.max())
    # FiLM
    for col in df.columns:
        cl = col.lower()
        if "film" in cl:
            v = df[col].astype(float).dropna()
            if len(v):
                s[col + "_mean"] = float(v.mean())
                s[col + "_final"] = float(v.iloc[-1])
    # n_valid
    if "n_valid" in df.columns:
        s["n_valid_final"] = int(df["n_valid"].iloc[-1])
    if "wall_s" in df.columns:
        s["wall_s_final"] = float(df["wall_s"].iloc[-1])
    return s


def main():
    log("=== QA #3: PPO + DPO summary ===")
    fetch_all()

    # local v1
    logs = [
        PROJECT_ROOT / "data" / "exp_ppo" / "ppo_zap70_log.csv",
    ]
    for f in ["ppo_v2_log.csv", "ppo_v2b_log.csv", "ppo_v3_log.csv",
              "ppo_v4_log.csv", "ppo_v5_log.csv", "ppo_v6_log.csv",
              "dpo_v1_log.csv"]:
        logs.append(LOCAL_DIR / f)

    out = []
    for p in logs:
        log(f"summarize {p.name}")
        s = summarize_log(p)
        out.append(s)
        log(f"  {p.stem}: n_steps={s.get('n_steps')} reward_final={s.get('reward_final')} "
            f"kl_final={s.get('kl_final')}")

    OUT = OUT_DIR / "exp4_ppo_dpo_summary.json"
    OUT.write_text(json.dumps({"variants": out}, indent=2))
    log(f"Wrote {OUT}")

    # markdown
    lines = ["# Exp 4 PPO/DPO RL iteration — log-derived summary\n"]
    lines.append("| Variant | n_steps | reward_final | reward_max | KL_final | KL_max | ent_final | n_valid_final | wall_s_final |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in out:
        if r.get("error"):
            continue
        lines.append(
            f"| {r['label']} | {r.get('n_steps')} | "
            f"{r.get('reward_final', '—'):.4f} | {r.get('reward_max', '—'):.4f} | "
            f"{r.get('kl_final', '—'):.3f} | {r.get('kl_max', '—'):.3f} | "
            f"{r.get('entropy_final', '—'):.3f} | "
            f"{r.get('n_valid_final', '—')} | {r.get('wall_s_final', '—'):.0f} |"
            if not isinstance(r.get('reward_final'), str) else
            f"| {r['label']} | {r.get('n_steps')} | error |"
        )
    lines.append("\n## Per-variant warhead / FiLM tracking (where logged)\n")
    for r in out:
        if r.get("error"):
            continue
        warhead_keys = [k for k in r if 'warhead' in k.lower() or 'thiq' in k.lower()]
        film_keys = [k for k in r if 'film' in k.lower()]
        if not warhead_keys and not film_keys:
            continue
        lines.append(f"### {r['label']}")
        for k in warhead_keys + film_keys:
            v = r[k]
            try:
                lines.append(f"- {k}: {v:.4f}")
            except Exception:
                lines.append(f"- {k}: {v}")
        lines.append("")
    md = OUT_DIR / "exp4_ppo_dpo_summary.md"
    md.write_text("\n".join(lines))
    log(f"Wrote {md}")
    log("DONE")


if __name__ == "__main__":
    main()
