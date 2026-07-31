"""Generate the exp2_v2_rl_v2 DAP RL training progress figure for the paper.

Reads the REINVENT4 per-sample log:
  results/paper_evaluation/reinvent4/exp2_v2_rl_v2/exp2_v2_rl_v2_1.csv

Emits:
  paper/manuscript/fig/exp2_rl_training.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV = Path(
    "/Users/shaharharel/Documents/github/edit-small-mol/results/paper_evaluation/"
    "reinvent4/exp2_v2_rl_v2/exp2_v2_rl_v2_1.csv"
)
OUT = Path(
    "/Users/shaharharel/Documents/github/edit-small-mol/paper/manuscript/fig/"
    "exp2_rl_training.pdf"
)


def main() -> None:
    df = pd.read_csv(CSV)
    # Restrict to samples that were actually scored (invalid SMILES => Score==0).
    valid = df[df["Score"] > 0].copy()
    agg = (
        valid.groupby("step")[
            ["Score", "FiLMDelta pIC50", "Acrylamide retained", "QED"]
        ]
        .mean()
        .reset_index()
    )

    # Light smoothing so the trajectory is legible without hiding dynamics.
    smoothed = agg.copy()
    for col in ["Score", "FiLMDelta pIC50", "Acrylamide retained", "QED"]:
        smoothed[col] = agg[col].rolling(window=3, min_periods=1, center=True).mean()

    fig, ax = plt.subplots(figsize=(5, 3))

    ax.plot(
        smoothed["step"],
        smoothed["Score"],
        color="#1a1a1a",
        linewidth=2.4,
        label="Composite reward",
    )
    ax.plot(
        smoothed["step"],
        smoothed["FiLMDelta pIC50"],
        color="#1f77b4",
        linewidth=1.6,
        label="FiLMDelta pIC50",
    )
    ax.plot(
        smoothed["step"],
        smoothed["Acrylamide retained"],
        color="#d62728",
        linewidth=1.6,
        label="Warhead SMARTS",
    )
    ax.plot(
        smoothed["step"],
        smoothed["QED"],
        color="#2ca02c",
        linewidth=1.6,
        label="QED",
    )

    ax.set_xlabel("RL step")
    ax.set_ylabel("Component score")
    ax.set_xlim(1, agg["step"].max())
    ax.set_ylim(0.0, 1.02)
    ax.grid(False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="center right", frameon=False, fontsize=8)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    finals = smoothed.iloc[-1]
    print(f"saved {OUT}")
    print(
        f"final (step {int(finals['step'])}): "
        f"composite={finals['Score']:.3f}, "
        f"pIC50={finals['FiLMDelta pIC50']:.3f}, "
        f"warhead={finals['Acrylamide retained']:.3f}, "
        f"QED={finals['QED']:.3f}"
    )


if __name__ == "__main__":
    main()
