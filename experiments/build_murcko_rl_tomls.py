"""Murcko-scaffold-preserving RL: 3 cohorts with Mol1 Murcko + THIQ-acryl rewards.

Reward composition (geometric mean):
  - FiLM pIC50 (REST):                    weight 0.35
  - Mol1 Murcko scaffold match (SMARTS):  weight 0.30
  - THIQ-acrylamide core (SMARTS):        weight 0.25
  - QED:                                  weight 0.10

Mol1 Murcko SMARTS uses non-aromatic-H form for slightly relaxed match:
  `O=C(Nc1cncn1)c1cccc2c1CNC2`  (was `O=C(Nc1c[nH]cn1)c1cccc2c1CNC2`)

Cohorts:
  - murcko_rl_mol1only  : seeds = Mol1 only
  - murcko_rl_zap70     : seeds = 220 ZAP70 + Mol1
  - murcko_rl_kinase    : seeds = 34K kinase + ZAP70x5 + Mol1x20

All use warhead_tokens prior, REST FiLM, batch_size matched to seed pool size.
"""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "experiments" / "murcko_rl_tomls"
REMOTE_HOME = "/home/shaharh_quris_ai"
REMOTE_PROJ = f"{REMOTE_HOME}/edit-small-mol"
FILM_PORT = 8088

# Mol1 Murcko scaffold SMARTS — slightly relaxed (n instead of [nH]) for tautomer tolerance
MOL1_MURCKO_SMARTS = "O=C(Nc1cncn1)c1cccc2c1CNC2"
THIQ_ACRYL_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"

COHORTS = [
    # warhead_tokens (EXP6) base
    {"tag": "murcko_rl_zap70",         "seed": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi", "batch": 8, "prior": "warhead_tokens"},
    {"tag": "murcko_rl_kinase",        "seed": "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20_clean.smi", "batch": 16, "prior": "warhead_tokens"},
    # covalent_ft (EXP2) base — the one with 98.4% baseline THIQ retention on Mol1
    {"tag": "murcko_rl_exp2_zap70",    "seed": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi", "batch": 8, "prior": "covalent_ft"},
    {"tag": "murcko_rl_exp2_kinase",   "seed": "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20_clean.smi", "batch": 16, "prior": "covalent_ft"},
]


def build_toml(tag: str, seed_rel: str, batch: int, prior_kind: str = "warhead_tokens") -> str:
    seed = f"{REMOTE_PROJ}/{seed_rel}"
    prior = f"{REMOTE_PROJ}/models/reinvent4_mol2mol_{prior_kind}.prior"
    return f"""# {tag} — Mol1 Murcko-scaffold-preserving RL
# Reward: FiLM 0.35 + Mol1-Murcko 0.30 + THIQ-acryl 0.25 + QED 0.10

run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_{tag}"
json_out_config = "_{tag}.json"

[parameters]
summary_csv_prefix = "{tag}"
use_checkpoint = false
purge_memories = false

prior_file = "{prior}"
agent_file = "{prior}"
smiles_file = "{seed}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = {batch}
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

[[stage]]
chkpt_file = "{tag}_stage1.chkpt"
termination = "simple"
max_score = 0.7
min_steps = 15
max_steps = 50

[stage.scoring]
type = "geometric_mean"

# 1. FiLMDelta pIC50 (REST)
[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.35
params.server_url = "http://127.0.0.1"
params.server_port = {FILM_PORT}
params.server_endpoint = "score"
params.predictor_id = "film_delta"
params.predictor_version = "v1"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

# 2. Mol1 Murcko scaffold match (the "preserve the whole scaffold" reward)
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "Mol1 Murcko scaffold retained"
weight = 0.30
params.smarts = "{MOL1_MURCKO_SMARTS}"
params.use_chirality = false

# 3. THIQ-acrylamide core
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "THIQ-acrylamide core retained"
weight = 0.25
params.smarts = "{THIQ_ACRYL_SMARTS}"
params.use_chirality = false

# 4. QED
[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in COHORTS:
        p = OUT_DIR / f"{c['tag']}.toml"
        p.write_text(build_toml(c["tag"], c["seed"], c["batch"], c.get("prior", "warhead_tokens")))
        print(f"wrote {p.relative_to(PROJECT)} (batch={c['batch']}, prior={c.get('prior', 'warhead_tokens')})")


if __name__ == "__main__":
    main()
