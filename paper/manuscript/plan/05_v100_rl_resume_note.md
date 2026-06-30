# V100 distilled-RL run — resume note (2026-06-23 night → tomorrow)

## State as of going to sleep

* **V100 (`ai-gpu`, us-central1-c)**: STOPPED (terminated, no cost incurred overnight).
* **Distilled validator model**: trained + saved at `models/boltz_distilled/boltz_distilled_v1.pkl` (local) AND deployed to `/home/shaharh_quris_ai/edit-small-mol/models/boltz_distilled/` on V100. CV r = 0.671 (iptm), 0.692 (mPAE_london).
* **Scoring server**: `experiments/distilled_validator_server.py` written + deployed to V100. Format-fix applied: now accepts REINVENT4's raw-list POST body and returns a raw list of floats.
* **RL config**: `experiments/thiq_rl_tomls/thiq_rl_exp2_zap70_distilled.toml` written + deployed. 4-component reward: FiLMpIC50 (0.45) + THIQ-SMARTS (0.35) + distilled iptm (0.15) + QED (0.05).
* **Run script**: `experiments/thiq_rl_tomls/run_thiq_rl_exp2_zap70_distilled.sh` deployed. Starts FiLM REST on :8088, distilled REST on :8089, waits for health, launches reinvent.

## Bugs surfaced overnight (all noted, two fully fixed, one fix flaky)

### Bug 1 — my scoring-server response format ✅ FIXED
- distilled server originally expected `{smiles:[...]}` and returned `{scores:[...]}` — REINVENT4 sends a raw list and expects a raw list back
- Fix in `experiments/distilled_validator_server.py`: accept raw list OR dict; return raw list. Already redeployed.

### Bug 2 — REINVENT4 source `NameError` ✅ PATCHED ON REMOTE
- File: `/home/shaharh_quris_ai/REINVENT4/reinvent_plugins/components/comp_generic_rest.py:91`
- Original: `f"Component {self.__name__} failed.\n"` inside a module-level `execute_request(url, data, header, params)` function — `self` doesn't exist in this scope
- Result: any non-200 REST response (e.g., from any of our scoring components) was masked by `NameError: name 'self' is not defined`, causing the whole RL run to crash with a misleading traceback
- Patch applied on V100: `sed -i 's|Component {self.__name__} failed|Component at {url} failed|' /home/shaharh_quris_ai/REINVENT4/reinvent_plugins/components/comp_generic_rest.py`
- Backup at `comp_generic_rest.py.bak`
- This patch will let the REAL underlying REST error surface (e.g., "Component at http://127.0.0.1:8089/score failed. Status Code: 500. Reason: ...") so we can debug further

### Bug 3 — tmux relaunch over ssh flaky ⚠ NEEDS DIFFERENT APPROACH TOMORROW
- Pattern tried: `ssh box "tmux new-session -d -s name 'bash script.sh > /tmp/log 2>&1'"`
- Symptoms: sometimes session created and persists, sometimes session created and instantly dies, sometimes session never created. Tmux state inconsistent across ssh probes.
- Likely cause: ssh disconnect propagating to tmux session group despite `-d` detach; possibly IAP throttling making ssh commands stale-state
- **Tomorrow's fix**: try `systemd-run --user --scope -d /bin/bash script.sh` for a cleaner detach. Or just SSH in interactively, start tmux foreground, detach with Ctrl+B D — old-school but reliable.

## Tomorrow's resume sequence (target: ~30 min hands-on, then leave to train)

```bash
# 1. Start V100
gcloud compute instances start ai-gpu --zone=us-central1-c

# 2. Wait ~30s for boot, then ssh in interactively
ssh -i ~/.ssh/google_compute_engine shaharh_quris_ai@$(gcloud compute instances describe ai-gpu --zone=us-central1-c --format='value(networkInterfaces[0].accessConfigs[0].natIP)')

# 3. Verify the REINVENT4 patch is still there
grep -n 'Component at' /home/shaharh_quris_ai/REINVENT4/reinvent_plugins/components/comp_generic_rest.py
# Expected: line 91 — f"Component at {url} failed.\n"

# 4. Sanity test scoring servers FIRST (don't launch RL until both servers respond):
source ~/miniconda3/etc/profile.d/conda.sh && conda activate quris

# Terminal 1: FiLM
python /home/shaharh_quris_ai/edit-small-mol/experiments/reinvent4_film_rest_server.py --host 127.0.0.1 --port 8088
# Terminal 2: distilled
python /home/shaharh_quris_ai/edit-small-mol/experiments/distilled_validator_server.py --host 127.0.0.1 --port 8089
# Terminal 3 (or scripted): test both
curl -X POST -H 'Content-Type: application/json' -d '["C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "CCO"]' http://127.0.0.1:8089/score
curl -X POST -H 'Content-Type: application/json' -d '["C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1", "CCO"]' http://127.0.0.1:8088/score
# Expected: both return a JSON list of 2 floats in [0,1]

# 5. If both scorers work, launch RL inside a real tmux:
tmux new -s distrl
bash /home/shaharh_quris_ai/edit-small-mol/experiments/thiq_rl_tomls/run_thiq_rl_exp2_zap70_distilled.sh
# Ctrl+B D to detach; come back later with `tmux attach -t distrl`

# 6. Expected runtime: ~2-3 hours to converge to max_steps=50

# 7. When chkpt exists at results/.../thiq_rl_exp2_zap70_distilled_stage1.chkpt:
#    - Sample 10K SMILES from the chkpt
#    - Score with distilled + FiLM + basic 2D filters
#    - Compare distribution shift vs thiq_rl_exp2_zap70 baseline (panel B/C of Fig 4 in plan/04_figures.md)

# 8. STOP V100 immediately after:
gcloud compute instances stop ai-gpu --zone=us-central1-c
```

## What the paper draft already references

- §3.4 "Distilled validator demonstration" in `plan/02_outline.md` and Figure 4 in `plan/04_figures.md` already reference this experiment — needs results to populate
- §2.10 in `02_methods.tex` describes the distilled validator with explicit math + deployment details — already written
- §2.11 in `02_methods.tex` describes the 4-component reward variant — written but with weights consistent with this run

## What you can do TODAY for the paper without the V100 result

Plenty:
- Review + push back on `01_introduction.tex` and `02_methods.tex` prose (already complete)
- Draft `03_results.tex` scaffold using the EXISTING 11 RL cohorts + 1684 Boltz cofolds + 47 final picks (the distilled-validator section is just one sub-section of Results)
- Populate `references.bib` and run `sudo /usr/local/texlive/2026basic/bin/universal-darwin/tlmgr install multirow mathtools natbib microtype siunitx listings booktabs geometry` to enable the full preamble
- Discuss and lock figure designs

The V100 cohort is one *additional* methods bullet (a small bonus). The paper stands on the existing data + wet-lab plan regardless.
