"""B2: REINVENT-style particle resampling inside DiffSBDD's reverse-diffusion loop.

This is the orchestrator that drives generation on the T4 VM.

Design:
    The DiffSBDD model + diffusion loop run on T4 (GPU). The FiLMDelta scorer
    *also* runs on T4 (the model is small and avoids back-SSH callbacks).

    Locally we:
      1. Sync `b2_inpaint.py` and `b2_film_scorer.py` to T4 (~/DiffSBDD/).
      2. Sync `covalent_constraint_manifold.py` and `fix_inpaint_warhead_bonds.py`
         to ~/anchordiff/ on T4.
      3. Sync the FiLMDelta model checkpoint to T4 (~/anchordiff/).
      4. Run b2_inpaint.py over SSH with the requested hyperparameters.
      5. Pull resulting SDF + ranking CSV back to local results/.

Usage:
    python anchordiff/diffsbdd_b2_sampler.py \
        --target zap70_cys346 \
        --n_samples 5 --n_particles 4 --resample_every 10 --temperature 0.5

Defaults match the headline-result spec: n_particles=16, resample_every=10,
temperature=0.5, n_samples=32.
"""
from __future__ import annotations
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ANCHORDIFF = PROJECT_ROOT / "anchordiff"
LOCAL_FILM_CKPT = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
LOCAL_RESULTS = PROJECT_ROOT / "results" / "paper_evaluation" / "diffsbdd_b2"

VM = "ai-gpu"
ZONE = "us-east1-c"

# Files to sync
T4_DIFFSBDD = "~/DiffSBDD"
T4_ANCHORDIFF = "~/anchordiff"
T4_RESULTS = "~/anchordiff_results/b2"


def gcloud_ssh(cmd: str, retries: int = 6, sleep_s: int = 4) -> str:
    """Run a command on T4 over gcloud ssh, retrying on transient connection failures."""
    full = ["gcloud", "compute", "ssh", VM, f"--zone={ZONE}", "--command", cmd]
    last_err = None
    for attempt in range(retries):
        try:
            r = subprocess.run(full, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0:
                return r.stdout
            # treat connection-refused / 4003 as transient
            err = (r.stderr or "")[-500:]
            if any(s in err for s in ("Connection refused", "4003", "exited with return code [255]")):
                last_err = err
                time.sleep(sleep_s)
                continue
            raise RuntimeError(f"gcloud ssh failed (rc={r.returncode}): {err}")
        except subprocess.TimeoutExpired:
            last_err = "timeout"
            time.sleep(sleep_s)
    raise RuntimeError(f"gcloud ssh failed after {retries} retries: {last_err}")


def gcloud_scp(local: Path, remote: str) -> None:
    """SCP a local file to T4 (under remote path)."""
    full = ["gcloud", "compute", "scp", str(local), f"{VM}:{remote}", f"--zone={ZONE}"]
    for attempt in range(6):
        r = subprocess.run(full, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            return
        if any(s in (r.stderr or "") for s in ("Connection refused", "4003", "255")):
            time.sleep(4)
            continue
        raise RuntimeError(f"gcloud scp {local} failed: {r.stderr}")
    raise RuntimeError(f"gcloud scp {local} -> {remote} failed after retries")


def gcloud_scp_pull(remote: str, local: Path) -> None:
    full = ["gcloud", "compute", "scp", f"{VM}:{remote}", str(local), f"--zone={ZONE}"]
    for attempt in range(6):
        r = subprocess.run(full, capture_output=True, text=True, timeout=900)
        if r.returncode == 0:
            return
        if any(s in (r.stderr or "") for s in ("Connection refused", "4003", "255")):
            time.sleep(4)
            continue
        raise RuntimeError(f"gcloud scp pull {remote} failed: {r.stderr}")
    raise RuntimeError(f"gcloud scp pull {remote} -> {local} failed after retries")


def sync_files() -> None:
    """Push all needed code + model artifacts to T4."""
    print("=== Syncing files to T4 ===")

    # Make sure remote dirs exist
    gcloud_ssh(f"mkdir -p {T4_DIFFSBDD} {T4_ANCHORDIFF} {T4_RESULTS}")

    # Sync DiffSBDD-side patches
    for fname in ["b2_inpaint.py", "b2_film_scorer.py"]:
        local = LOCAL_ANCHORDIFF / fname
        if not local.exists():
            raise FileNotFoundError(f"missing local file: {local}")
        gcloud_scp(local, f"{T4_DIFFSBDD}/{fname}")
        print(f"  synced {fname} -> {T4_DIFFSBDD}/")

    # Sync anchordiff helpers (constraint projection + warhead bond fix)
    for fname in ["covalent_constraint_manifold.py", "fix_inpaint_warhead_bonds.py"]:
        local = LOCAL_ANCHORDIFF / fname
        gcloud_scp(local, f"{T4_ANCHORDIFF}/{fname}")
        print(f"  synced {fname} -> {T4_ANCHORDIFF}/")

    # Sync FiLMDelta model checkpoint (idempotent — only push if remote missing or
    # local newer)
    out = gcloud_ssh(
        f"[ -f {T4_ANCHORDIFF}/reinvent4_film_model.pt ] && "
        f"stat -c %Y {T4_ANCHORDIFF}/reinvent4_film_model.pt || echo 0"
    ).strip()
    remote_mtime = int(out or "0")
    local_mtime = int(LOCAL_FILM_CKPT.stat().st_mtime)
    if local_mtime > remote_mtime:
        print("  syncing reinvent4_film_model.pt (35 MB)…")
        gcloud_scp(LOCAL_FILM_CKPT, f"{T4_ANCHORDIFF}/reinvent4_film_model.pt")
    else:
        print("  reinvent4_film_model.pt: remote up-to-date")


def run_remote(target: str, n_samples: int, n_particles: int,
               resample_every: int, temperature: float, timesteps: int) -> str:
    """Launch b2_inpaint.py on T4 in a detached nohup session, then poll for
    completion (ranking.csv presence). This is robust to SSH session drops:
    the diffsbdd job keeps running even when our control SSH disconnects."""
    remote_outdir = f"{T4_RESULTS}/{target}_N{n_samples}_P{n_particles}_K{resample_every}_T{temperature}"

    launch_cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || "
        "source /opt/conda/etc/profile.d/conda.sh; "
        "conda activate diffsbdd; "
        f"mkdir -p {remote_outdir}; "
        f"rm -f {remote_outdir}/ranking.csv {remote_outdir}/DONE; "
        f"cd ~/DiffSBDD; "
        "nohup bash -c '"
        "WANDB_DISABLED=true PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python "
        "python b2_inpaint.py "
        f"  ~/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt "
        f"  --pdbfile ~/anchordiff/pockets/{target}/receptor.pdb "
        f"  --ref_ligand ~/anchordiff/pockets/{target}/ref_ligand.sdf "
        f"  --fixed_atoms_file ~/anchordiff/pockets/{target}/fixed_atoms.txt "
        f"  --warhead_sdf ~/anchordiff/pockets/{target}/warhead_at_cys.sdf "
        f"  --film_ckpt ~/anchordiff/reinvent4_film_model.pt "
        f"  --outdir {remote_outdir} "
        f"  --n_samples {n_samples} "
        f"  --n_particles {n_particles} "
        f"  --resample_every {resample_every} "
        f"  --temperature {temperature} "
        f"  --timesteps {timesteps} "
        f"  > {remote_outdir}/run.log 2>&1 ; "
        f"  touch {remote_outdir}/DONE"
        f"' > /dev/null 2>&1 &"
        " echo launched"
    )
    print(f"\n=== Running on T4: {target} (N={n_samples} P={n_particles} K={resample_every} T={temperature}) ===")
    gcloud_ssh(launch_cmd, retries=4)

    # Poll until DONE file appears; tail the log periodically so we see progress.
    print("  polling for completion (tail every 30s)…")
    last_lines = ""
    for poll in range(2 * 60 * 60 // 30):  # up to 2 hours
        try:
            out = gcloud_ssh(
                f"if [ -f {remote_outdir}/DONE ]; then echo __DONE__; "
                f"else tail -3 {remote_outdir}/run.log 2>/dev/null; fi",
                retries=2,
            )
        except RuntimeError:
            time.sleep(8); continue
        if "__DONE__" in out:
            print("  remote run finished.")
            return remote_outdir
        if out.strip() and out != last_lines:
            print("    " + out.strip().splitlines()[-1])
            last_lines = out
        time.sleep(30)
    raise TimeoutError(f"remote run did not finish within polling window")


def fetch_outputs(target: str, remote_outdir: str, run_label: str) -> Path:
    local_outdir = LOCAL_RESULTS / run_label
    local_outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Pulling outputs to {local_outdir} ===")
    for fname in ["mols.sdf", "ranking.csv", "run.log"]:
        try:
            gcloud_scp_pull(f"{remote_outdir}/{fname}", local_outdir / fname)
            print(f"  fetched {fname}")
        except RuntimeError as e:
            print(f"  WARN: could not fetch {fname}: {e}")
    return local_outdir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="zap70_cys346",
                   choices=["zap70_cys346", "btk_cys481"])
    p.add_argument("--n_samples", type=int, default=32)
    p.add_argument("--n_particles", type=int, default=16)
    p.add_argument("--resample_every", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.5,
                   help="softmax temperature in pIC50 units")
    p.add_argument("--timesteps", type=int, default=50,
                   help="DDPM denoising steps")
    p.add_argument("--skip_sync", action="store_true",
                   help="skip syncing files (assume already done)")
    p.add_argument("--smoke", action="store_true",
                   help="smoke-test mode: 5 samples, 4 particles")
    args = p.parse_args()

    if args.smoke:
        # Smoke test: verifies the loop end-to-end. T4 (14.5 GB) cannot fit
        # P=4 particles with pocket-conditioned DiffSBDD; P=2 is the sweet
        # spot for smoke (P=4 OOMs on a 56-residue pocket like ZAP70).
        args.n_samples = 5
        args.n_particles = 2
        args.resample_every = 10
        args.temperature = 0.5
        args.timesteps = 50

    if not args.skip_sync:
        sync_files()

    remote_outdir = run_remote(
        target=args.target,
        n_samples=args.n_samples,
        n_particles=args.n_particles,
        resample_every=args.resample_every,
        temperature=args.temperature,
        timesteps=args.timesteps,
    )

    run_label = f"{args.target}_N{args.n_samples}_P{args.n_particles}_K{args.resample_every}_T{args.temperature}"
    fetch_outputs(args.target, remote_outdir, run_label)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
