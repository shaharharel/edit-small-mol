"""ESM-2 pocket embedding extractor for C+D v2.6.

For each CovInDB2 cocrystal record, computes the mean-pooled ESM-2 embedding
over the pocket residues (those within 8 Å of the Cys Cβ). Caches the
result so the v2.6 dataset can do an O(1) lookup at training time.

Why: v2.5's 20-d residue-count slice gives only bag-of-residues. ESM-2
captures side-chain context (kinase Lys vs nuclear-receptor Lys look
different to ESM-2) at residue resolution. Better cross-family
generalization for the C-arm.

ESM-2-650M (esm2_t33_650M_UR50D) → 1280-d per residue. We mean-pool over
pocket residues → single 1280-d pocket fingerprint per record.

Memory: 650M model is ~2.5 GB on GPU; attention is O(L²). On V100 (16 GB),
proteins > 1024 residues need chunking. CovInDB2 proteins are typically
200-600 residues; chunking is a safety net.

Cache file: data/covbinder/esm2_pocket_embs.npz
  - One key per record_id, value is (1280,) float32 array
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser, NeighborSearch, Selection, Polypeptide
from Bio.PDB.Polypeptide import three_to_one

ESM2_DIM = 1280  # ESM-2-650M hidden dim
DEFAULT_POCKET_CUTOFF = 8.0  # Å — matches dataset.py


def _load_esm():
    """Load ESM-2-650M and its batch converter. Returns (model, alphabet)."""
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


def _pdb_to_sequence_and_resi_map(pdb_path: Path, chain_id: str) -> tuple[str, dict[int, int]]:
    """Return (sequence_str, pdb_resi_to_seq_idx_map).

    Maps each residue's PDB residue number (e.g. 346 for Cys346) to its
    0-based position in the ESM-2 sequence input.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", str(pdb_path))[0]
    if chain_id not in structure:
        return "", {}
    chain = structure[chain_id]
    seq_chars = []
    pdb_to_seq: dict[int, int] = {}
    for res in chain:
        if res.id[0] != " ":  # hetero or water
            continue
        resname = res.get_resname()
        try:
            one = three_to_one(resname)
        except KeyError:
            one = "X"
        pdb_to_seq[res.id[1]] = len(seq_chars)
        seq_chars.append(one)
    return "".join(seq_chars), pdb_to_seq


def _find_pocket_pdb_resi(
    pdb_path: Path, chain_id: str, cys_resi: int, cutoff: float = DEFAULT_POCKET_CUTOFF,
) -> list[int]:
    """Return the PDB residue numbers of residues within `cutoff` of the
    Cys Cβ — same logic as dataset.py's pocket extraction."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", str(pdb_path))[0]
    if chain_id not in structure:
        return []
    chain = structure[chain_id]
    if cys_resi not in [r.id[1] for r in chain]:
        return []
    cys_res = chain[(' ', cys_resi, ' ')] if (' ', cys_resi, ' ') in chain else None
    # robust lookup
    if cys_res is None:
        for r in chain:
            if r.id[1] == cys_resi: cys_res = r; break
    if cys_res is None or "CB" not in cys_res:
        return []
    cb_xyz = cys_res["CB"].get_coord()

    pocket_pdb_resi: set[int] = set()
    # iterate ALL atoms in the structure (other chains too) — pocket can
    # span chains. We'll restrict to the same chain for simplicity here.
    for res in chain:
        if res.id[0] != " ": continue
        for atom in res.get_atoms():
            if atom.element == "H": continue
            d = np.linalg.norm(atom.get_coord() - cb_xyz)
            if d <= cutoff:
                pocket_pdb_resi.add(res.id[1])
                break
    return sorted(pocket_pdb_resi)


@torch.no_grad()
def _run_esm_on_sequence(model, alphabet, sequence: str, device: str = "cuda") -> np.ndarray:
    """Run ESM-2 on a single sequence. Returns (L, ESM2_DIM) per-residue embeddings.

    For sequences > 1024 residues, chunks with 64-residue overlap and merges
    (memory safety on V100).
    """
    if not sequence:
        return np.zeros((0, ESM2_DIM), dtype=np.float32)

    batch_converter = alphabet.get_batch_converter()

    def _embed_chunk(seq_chunk: str) -> np.ndarray:
        _, _, batch_tokens = batch_converter([("p", seq_chunk)])
        if device == "cuda":
            batch_tokens = batch_tokens.cuda()
        out = model(batch_tokens, repr_layers=[33])
        repr33 = out["representations"][33][0]  # (L+2, 1280) with BOS, EOS
        # strip BOS (idx 0) and EOS (idx -1)
        per_residue = repr33[1:-1]  # (L, 1280)
        return per_residue.cpu().numpy().astype(np.float32)

    L = len(sequence)
    if L <= 1024:
        return _embed_chunk(sequence)

    # Chunk with overlap, average overlapping positions
    chunk_size = 1000
    overlap = 64
    step = chunk_size - overlap
    out = np.zeros((L, ESM2_DIM), dtype=np.float32)
    counts = np.zeros(L, dtype=np.float32)
    for start in range(0, L, step):
        end = min(start + chunk_size, L)
        chunk_emb = _embed_chunk(sequence[start:end])
        out[start:end] += chunk_emb
        counts[start:end] += 1.0
        if end == L: break
    out /= counts[:, None].clip(min=1.0)
    return out


def compute_record_pocket_emb(
    model, alphabet, pdb_path: Path, chain_id: str, cys_resi: int,
    cutoff: float = DEFAULT_POCKET_CUTOFF, device: str = "cuda",
) -> np.ndarray | None:
    """Compute the mean-pooled ESM-2 embedding over the pocket residues."""
    sequence, pdb_to_seq = _pdb_to_sequence_and_resi_map(pdb_path, chain_id)
    if not sequence:
        return None
    pocket_pdb_resi = _find_pocket_pdb_resi(pdb_path, chain_id, cys_resi, cutoff)
    if not pocket_pdb_resi:
        return None
    pocket_seq_idx = [pdb_to_seq[r] for r in pocket_pdb_resi if r in pdb_to_seq]
    if not pocket_seq_idx:
        return None
    per_residue = _run_esm_on_sequence(model, alphabet, sequence, device=device)
    pocket_embs = per_residue[pocket_seq_idx]  # (n_pocket, 1280)
    return pocket_embs.mean(axis=0).astype(np.float32)


def build_cache(csv_path: Path, out_npz: Path, limit: int | None = None):
    """Iterate every record in covind_training_set.csv, compute ESM-2 pocket
    embedding, save to NPZ keyed by record_id."""
    print(f"Loading ESM-2-650M…")
    model, alphabet = _load_esm()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    df = pd.read_csv(csv_path)
    print(f"  {len(df)} records to process")
    if limit:
        df = df.head(limit)

    embeddings: dict[str, np.ndarray] = {}
    n_ok = n_skip = 0
    import time
    t0 = time.time()
    for i, r in df.iterrows():
        record_id = str(r["record_id"])
        pdb_path = PROJECT_ROOT / r["pdb_path"]
        if not pdb_path.exists():
            n_skip += 1
            continue
        chain = r["cys_chain"]
        cys_resi = int(r["cys_resi"])
        try:
            emb = compute_record_pocket_emb(
                model, alphabet, pdb_path, chain, cys_resi, device=device,
            )
        except Exception as e:
            print(f"  [{i+1}/{len(df)}] {record_id}: ERROR {type(e).__name__}: {str(e)[:80]}")
            n_skip += 1
            continue
        if emb is None:
            n_skip += 1
            continue
        embeddings[record_id] = emb
        n_ok += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (len(df) - i - 1) / max(rate, 1e-6)
            print(f"  [{i+1}/{len(df)}] ok={n_ok} skip={n_skip}  "
                  f"rate={rate:.1f}/s  eta={eta/60:.1f} min")

    print(f"\nDone. {n_ok} OK, {n_skip} skipped. Saving to {out_npz}…")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), **embeddings)
    print(f"Wrote {len(embeddings)} entries.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "covbinder" / "esm2_pocket_embs.npz"))
    ap.add_argument("--limit", type=int, default=None, help="process only first N for smoke test")
    args = ap.parse_args()
    build_cache(Path(args.csv), Path(args.out), limit=args.limit)


if __name__ == "__main__":
    main()
