#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) for Molecular Generation.

Supports multiple model backends:
  - reinvent_mol2mol: REINVENT4 Mol2Mol Transformer (default)
  - molt5: MolT5-small (T5-based, 60M params, from laituan245/molt5-small)
  - smiles_gpt2: GPT-2 with character-level SMILES tokenization (124M params)

All backends are scored by FiLMDelta anchor-based pIC50 predictions for ZAP70.

Pipeline:
  Phase 1: Generate preference data from prior + FiLMDelta scoring
  Phase 2: DPO training (pi_theta vs pi_ref)
  Phase 3: Generate from trained policy, evaluate, compare with DAP baseline

Usage:
    # Default (REINVENT4 Mol2Mol):
    conda run --no-capture-output -n quris python experiments/run_dpo_generation.py

    # MolT5-small (requires: pip install transformers sentencepiece):
    conda run --no-capture-output -n quris python experiments/run_dpo_generation.py --model molt5

    # SMILES-GPT2 (requires: pip install transformers):
    conda run --no-capture-output -n quris python experiments/run_dpo_generation.py --model smiles_gpt2

Reference:
    Rafailov et al., "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model", NeurIPS 2023.
"""

import abc
import argparse
import sys
import os
import gc
import json
import copy
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent / "REINVENT4"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPOConfig:
    """All hyperparameters for the DPO pipeline."""

    # Paths
    prior_path: str = str(
        PROJECT_ROOT.parent / "REINVENT4" / "priors" / "mol2mol_medium_similarity.prior"
    )
    actives_path: str = str(PROJECT_ROOT / "data" / "zap70_top_actives_clean.smi")
    results_dir: str = str(
        PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_dpo"
    )

    # Model backend: 'reinvent_mol2mol', 'molt5', 'smiles_gpt2'
    model_backend: str = "reinvent_mol2mol"

    # Device
    device: str = "mps"

    # Phase 1: preference data generation
    n_samples_per_source: int = 50
    temperature_sample: float = 1.0
    sample_batch_size: int = 64
    preference_margin: float = 0.5  # min score gap (pIC50) for a valid pair

    # Phase 2: DPO training
    beta_dpo: float = 0.1
    lr: float = 1e-5
    weight_decay: float = 0.0
    n_epochs: int = 8
    train_batch_size: int = 32
    grad_clip: float = 1.0
    val_fraction: float = 0.1

    # Phase 3: evaluation
    n_eval_samples: int = 50
    temperature_eval: float = 1.0

    # Logging
    log_every: int = 20
    seed: int = 42


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dpo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(results_dir / "dpo_training.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Abstract ModelBackend
# ---------------------------------------------------------------------------

class ModelBackend(abc.ABC):
    """Abstract interface for generative model backends used in DPO.

    Each backend must support:
      - Sampling molecules from a source SMILES (or de novo)
      - Computing log-probabilities for source->target pairs
      - Saving/loading checkpoints
      - Exposing trainable parameters for the optimizer
      - Switching between training and inference modes
    """

    @abc.abstractmethod
    def sample(
        self, source_smiles: List[str], n_samples_per_source: int,
        temperature: float = 1.0, batch_size: int = 64,
    ) -> List[List[str]]:
        """Generate molecules for each source SMILES.

        Args:
            source_smiles: List of source molecules.
            n_samples_per_source: Number of samples to generate per source.
            temperature: Sampling temperature.
            batch_size: Internal batch size for generation.

        Returns:
            List of lists: generated SMILES for each source.
        """

    @abc.abstractmethod
    def compute_log_probs(
        self, source_smiles: List[str], target_smiles: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute log p(target | source) for each pair.

        Returns:
            1-D tensor of log-probabilities (one per pair).
        """

    @abc.abstractmethod
    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoder hidden states for the critic network.

        Returns:
            (encoder_output, mask) tensors.
        """

    @abc.abstractmethod
    def parameters(self):
        """Return iterable of trainable parameters."""

    @abc.abstractmethod
    def set_mode(self, mode: str):
        """Set model mode: 'training' or 'inference'."""

    @abc.abstractmethod
    def save(self, path: str):
        """Save model checkpoint."""

    @abc.abstractmethod
    def load(self, path: str):
        """Load model checkpoint."""

    @abc.abstractmethod
    def to(self, device: torch.device):
        """Move model to device."""

    @abc.abstractmethod
    def d_model(self) -> int:
        """Return the hidden dimension of the model (for critic sizing)."""


# ---------------------------------------------------------------------------
# Backend 1: REINVENT4 Mol2Mol (default)
# ---------------------------------------------------------------------------

class ReinventMol2MolBackend(ModelBackend):
    """Wraps the REINVENT4 Mol2Mol Transformer."""

    def __init__(self, prior_path: str, device: torch.device, mode: str = "inference"):
        from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
        from reinvent.models.model_mode_enum import ModelModeEnum

        save_dict = torch.load(prior_path, map_location="cpu", weights_only=False)

        # Handle missing/empty metadata
        if save_dict.get("metadata") is None:
            from reinvent.models import meta_data as md
            import uuid, time as _t

            save_dict["metadata"] = md.ModelMetaData(
                hash_id=None,
                hash_id_format="",
                model_id=uuid.uuid4().hex,
                origina_data_source="unknown",
                creation_date=_t.time(),
                comments=["REINVENT4-DPO"],
            )

        self.model = Mol2MolModel.create_from_dict(save_dict, mode, device)
        self._device = device

    def sample(
        self, source_smiles: List[str], n_samples_per_source: int,
        temperature: float = 1.0, batch_size: int = 64,
    ) -> List[List[str]]:
        from reinvent.models.transformer.core.enums.sampling_mode_enum import SamplingModesEnum

        self.model.set_mode("inference")
        self.model.set_temperature(temperature)

        results = []
        for src_smi in source_smiles:
            src_encoded, src_mask = self._encode_smiles(
                [src_smi], self.model.vocabulary, self.model.tokenizer, self._device
            )
            generated = []
            n_remaining = n_samples_per_source
            while n_remaining > 0:
                bsz = min(n_remaining, batch_size)
                src_batch = src_encoded.expand(bsz, -1)
                src_mask_batch = src_mask.expand(bsz, -1, -1)
                with torch.no_grad():
                    _, output_smiles, _ = self.model.sample(
                        src_batch, src_mask_batch, SamplingModesEnum.MULTINOMIAL
                    )
                generated.extend(output_smiles)
                n_remaining -= bsz
            results.append(generated)

        return results

    def compute_log_probs(
        self, source_smiles: List[str], target_smiles: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute -NLL (log prob) for source->target pairs."""
        from reinvent.models.transformer.core.dataset.paired_dataset import PairedDataset
        from torch.utils.data import DataLoader

        dataset = PairedDataset(
            source_smiles, target_smiles,
            vocabulary=self.model.vocabulary,
            tokenizer=self.model.tokenizer,
        )
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=PairedDataset.collate_fn, drop_last=False,
        )

        nlls = []
        for batch in loader:
            nll = self.model.likelihood(
                batch.input, batch.input_mask, batch.output, batch.output_mask
            )
            nlls.append(nll.detach())

        return -torch.cat(nlls, dim=0)  # return log-probs (not NLL)

    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_encoded, src_mask = self._encode_smiles(
            source_smiles, self.model.vocabulary, self.model.tokenizer, self._device
        )
        with torch.no_grad():
            encoder_out = self.model.network.encode(src_encoded, src_mask)
        return encoder_out, src_mask

    def parameters(self):
        return self.model.get_network_parameters()

    def set_mode(self, mode: str):
        self.model.set_mode(mode)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        save_dict = torch.load(path, map_location="cpu", weights_only=False)
        from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
        self.model = Mol2MolModel.create_from_dict(save_dict, "inference", self._device)

    def to(self, device: torch.device):
        self._device = device
        # Mol2Mol handles device internally
        return self

    def d_model(self) -> int:
        return self.model.network.model_dimension

    @staticmethod
    def _encode_smiles(smiles_list, vocabulary, tokenizer, device):
        """Tokenize and encode a list of SMILES into a padded tensor + mask."""
        encoded_seqs = []
        for smi in smiles_list:
            tokens = tokenizer.tokenize(smi)
            enc = vocabulary.encode(tokens)
            encoded_seqs.append(torch.tensor(enc, dtype=torch.long))

        max_len = max(s.size(0) for s in encoded_seqs)
        batch_size = len(encoded_seqs)

        padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask = torch.zeros(batch_size, 1, max_len, dtype=torch.bool)

        for i, seq in enumerate(encoded_seqs):
            padded[i, : len(seq)] = seq
            mask[i, 0, : len(seq)] = True

        return padded.to(device), mask.to(device)


# ---------------------------------------------------------------------------
# Backend 2: MolT5-small (T5-based, laituan245/molt5-small)
# ---------------------------------------------------------------------------

class MolT5Backend(ModelBackend):
    """MolT5-small: T5-based encoder-decoder for SMILES generation.

    Uses laituan245/molt5-small (60M params) from HuggingFace.
    Source SMILES is fed as encoder input; target SMILES is the decoder output.

    Requires: pip install transformers sentencepiece
    """

    HF_MODEL_ID = "laituan245/molt5-small"

    def __init__(self, device: torch.device, model_id: str = None):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError:
            raise ImportError(
                "MolT5 backend requires the transformers library. "
                "Install with: pip install transformers sentencepiece"
            )

        model_id = model_id or self.HF_MODEL_ID
        self._device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        self._model_dim = self.model.config.d_model  # 512 for t5-small

    def sample(
        self, source_smiles: List[str], n_samples_per_source: int,
        temperature: float = 1.0, batch_size: int = 64,
    ) -> List[List[str]]:
        self.model.eval()
        results = []

        for src_smi in source_smiles:
            generated = []
            n_remaining = n_samples_per_source

            while n_remaining > 0:
                bsz = min(n_remaining, batch_size)
                inputs = self.tokenizer(
                    [src_smi] * bsz, return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=256,
                        num_return_sequences=1,
                    )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated.extend(decoded)
                n_remaining -= bsz

            results.append(generated)

        return results

    def compute_log_probs(
        self, source_smiles: List[str], target_smiles: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute log p(target | source) via teacher-forced forward pass."""
        all_log_probs = []

        for start in range(0, len(source_smiles), batch_size):
            end = min(start + batch_size, len(source_smiles))
            src_batch = source_smiles[start:end]
            tgt_batch = target_smiles[start:end]

            inputs = self.tokenizer(
                src_batch, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(self._device)

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    tgt_batch, return_tensors="pt",
                    padding=True, truncation=True, max_length=512,
                ).to(self._device)

            label_ids = labels.input_ids.clone()
            # Replace padding token id with -100 so it is ignored in loss
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=label_ids,
            )

            # Per-token log-probs
            logits = outputs.logits  # (batch, seq_len, vocab)
            log_probs_all = F.log_softmax(logits, dim=-1)

            # Gather log-probs at target token positions
            target_ids = labels.input_ids  # (batch, seq_len)
            # Shift: T5 shifts internally, but logits align with labels
            gathered = log_probs_all.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)  # (batch, seq_len)

            # Mask out padding
            mask = (target_ids != self.tokenizer.pad_token_id).float()
            seq_log_probs = (gathered * mask).sum(dim=-1)  # (batch,)
            all_log_probs.append(seq_log_probs.detach())

        return torch.cat(all_log_probs, dim=0)

    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            source_smiles, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self._device)

        with torch.no_grad():
            encoder_out = self.model.encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            ).last_hidden_state

        # Convert attention_mask to (batch, 1, seq_len) bool for SmilesCritic
        mask = inputs.attention_mask.unsqueeze(1).bool()
        return encoder_out, mask

    def parameters(self):
        return self.model.parameters()

    def set_mode(self, mode: str):
        if mode == "training":
            self.model.train()
        else:
            self.model.eval()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)

    def to(self, device: torch.device):
        self._device = device
        self.model = self.model.to(device)
        return self

    def d_model(self) -> int:
        return self._model_dim


# ---------------------------------------------------------------------------
# Backend 3: SMILES-GPT2 (character-level GPT-2)
# ---------------------------------------------------------------------------

class SmilesGPT2Backend(ModelBackend):
    """GPT-2 with character-level SMILES tokenization for molecular generation.

    Uses HuggingFace gpt2 (124M params) with a custom character-level tokenizer
    where each SMILES character is a single token (~60 vocab).

    For source->target generation, the prompt format is:
        "<source_smiles>><target_smiles>"
    Log-probs are computed only on the target portion (after ">").

    Requires: pip install transformers
    """

    # SMILES character vocabulary (covers most common characters)
    SMILES_CHARS = list(
        "CNOSFPIBrcnoslp1234567890=#-+()[]@/\\. >{}<"
    )
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    SEP_TOKEN = ">"

    def __init__(self, device: torch.device, pretrained_id: str = "gpt2"):
        try:
            from transformers import GPT2LMHeadModel, GPT2Config
        except ImportError:
            raise ImportError(
                "SmilesGPT2 backend requires the transformers library. "
                "Install with: pip install transformers"
            )

        self._device = device

        # Build character-level vocabulary
        self._build_vocab()

        # Initialize GPT-2 with character-level vocab
        config = GPT2Config(
            vocab_size=len(self.char2idx),
            n_positions=1024,
            n_embd=768,
            n_layer=12,
            n_head=12,
            pad_token_id=self.char2idx[self.PAD_TOKEN],
            bos_token_id=self.char2idx[self.BOS_TOKEN],
            eos_token_id=self.char2idx[self.EOS_TOKEN],
        )
        self.model = GPT2LMHeadModel(config).to(device)
        self._model_dim = config.n_embd  # 768

    def _build_vocab(self):
        """Build character-level SMILES vocabulary."""
        special = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        all_tokens = special + self.SMILES_CHARS
        self.char2idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_id = self.char2idx[self.PAD_TOKEN]
        self.bos_id = self.char2idx[self.BOS_TOKEN]
        self.eos_id = self.char2idx[self.EOS_TOKEN]
        self.sep_id = self.char2idx[self.SEP_TOKEN]

    def _encode_text(self, text: str) -> List[int]:
        """Encode a string to token IDs, skipping unknown characters."""
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def _decode_ids(self, ids: List[int]) -> str:
        """Decode token IDs to string."""
        chars = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            c = self.idx2char.get(idx, "")
            chars.append(c)
        return "".join(chars)

    def _prepare_prompt(self, source_smi: str) -> List[int]:
        """Prepare input IDs: <bos> + source + >"""
        return [self.bos_id] + self._encode_text(source_smi) + [self.sep_id]

    def _prepare_pair(self, source_smi: str, target_smi: str) -> Tuple[List[int], int]:
        """Prepare full sequence: <bos> + source + > + target + <eos>.

        Returns (token_ids, target_start_idx) where target_start_idx is the
        position where the target tokens begin.
        """
        src_tokens = [self.bos_id] + self._encode_text(source_smi) + [self.sep_id]
        tgt_tokens = self._encode_text(target_smi) + [self.eos_id]
        target_start = len(src_tokens)
        return src_tokens + tgt_tokens, target_start

    def sample(
        self, source_smiles: List[str], n_samples_per_source: int,
        temperature: float = 1.0, batch_size: int = 64,
    ) -> List[List[str]]:
        self.model.eval()
        results = []

        for src_smi in source_smiles:
            prompt_ids = self._prepare_prompt(src_smi)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self._device)

            generated = []
            n_remaining = n_samples_per_source

            while n_remaining > 0:
                bsz = min(n_remaining, batch_size)
                input_ids = prompt_tensor.expand(bsz, -1)

                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        do_sample=True,
                        temperature=temperature,
                        max_new_tokens=256,
                        eos_token_id=self.eos_id,
                        pad_token_id=self.pad_id,
                    )

                for seq in outputs:
                    # Extract target portion (after the prompt)
                    target_ids = seq[len(prompt_ids):].tolist()
                    smi = self._decode_ids(target_ids)
                    generated.append(smi)

                n_remaining -= bsz

            results.append(generated)

        return results

    def compute_log_probs(
        self, source_smiles: List[str], target_smiles: List[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute log p(target | source) for concatenated sequences.

        Only log-probs on the target portion (after ">") are summed.
        """
        all_log_probs = []

        for start in range(0, len(source_smiles), batch_size):
            end = min(start + batch_size, len(source_smiles))

            # Prepare sequences
            sequences = []
            target_starts = []
            for i in range(start, end):
                ids, tgt_start = self._prepare_pair(source_smiles[i], target_smiles[i])
                sequences.append(ids)
                target_starts.append(tgt_start)

            # Pad to same length
            max_len = max(len(s) for s in sequences)
            input_ids = torch.full(
                (len(sequences), max_len), self.pad_id,
                dtype=torch.long, device=self._device,
            )
            attention_mask = torch.zeros(
                len(sequences), max_len, dtype=torch.long, device=self._device,
            )
            for i, seq in enumerate(sequences):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
                attention_mask[i, :len(seq)] = 1

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # logits shape: (batch, seq_len, vocab)
            logits = outputs.logits
            log_probs_all = F.log_softmax(logits, dim=-1)

            # For autoregressive models, logits[t] predicts token[t+1]
            # So log_prob of token at position t is log_probs_all[t-1, token[t]]
            batch_lps = []
            for i in range(len(sequences)):
                tgt_start = target_starts[i]
                seq_len = len(sequences[i])
                if tgt_start >= seq_len:
                    batch_lps.append(torch.tensor(0.0, device=self._device))
                    continue

                # Target token positions: tgt_start to seq_len-1
                # The log-prob for token at position t comes from logits at t-1
                target_positions = torch.arange(
                    tgt_start, seq_len, device=self._device
                )
                logit_positions = target_positions - 1  # shifted by 1
                target_token_ids = input_ids[i, target_positions]

                lp = log_probs_all[i, logit_positions, target_token_ids]
                batch_lps.append(lp.sum())

            all_log_probs.append(torch.stack(batch_lps).detach())

        return torch.cat(all_log_probs, dim=0)

    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get hidden states from the GPT-2 transformer for the source portion.

        For a decoder-only model, we pass the source tokens through and use
        the final hidden states as the 'encoder output' for the critic.
        """
        # Encode source SMILES
        sequences = []
        for smi in source_smiles:
            ids = self._prepare_prompt(smi)
            sequences.append(ids)

        max_len = max(len(s) for s in sequences)
        input_ids = torch.full(
            (len(sequences), max_len), self.pad_id,
            dtype=torch.long, device=self._device,
        )
        attention_mask = torch.zeros(
            len(sequences), max_len, dtype=torch.long, device=self._device,
        )
        for i, seq in enumerate(sequences):
            input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :len(seq)] = 1

        with torch.no_grad():
            outputs = self.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state  # (batch, seq_len, 768)

        mask = attention_mask.unsqueeze(1).bool()  # (batch, 1, seq_len)
        return hidden, mask

    def parameters(self):
        return self.model.parameters()

    def set_mode(self, mode: str):
        if mode == "training":
            self.model.train()
        else:
            self.model.eval()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "char2idx": self.char2idx,
        }, os.path.join(path, "smiles_gpt2.pt"))

    def load(self, path: str):
        ckpt = torch.load(
            os.path.join(path, "smiles_gpt2.pt"),
            map_location="cpu", weights_only=False,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model = self.model.to(self._device)

    def to(self, device: torch.device):
        self._device = device
        self.model = self.model.to(device)
        return self

    def d_model(self) -> int:
        return self._model_dim


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def create_backend(
    backend_name: str, device: torch.device, prior_path: str = None, mode: str = "inference",
) -> ModelBackend:
    """Create a ModelBackend instance by name.

    Args:
        backend_name: One of 'reinvent_mol2mol', 'molt5', 'smiles_gpt2'.
        device: Torch device.
        prior_path: Path to REINVENT4 prior (only for reinvent_mol2mol).
        mode: 'training' or 'inference'.

    Returns:
        A ModelBackend instance.
    """
    if backend_name == "reinvent_mol2mol":
        if prior_path is None:
            raise ValueError("prior_path is required for reinvent_mol2mol backend")
        return ReinventMol2MolBackend(prior_path, device, mode=mode)
    elif backend_name == "molt5":
        backend = MolT5Backend(device)
        backend.set_mode(mode)
        return backend
    elif backend_name == "smiles_gpt2":
        backend = SmilesGPT2Backend(device)
        backend.set_mode(mode)
        return backend
    else:
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Choose from: reinvent_mol2mol, molt5, smiles_gpt2"
        )


# ---------------------------------------------------------------------------
# Phase 1: Generate preference data
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    source_smi: str
    win_smi: str
    lose_smi: str
    win_score: float
    lose_score: float


def generate_preference_data(
    cfg: DPOConfig,
    backend: ModelBackend,
    film_model,
    film_scaler,
    film_anchors,
    film_pIC50,
    logger: logging.Logger,
) -> List[PreferencePair]:
    """Phase 1: sample analogs for each source, score, construct preference pairs."""

    # Load source molecules
    actives = [line.strip() for line in open(cfg.actives_path) if line.strip()]
    logger.info(f"Phase 1: {len(actives)} source molecules, {cfg.n_samples_per_source} samples each")

    backend.set_mode("inference")

    all_pairs: List[PreferencePair] = []
    total_generated = 0
    total_valid = 0

    for src_idx, src_smi in enumerate(actives):
        # Sample from backend
        generated_lists = backend.sample(
            [src_smi], cfg.n_samples_per_source,
            temperature=cfg.temperature_sample, batch_size=cfg.sample_batch_size,
        )
        generated_smiles = generated_lists[0] if generated_lists else []

        total_generated += len(generated_smiles)

        # Validate and deduplicate
        valid_smiles = []
        for smi in generated_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                if canon and canon != src_smi:
                    valid_smiles.append(canon)
        valid_smiles = list(set(valid_smiles))
        total_valid += len(valid_smiles)

        if len(valid_smiles) < 2:
            continue

        # Score with FiLMDelta
        from experiments.reinvent4_film_scorer import score_smiles

        scores = score_smiles(valid_smiles, film_model, film_scaler, film_anchors, film_pIC50)

        # Filter out NaN scores
        scored = [(s, sc) for s, sc in zip(valid_smiles, scores) if not np.isnan(sc)]
        if len(scored) < 2:
            continue

        scored.sort(key=lambda x: x[1], reverse=True)

        # Construct preference pairs: top vs bottom with margin
        for i in range(len(scored)):
            for j in range(len(scored) - 1, i, -1):
                if scored[i][1] - scored[j][1] >= cfg.preference_margin:
                    all_pairs.append(PreferencePair(
                        source_smi=src_smi,
                        win_smi=scored[i][0],
                        lose_smi=scored[j][0],
                        win_score=scored[i][1],
                        lose_score=scored[j][1],
                    ))

        if (src_idx + 1) % 5 == 0 or src_idx == len(actives) - 1:
            logger.info(
                f"  Source {src_idx + 1}/{len(actives)}: "
                f"{len(valid_smiles)} valid, {len(all_pairs)} pairs so far"
            )

    logger.info(
        f"Phase 1 complete: {total_generated} generated, {total_valid} valid unique, "
        f"{len(all_pairs)} preference pairs"
    )

    return all_pairs


# ---------------------------------------------------------------------------
# Phase 2: DPO training
# ---------------------------------------------------------------------------

def dpo_train(
    cfg: DPOConfig,
    pi_theta: ModelBackend,
    pi_ref: ModelBackend,
    pairs: List[PreferencePair],
    logger: logging.Logger,
) -> dict:
    """Phase 2: DPO fine-tuning of pi_theta against frozen pi_ref."""

    logger.info(f"Phase 2: DPO training -- {len(pairs)} pairs, {cfg.n_epochs} epochs, "
                f"beta={cfg.beta_dpo}, lr={cfg.lr}")

    # Split into train/val
    np.random.seed(cfg.seed)
    indices = np.random.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * cfg.val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    logger.info(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Set up optimizer
    pi_theta.set_mode("training")
    pi_ref.set_mode("inference")

    optimizer = torch.optim.AdamW(
        pi_theta.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "kl_div": [],
    }

    for epoch in range(1, cfg.n_epochs + 1):
        epoch_start = time.time()

        # --- Training ---
        pi_theta.set_mode("training")
        np.random.shuffle(train_pairs)

        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch_start in range(0, len(train_pairs), cfg.train_batch_size):
            batch = train_pairs[batch_start : batch_start + cfg.train_batch_size]

            src_list = [p.source_smi for p in batch]
            win_list = [p.win_smi for p in batch]
            lose_list = [p.lose_smi for p in batch]

            # Compute log-probs for win and lose under pi_theta
            lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
            lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            # Compute log-probs under pi_ref (frozen)
            with torch.no_grad():
                lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))

            # log_ratio = log pi_theta(y|x) - log pi_ref(y|x)
            log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
            log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)

            # DPO loss: -E[log sigmoid(beta * (log_ratio_win - log_ratio_lose))]
            logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
            loss = -F.logsigmoid(logits).mean()

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    pi_theta.parameters(), cfg.grad_clip
                )
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_correct += (logits.detach() > 0).sum().item()
            epoch_total += len(batch)

            step = batch_start // cfg.train_batch_size + 1
            if step % cfg.log_every == 0:
                logger.info(
                    f"  Epoch {epoch}, step {step}: loss={loss.item():.4f}"
                )

        train_loss = np.mean(epoch_losses)
        train_acc = epoch_correct / max(epoch_total, 1)

        # --- Validation ---
        pi_theta.set_mode("inference")
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_start in range(0, len(val_pairs), cfg.train_batch_size):
                batch = val_pairs[batch_start : batch_start + cfg.train_batch_size]

                src_list = [p.source_smi for p in batch]
                win_list = [p.win_smi for p in batch]
                lose_list = [p.lose_smi for p in batch]

                lp_theta_win = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_theta_lose = pi_theta.compute_log_probs(src_list, lose_list, batch_size=len(batch))
                lp_ref_win = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_ref_lose = pi_ref.compute_log_probs(src_list, lose_list, batch_size=len(batch))

                log_ratio_win = lp_theta_win - lp_ref_win.to(lp_theta_win.device)
                log_ratio_lose = lp_theta_lose - lp_ref_lose.to(lp_theta_lose.device)

                logits = cfg.beta_dpo * (log_ratio_win - log_ratio_lose)
                loss = -F.logsigmoid(logits).mean()

                val_losses.append(loss.item())
                val_correct += (logits > 0).sum().item()
                val_total += len(batch)

        val_loss = np.mean(val_losses) if val_losses else float("nan")
        val_acc = val_correct / max(val_total, 1)

        # --- KL divergence estimate: E[log_pi_theta - log_pi_ref] over wins ---
        kl_samples = []
        with torch.no_grad():
            for batch_start in range(0, min(len(val_pairs), 200), cfg.train_batch_size):
                batch = val_pairs[batch_start : batch_start + cfg.train_batch_size]
                src_list = [p.source_smi for p in batch]
                win_list = [p.win_smi for p in batch]
                lp_theta = pi_theta.compute_log_probs(src_list, win_list, batch_size=len(batch))
                lp_ref = pi_ref.compute_log_probs(src_list, win_list, batch_size=len(batch))
                kl_samples.append((lp_theta - lp_ref.to(lp_theta.device)).mean().item())
        kl_est = np.mean(kl_samples) if kl_samples else float("nan")

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{cfg.n_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
            f"KL={kl_est:.4f}, time={elapsed:.1f}s"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["kl_div"].append(kl_est)

    logger.info("Phase 2 complete.")
    return history


# ---------------------------------------------------------------------------
# Phase 3: Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    cfg: DPOConfig,
    backend: ModelBackend,
    model_label: str,
    film_model,
    film_scaler,
    film_anchors,
    film_pIC50,
    logger: logging.Logger,
) -> dict:
    """Sample from a model and score with FiLMDelta. Returns summary statistics."""
    from experiments.reinvent4_film_scorer import score_smiles

    actives = [line.strip() for line in open(cfg.actives_path) if line.strip()]

    backend.set_mode("inference")

    all_results = []

    for src_idx, src_smi in enumerate(actives):
        generated_lists = backend.sample(
            [src_smi], cfg.n_eval_samples,
            temperature=cfg.temperature_eval, batch_size=cfg.sample_batch_size,
        )
        generated = generated_lists[0] if generated_lists else []

        # Validate
        valid = []
        for smi in generated:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon = Chem.MolToSmiles(mol)
                if canon:
                    valid.append(canon)
        valid_unique = list(set(valid))

        if not valid_unique:
            continue

        scores = score_smiles(valid_unique, film_model, film_scaler, film_anchors, film_pIC50)
        valid_scores = [(s, sc) for s, sc in zip(valid_unique, scores) if not np.isnan(sc)]

        for smi, sc in valid_scores:
            all_results.append({
                "source": src_smi,
                "generated": smi,
                "pIC50": sc,
            })

    if not all_results:
        logger.warning(f"  {model_label}: no valid scored molecules")
        return {"label": model_label, "n_generated": 0}

    pIC50s = [r["pIC50"] for r in all_results]
    n_unique = len(set(r["generated"] for r in all_results))
    n_potent = sum(1 for p in pIC50s if p >= 7.0)

    summary = {
        "label": model_label,
        "n_scored": len(all_results),
        "n_unique": n_unique,
        "n_potent_7": n_potent,
        "frac_potent_7": n_potent / max(len(all_results), 1),
        "mean_pIC50": float(np.mean(pIC50s)),
        "median_pIC50": float(np.median(pIC50s)),
        "max_pIC50": float(np.max(pIC50s)),
        "std_pIC50": float(np.std(pIC50s)),
    }

    logger.info(
        f"  {model_label}: {n_unique} unique, mean={summary['mean_pIC50']:.3f}, "
        f"max={summary['max_pIC50']:.3f}, potent(>=7)={n_potent} ({summary['frac_potent_7']:.1%})"
    )

    return summary, all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPO molecular generation with multiple model backends"
    )
    parser.add_argument(
        "--model",
        choices=["reinvent_mol2mol", "molt5", "smiles_gpt2"],
        default="reinvent_mol2mol",
        help="Model backend to use (default: reinvent_mol2mol)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = DPOConfig()
    cfg.model_backend = args.model

    # Adjust results dir to include model name
    if cfg.model_backend != "reinvent_mol2mol":
        cfg.results_dir = str(
            PROJECT_ROOT / "results" / "paper_evaluation"
            / f"reinvent4_dpo_{cfg.model_backend}"
        )

    results_dir = Path(cfg.results_dir)
    logger = setup_logging(results_dir)

    logger.info("=" * 70)
    logger.info(f"DPO Molecular Generation for ZAP70 (backend: {cfg.model_backend})")
    logger.info("=" * 70)

    # Set seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load FiLMDelta scorer (in-process)
    # ------------------------------------------------------------------
    logger.info("Loading FiLMDelta scorer...")
    from experiments.reinvent4_film_scorer import load_film_model

    film_model, film_scaler, film_anchors, film_pIC50 = load_film_model()
    logger.info(f"FiLMDelta loaded: {len(film_pIC50)} anchors")

    # ------------------------------------------------------------------
    # Load model backends: pi_ref (frozen) and pi_theta (trainable)
    # ------------------------------------------------------------------
    logger.info(f"Loading {cfg.model_backend} as pi_ref (frozen)...")
    pi_ref = create_backend(
        cfg.model_backend, device, prior_path=cfg.prior_path, mode="inference",
    )

    logger.info(f"Loading {cfg.model_backend} as pi_theta (trainable)...")
    pi_theta = create_backend(
        cfg.model_backend, device, prior_path=cfg.prior_path, mode="training",
    )

    n_params = sum(p.numel() for p in pi_theta.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Model d_model: {pi_theta.d_model()}")

    # Freeze reference model
    for p in pi_ref.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # Phase 1: Generate preference data
    # ------------------------------------------------------------------
    pref_cache = results_dir / "preference_pairs.json"

    if pref_cache.exists():
        logger.info(f"Loading cached preference pairs from {pref_cache}")
        with open(pref_cache) as f:
            raw_pairs = json.load(f)
        pairs = [PreferencePair(**p) for p in raw_pairs]
        logger.info(f"Loaded {len(pairs)} cached preference pairs")
    else:
        pairs = generate_preference_data(
            cfg, pi_ref, film_model, film_scaler, film_anchors, film_pIC50, logger
        )
        # Cache for reruns
        with open(pref_cache, "w") as f:
            json.dump([asdict(p) for p in pairs], f, indent=2)
        logger.info(f"Saved {len(pairs)} preference pairs to {pref_cache}")

    if len(pairs) < 10:
        logger.error(f"Only {len(pairs)} preference pairs -- too few for DPO training. Aborting.")
        return

    # Log pair statistics
    score_gaps = [p.win_score - p.lose_score for p in pairs]
    logger.info(
        f"Preference pairs: {len(pairs)}, "
        f"mean gap={np.mean(score_gaps):.3f}, "
        f"median gap={np.median(score_gaps):.3f}, "
        f"max gap={np.max(score_gaps):.3f}"
    )

    # ------------------------------------------------------------------
    # Phase 2: DPO training
    # ------------------------------------------------------------------
    history = dpo_train(cfg, pi_theta, pi_ref, pairs, logger)

    # Save training history
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save trained model
    model_path = results_dir / "dpo_model"
    if cfg.model_backend == "reinvent_mol2mol":
        model_path = results_dir / "dpo_mol2mol.pt"
    pi_theta.save(str(model_path))
    logger.info(f"Saved DPO-trained model to {model_path}")

    # ------------------------------------------------------------------
    # Phase 3: Evaluate DPO policy vs prior (DAP baseline proxy)
    # ------------------------------------------------------------------
    logger.info("Phase 3: Evaluation")

    dpo_summary, dpo_results = evaluate_policy(
        cfg, pi_theta, "DPO", film_model, film_scaler, film_anchors, film_pIC50, logger
    )
    prior_summary, prior_results = evaluate_policy(
        cfg, pi_ref, "Prior (DAP baseline)", film_model, film_scaler, film_anchors, film_pIC50, logger
    )

    # Save detailed results
    with open(results_dir / "dpo_generated.json", "w") as f:
        json.dump(dpo_results, f, indent=2)
    with open(results_dir / "prior_generated.json", "w") as f:
        json.dump(prior_results, f, indent=2)

    # Compare
    comparison = {
        "model_backend": cfg.model_backend,
        "dpo": dpo_summary,
        "prior": prior_summary,
        "config": asdict(cfg),
        "training_history": history,
        "n_preference_pairs": len(pairs),
        "preference_margin": cfg.preference_margin,
    }

    # Compute improvement metrics
    if dpo_summary.get("mean_pIC50") and prior_summary.get("mean_pIC50"):
        comparison["improvement"] = {
            "mean_pIC50_delta": dpo_summary["mean_pIC50"] - prior_summary["mean_pIC50"],
            "max_pIC50_delta": dpo_summary["max_pIC50"] - prior_summary["max_pIC50"],
            "potent_frac_delta": dpo_summary["frac_potent_7"] - prior_summary["frac_potent_7"],
        }

    with open(results_dir / "dpo_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info(f"RESULTS SUMMARY (backend: {cfg.model_backend})")
    logger.info("=" * 70)
    logger.info(f"{'Metric':<25} {'Prior':>12} {'DPO':>12}")
    logger.info("-" * 50)
    for key in ["n_scored", "n_unique", "mean_pIC50", "max_pIC50", "n_potent_7", "frac_potent_7"]:
        pv = prior_summary.get(key, "N/A")
        dv = dpo_summary.get(key, "N/A")
        if isinstance(pv, float):
            pv = f"{pv:.3f}"
            dv = f"{dv:.3f}"
        logger.info(f"{key:<25} {str(pv):>12} {str(dv):>12}")

    if "improvement" in comparison:
        imp = comparison["improvement"]
        logger.info("-" * 50)
        logger.info(f"Mean pIC50 improvement: {imp['mean_pIC50_delta']:+.3f}")
        logger.info(f"Max pIC50 improvement:  {imp['max_pIC50_delta']:+.3f}")
        logger.info(f"Potent fraction delta:  {imp['potent_frac_delta']:+.3f}")

    logger.info(f"\nResults saved to {results_dir}/")
    logger.info("Done.")


if __name__ == "__main__":
    main()
