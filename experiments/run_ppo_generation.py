#!/usr/bin/env python3
"""
PPO-based Molecular Generation with multiple model backends and FiLMDelta Scoring.

Supports multiple model backends:
  - reinvent_mol2mol: REINVENT4 Mol2Mol Transformer (default)
  - molt5: MolT5-small (T5-based, 60M params, from laituan245/molt5-small)
  - smiles_gpt2: GPT-2 with character-level SMILES tokenization (124M params)

Uses Proximal Policy Optimization to fine-tune the model for generating ZAP70
inhibitors, scored by anchor-based FiLMDelta pIC50 predictions.

The generative model acts as the policy (actor), while a small critic network
estimates value from encoder hidden states. FiLMDelta provides in-process reward.

Usage:
    # Default (REINVENT4 Mol2Mol):
    conda run --no-capture-output -n quris python experiments/run_ppo_generation.py

    # MolT5-small (requires: pip install transformers sentencepiece):
    conda run --no-capture-output -n quris python experiments/run_ppo_generation.py --model molt5

    # SMILES-GPT2 (requires: pip install transformers):
    conda run --no-capture-output -n quris python experiments/run_ppo_generation.py --model smiles_gpt2

Results saved to: results/paper_evaluation/reinvent4_ppo[_<model>]/
"""

import abc
import argparse
import sys
import os
import json
import gc
import copy
import time
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["RDK_DEPRECATION_WARNING"] = "off"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

# ------------------------------------------------------------------
# Project imports
# ------------------------------------------------------------------
from src.models.predictors.smiles_critic import SmilesCritic

# FiLMDelta scorer — reuse the helpers from reinvent4_film_scorer
from experiments.reinvent4_film_scorer import (
    load_film_model,
    score_smiles,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
PRIOR_PATH = REINVENT4_ROOT / "priors" / "mol2mol_medium_similarity.prior"

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
BATCH_SIZE = 64
TOTAL_ROLLOUTS = 200
PPO_EPOCHS = 4
CLIP_EPS = 0.2
ACTOR_LR = 3e-5
CRITIC_LR = 1e-4
KL_TARGET = 0.01
BETA_KL_INIT = 0.1
BETA_KL_MIN = 1e-4
BETA_KL_MAX = 10.0
VALUE_COEFF = 0.5
TEMPERATURE = 1.0
FINAL_SAMPLE_SIZE = 5000
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print(f"[ppo] Device: {DEVICE}")


# ==================================================================
# Abstract ModelBackend for PPO
# ==================================================================

class ModelBackend(abc.ABC):
    """Abstract interface for generative model backends used in PPO.

    Each backend must support:
      - Sampling molecules from source SMILES
      - Computing log-probabilities for source->target pairs
      - Getting encoder output for the critic network
      - Saving/loading checkpoints
      - Exposing trainable parameters and network for the optimizer
    """

    @abc.abstractmethod
    def sample(
        self, source_smiles: List[str], batch_size: int = 64,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[str]]:
        """Sample one molecule per source SMILES.

        Args:
            source_smiles: Encoded source molecules.
            batch_size: Batch size (should match len(source_smiles)).
            temperature: Sampling temperature.

        Returns:
            (input_smiles, output_smiles): Lists of source and generated SMILES.
        """

    @abc.abstractmethod
    def compute_log_probs(
        self, source_smiles: List[str], output_smiles: List[str],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute per-sequence log-probs for generated outputs.

        Returns:
            (log_probs, valid_indices): log-probs tensor and list of valid indices.
        """

    @abc.abstractmethod
    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get encoder hidden states for the critic.

        Returns:
            (encoder_output, mask) tensors.
        """

    @abc.abstractmethod
    def network_parameters(self):
        """Return iterable of trainable network parameters."""

    @abc.abstractmethod
    def set_mode(self, mode: str):
        """Set model mode: 'training' or 'inference'."""

    @abc.abstractmethod
    def set_eval(self):
        """Set the underlying network to eval mode."""

    @abc.abstractmethod
    def set_train(self):
        """Set the underlying network to train mode."""

    @abc.abstractmethod
    def save(self, path: str):
        """Save model checkpoint."""

    @abc.abstractmethod
    def d_model(self) -> int:
        """Return the hidden dimension (for critic sizing)."""


# ==================================================================
# Backend 1: REINVENT4 Mol2Mol
# ==================================================================

class ReinventMol2MolBackend(ModelBackend):
    """Wraps the REINVENT4 Mol2Mol Transformer for PPO."""

    def __init__(self, prior_path: str, device: torch.device, mode: str = "training"):
        sys.path.insert(0, str(REINVENT4_ROOT))
        from reinvent.models.transformer.mol2mol.mol2mol import Mol2MolModel
        from reinvent.models.transformer.core.network.module.subsequent_mask import subsequent_mask

        save_dict = torch.load(str(prior_path), map_location="cpu", weights_only=False)
        self.model = Mol2MolModel.create_from_dict(save_dict, mode=mode, device=device)
        self.vocabulary = self.model.vocabulary
        self.tokenizer = self.model.tokenizer
        self._device = device
        self._subsequent_mask = subsequent_mask

    def _encode_smiles(self, smiles_list):
        """Encode SMILES into padded tensors."""
        encoded = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            try:
                tokens = self.tokenizer.tokenize(smi)
                enc = self.vocabulary.encode(tokens)
                if len(enc) > 0:
                    encoded.append(torch.tensor(enc).long())
                    valid_indices.append(i)
            except (KeyError, Exception):
                continue

        if not encoded:
            return None, None, []

        max_len = max(e.size(0) for e in encoded)
        src = torch.zeros(len(encoded), max_len).long()
        src_mask = torch.zeros(len(encoded), 1, max_len).bool()
        for i, e in enumerate(encoded):
            src[i, : len(e)] = e
            src_mask[i, 0, : len(e)] = True

        return src.to(self._device), src_mask.to(self._device), valid_indices

    def sample(
        self, source_smiles: List[str], batch_size: int = 64,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[str]]:
        self.model.set_mode("inference")
        self.model.network.eval()
        self.model.set_temperature(temperature)

        src, src_mask, valid_idx = self._encode_smiles(source_smiles)
        if src is None:
            return [], []

        with torch.no_grad():
            inp_list, out_list, sample_nlls = self.model.sample(
                src, src_mask, "multinomial"
            )

        return inp_list, out_list

    def compute_log_probs(
        self, source_smiles: List[str], output_smiles: List[str],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute log-probs using the model's likelihood method."""
        src, src_mask, _ = self._encode_smiles(source_smiles)

        encoded_outputs = []
        valid = []
        for i, smi in enumerate(output_smiles):
            try:
                tokens = self.tokenizer.tokenize(smi)
                enc = self.vocabulary.encode(tokens)
                if len(enc) > 0:
                    encoded_outputs.append(torch.tensor(enc).long())
                    valid.append(i)
            except (KeyError, Exception):
                continue

        if not encoded_outputs:
            return torch.tensor([]), []

        max_out_len = max(e.size(0) for e in encoded_outputs)
        trg = torch.zeros(len(encoded_outputs), max_out_len).long()
        for i, e in enumerate(encoded_outputs):
            trg[i, : len(e)] = e
        trg = trg.to(self._device)

        trg_pad_mask = torch.zeros(len(encoded_outputs), 1, max_out_len).bool()
        for i, e in enumerate(encoded_outputs):
            trg_pad_mask[i, 0, : len(e)] = True
        trg_pad_mask = trg_pad_mask.to(self._device)

        trg_mask_causal = self._subsequent_mask(max_out_len).to(self._device)
        trg_mask = trg_pad_mask & Variable(trg_mask_causal.type_as(trg_pad_mask))
        trg_mask = trg_mask[:, :-1, :-1]

        src_sel = src[valid]
        src_mask_sel = src_mask[valid]

        nll = self.model.likelihood(src_sel, src_mask_sel, trg, trg_mask)
        log_probs = -nll  # log p = -NLL

        return log_probs, valid

    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src, src_mask, _ = self._encode_smiles(source_smiles)
        with torch.no_grad():
            encoder_out = self.model.network.encode(src, src_mask)
        return encoder_out, src_mask

    def network_parameters(self):
        return self.model.network.parameters()

    def set_mode(self, mode: str):
        self.model.set_mode(mode)

    def set_eval(self):
        self.model.network.eval()

    def set_train(self):
        self.model.network.train()

    def save(self, path: str):
        self.model.save(path)

    def d_model(self) -> int:
        return self.model.network.model_dimension


# ==================================================================
# Backend 2: MolT5-small (T5-based)
# ==================================================================

class MolT5Backend(ModelBackend):
    """MolT5-small: T5-based encoder-decoder for SMILES generation.

    Uses laituan245/molt5-small (60M params) from HuggingFace.

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
        self._model_dim = self.model.config.d_model

    def sample(
        self, source_smiles: List[str], batch_size: int = 64,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[str]]:
        self.model.eval()

        inputs = self.tokenizer(
            source_smiles, return_tensors="pt",
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
        return source_smiles, decoded

    def compute_log_probs(
        self, source_smiles: List[str], output_smiles: List[str],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Compute log p(target | source) via teacher-forced forward pass."""
        inputs = self.tokenizer(
            source_smiles, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(self._device)

        with self.tokenizer.as_target_tokenizer():
            labels_enc = self.tokenizer(
                output_smiles, return_tensors="pt",
                padding=True, truncation=True, max_length=512,
            ).to(self._device)

        label_ids = labels_enc.input_ids.clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=label_ids,
        )

        logits = outputs.logits
        log_probs_all = F.log_softmax(logits, dim=-1)

        target_ids = labels_enc.input_ids
        gathered = log_probs_all.gather(
            2, target_ids.unsqueeze(-1)
        ).squeeze(-1)

        mask = (target_ids != self.tokenizer.pad_token_id).float()
        seq_log_probs = (gathered * mask).sum(dim=-1)

        valid = list(range(len(source_smiles)))
        return seq_log_probs, valid

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

        mask = inputs.attention_mask.unsqueeze(1).bool()
        return encoder_out, mask

    def network_parameters(self):
        return self.model.parameters()

    def set_mode(self, mode: str):
        if mode == "training":
            self.model.train()
        else:
            self.model.eval()

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def d_model(self) -> int:
        return self._model_dim


# ==================================================================
# Backend 3: SMILES-GPT2 (character-level GPT-2)
# ==================================================================

class SmilesGPT2Backend(ModelBackend):
    """GPT-2 with character-level SMILES tokenization for molecular generation.

    Uses a custom GPT-2 model with character-level tokenization where each
    SMILES character is a single token (~60 vocab).

    For source->target generation, the prompt format is:
        "<bos><source_smiles>><target_smiles><eos>"

    Requires: pip install transformers
    """

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
        self._build_vocab()

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
        self._model_dim = config.n_embd

    def _build_vocab(self):
        special = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        all_tokens = special + self.SMILES_CHARS
        self.char2idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.pad_id = self.char2idx[self.PAD_TOKEN]
        self.bos_id = self.char2idx[self.BOS_TOKEN]
        self.eos_id = self.char2idx[self.EOS_TOKEN]
        self.sep_id = self.char2idx[self.SEP_TOKEN]

    def _encode_text(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def _decode_ids(self, ids: List[int]) -> str:
        chars = []
        for idx in ids:
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            c = self.idx2char.get(idx, "")
            chars.append(c)
        return "".join(chars)

    def _prepare_prompt(self, source_smi: str) -> List[int]:
        return [self.bos_id] + self._encode_text(source_smi) + [self.sep_id]

    def _prepare_pair(self, source_smi: str, target_smi: str) -> Tuple[List[int], int]:
        src_tokens = [self.bos_id] + self._encode_text(source_smi) + [self.sep_id]
        tgt_tokens = self._encode_text(target_smi) + [self.eos_id]
        target_start = len(src_tokens)
        return src_tokens + tgt_tokens, target_start

    def sample(
        self, source_smiles: List[str], batch_size: int = 64,
        temperature: float = 1.0,
    ) -> Tuple[List[str], List[str]]:
        self.model.eval()

        all_inputs = []
        all_outputs = []

        for src_smi in source_smiles:
            prompt_ids = self._prepare_prompt(src_smi)
            prompt_tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=self._device
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=prompt_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=256,
                    eos_token_id=self.eos_id,
                    pad_token_id=self.pad_id,
                )

            target_ids = outputs[0, len(prompt_ids):].tolist()
            smi = self._decode_ids(target_ids)
            all_inputs.append(src_smi)
            all_outputs.append(smi)

        return all_inputs, all_outputs

    def compute_log_probs(
        self, source_smiles: List[str], output_smiles: List[str],
    ) -> Tuple[torch.Tensor, List[int]]:
        sequences = []
        target_starts = []
        valid = []

        for i, (src, tgt) in enumerate(zip(source_smiles, output_smiles)):
            try:
                ids, tgt_start = self._prepare_pair(src, tgt)
                if len(ids) > 1:
                    sequences.append(ids)
                    target_starts.append(tgt_start)
                    valid.append(i)
            except Exception:
                continue

        if not sequences:
            return torch.tensor([]), []

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

        logits = outputs.logits
        log_probs_all = F.log_softmax(logits, dim=-1)

        batch_lps = []
        for i in range(len(sequences)):
            tgt_start = target_starts[i]
            seq_len = len(sequences[i])
            if tgt_start >= seq_len:
                batch_lps.append(torch.tensor(0.0, device=self._device))
                continue

            target_positions = torch.arange(tgt_start, seq_len, device=self._device)
            logit_positions = target_positions - 1
            target_token_ids = input_ids[i, target_positions]

            lp = log_probs_all[i, logit_positions, target_token_ids]
            batch_lps.append(lp.sum())

        return torch.stack(batch_lps), valid

    def get_encoder_output(
        self, source_smiles: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(1).bool()
        return hidden, mask

    def network_parameters(self):
        return self.model.parameters()

    def set_mode(self, mode: str):
        if mode == "training":
            self.model.train()
        else:
            self.model.eval()

    def set_eval(self):
        self.model.eval()

    def set_train(self):
        self.model.train()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if not path.endswith("/") else path, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "char2idx": self.char2idx,
        }, path)

    def d_model(self) -> int:
        return self._model_dim


# ==================================================================
# Backend factory
# ==================================================================

def create_backend(
    backend_name: str, device: torch.device, mode: str = "training",
) -> ModelBackend:
    """Create a ModelBackend instance by name.

    Args:
        backend_name: One of 'reinvent_mol2mol', 'molt5', 'smiles_gpt2'.
        device: Torch device.
        mode: 'training' or 'inference'.
    """
    if backend_name == "reinvent_mol2mol":
        return ReinventMol2MolBackend(str(PRIOR_PATH), device, mode=mode)
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


# ==================================================================
# ZAP70 source molecules
# ==================================================================

def load_zap70_sources():
    """Load ZAP70 active SMILES for use as source inputs."""
    actives_file = PROJECT_ROOT / "data" / "zap70_actives.smi"
    if actives_file.exists():
        smiles = [line.strip() for line in open(actives_file) if line.strip()]
        print(f"[ppo] Loaded {len(smiles)} ZAP70 actives from {actives_file}")
        return smiles

    # Fall back: extract from the scorer's training data
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    smiles = smiles_df["smiles"].tolist()

    # Cache for future runs
    actives_file.parent.mkdir(parents=True, exist_ok=True)
    with open(actives_file, "w") as f:
        for s in smiles:
            f.write(s + "\n")
    print(f"[ppo] Extracted and cached {len(smiles)} ZAP70 actives")
    return smiles


# ==================================================================
# Rollout: sample, score, compute log-probs + values
# ==================================================================

def run_rollout(
    agent: ModelBackend,
    ref_model: ModelBackend,
    critic,
    src_pool,
    film_model,
    film_scaler,
    film_anchors,
    film_pIC50,
    device,
):
    """Execute one rollout: sample molecules, score, compute log-probs and values.

    Returns dict with keys:
        output_smiles, source_smiles, rewards, old_log_probs, values,
        ref_log_probs, valid_mask, n_valid, n_total, raw_scores
    """
    # Sample source molecules
    indices = np.random.choice(len(src_pool), size=BATCH_SIZE, replace=True)
    source_smiles = [src_pool[i] for i in indices]

    # Sample from agent
    agent.set_mode("inference")
    agent.set_eval()
    input_smi_list, output_smi_list = agent.sample(
        source_smiles, batch_size=BATCH_SIZE, temperature=TEMPERATURE,
    )
    agent.set_mode("training")

    if not output_smi_list:
        return None

    # Filter invalid SMILES
    valid_outputs = []
    valid_inputs = []
    valid_out_indices = []
    for i, (inp, out) in enumerate(zip(input_smi_list, output_smi_list)):
        out_clean = out.strip()
        if not out_clean:
            continue
        mol = Chem.MolFromSmiles(out_clean)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can:
            valid_outputs.append(can)
            valid_inputs.append(inp)
            valid_out_indices.append(i)

    if len(valid_outputs) < 2:
        return None

    # Score with FiLMDelta
    scores = score_smiles(valid_outputs, film_model, film_scaler, film_anchors, film_pIC50)
    rewards = np.array(scores, dtype=np.float32)

    # Replace NaN rewards with minimum
    nan_mask = np.isnan(rewards)
    if nan_mask.any():
        rewards[nan_mask] = np.nanmin(rewards) if not np.all(nan_mask) else 0.0

    # Normalize rewards (zero-mean, unit-std)
    if rewards.std() > 1e-6:
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    rewards_tensor = torch.FloatTensor(rewards).to(device)

    # Compute log-probs under current policy
    agent.set_train()
    old_log_probs, lp_valid = agent.compute_log_probs(valid_inputs, valid_outputs)

    if len(old_log_probs) == 0:
        return None

    # Align rewards with successfully computed log-probs
    rewards_aligned = rewards_tensor[lp_valid]
    valid_inputs_aligned = [valid_inputs[i] for i in lp_valid]
    valid_outputs_aligned = [valid_outputs[i] for i in lp_valid]

    # Compute values from critic
    agent.set_eval()
    with torch.no_grad():
        encoder_out, enc_mask = agent.get_encoder_output(valid_inputs_aligned)
        values = critic(encoder_out, enc_mask)

    # Compute reference log-probs for KL penalty
    with torch.no_grad():
        ref_log_probs, _ = ref_model.compute_log_probs(
            valid_inputs_aligned, valid_outputs_aligned,
        )

    old_log_probs = old_log_probs.detach()

    return {
        "source_smiles": valid_inputs_aligned,
        "output_smiles": valid_outputs_aligned,
        "rewards": rewards_aligned.detach(),
        "old_log_probs": old_log_probs,
        "values": values.detach(),
        "ref_log_probs": ref_log_probs.detach(),
        "n_valid": len(valid_outputs),
        "n_total": len(output_smi_list),
        "raw_scores": scores,
    }


# ==================================================================
# PPO update
# ==================================================================

def ppo_update(
    agent: ModelBackend,
    critic,
    actor_optimizer,
    critic_optimizer,
    rollout_data,
    device,
    beta_kl,
):
    """Run PPO update: multiple mini-epochs over the rollout data."""
    source_smiles = rollout_data["source_smiles"]
    output_smiles = rollout_data["output_smiles"]
    rewards = rollout_data["rewards"]
    old_log_probs = rollout_data["old_log_probs"]
    old_values = rollout_data["values"]
    ref_log_probs = rollout_data["ref_log_probs"]

    # Advantages: A = reward - V(s)  (no GAE, single-step)
    advantages = rewards - old_values
    returns = rewards

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_kl = 0.0
    total_loss = 0.0
    n_updates = 0

    for _epoch in range(PPO_EPOCHS):
        # Compute new log-probs
        agent.set_train()
        new_log_probs, valid = agent.compute_log_probs(source_smiles, output_smiles)

        if len(new_log_probs) == 0:
            continue

        # Align tensors to valid indices
        adv = advantages[valid]
        ret = returns[valid]
        old_lp = old_log_probs[valid]
        ref_lp = ref_log_probs[valid]

        # Policy ratio
        ratio = torch.exp(new_log_probs - old_lp)

        # Clipped surrogate objective
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL penalty: mean(log pi_theta - log pi_ref)
        kl = (new_log_probs - ref_lp).mean()

        # Value loss
        valid_sources = [source_smiles[i] for i in valid]
        encoder_out, enc_mask = agent.get_encoder_output(valid_sources)
        new_values = critic(encoder_out, enc_mask)
        value_loss = F.mse_loss(new_values, ret)

        # Combined loss
        loss = policy_loss + VALUE_COEFF * value_loss + beta_kl * kl

        # Update actor
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(list(agent.network_parameters()), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)

        actor_optimizer.step()
        critic_optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_kl += kl.item()
        total_loss += loss.item()
        n_updates += 1

    # Adaptive KL penalty
    avg_kl = total_kl / max(n_updates, 1)
    if avg_kl > 1.5 * KL_TARGET:
        beta_kl = min(beta_kl * 2.0, BETA_KL_MAX)
    elif avg_kl < KL_TARGET / 1.5:
        beta_kl = max(beta_kl / 2.0, BETA_KL_MIN)

    stats = {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "kl": avg_kl,
        "total_loss": total_loss / max(n_updates, 1),
        "beta_kl": beta_kl,
        "n_updates": n_updates,
    }

    return stats, beta_kl


# ==================================================================
# Final generation + evaluation
# ==================================================================

def generate_final(
    agent: ModelBackend, src_pool, device, n_samples=5000,
):
    """Sample molecules from the trained policy."""
    agent.set_mode("inference")
    agent.set_eval()

    all_outputs = []
    all_inputs = []
    batch = 128

    with torch.no_grad():
        while len(all_outputs) < n_samples:
            indices = np.random.choice(len(src_pool), size=batch, replace=True)
            sources = [src_pool[i] for i in indices]
            inp_list, out_list = agent.sample(
                sources, batch_size=batch, temperature=TEMPERATURE,
            )
            for inp, out in zip(inp_list, out_list):
                out_clean = out.strip()
                mol = Chem.MolFromSmiles(out_clean)
                if mol is not None:
                    can = Chem.MolToSmiles(mol)
                    if can:
                        all_outputs.append(can)
                        all_inputs.append(inp)
            print(f"\r[ppo] Generated {len(all_outputs)}/{n_samples}", end="", flush=True)

    print()
    return all_outputs[:n_samples], all_inputs[:n_samples]


def evaluate_generation(
    smiles_list, film_model, film_scaler, film_anchors, film_pIC50, label="PPO"
):
    """Score and summarize a set of generated molecules."""
    scores = score_smiles(smiles_list, film_model, film_scaler, film_anchors, film_pIC50)
    scores_arr = np.array(scores, dtype=np.float64)
    valid_mask = ~np.isnan(scores_arr)
    valid_scores = scores_arr[valid_mask]

    unique_smiles = set(smiles_list)
    n_total = len(smiles_list)
    n_valid = int(valid_mask.sum())
    n_unique = len(unique_smiles)

    stats = {
        "label": label,
        "n_total": n_total,
        "n_valid": n_valid,
        "validity": n_valid / max(n_total, 1),
        "n_unique": n_unique,
        "uniqueness": n_unique / max(n_valid, 1),
        "mean_pIC50": float(np.mean(valid_scores)) if n_valid > 0 else float("nan"),
        "std_pIC50": float(np.std(valid_scores)) if n_valid > 0 else float("nan"),
        "median_pIC50": float(np.median(valid_scores)) if n_valid > 0 else float("nan"),
        "max_pIC50": float(np.max(valid_scores)) if n_valid > 0 else float("nan"),
        "min_pIC50": float(np.min(valid_scores)) if n_valid > 0 else float("nan"),
        "frac_ge_6": float((valid_scores >= 6.0).mean()) if n_valid > 0 else 0.0,
        "frac_ge_7": float((valid_scores >= 7.0).mean()) if n_valid > 0 else 0.0,
        "frac_ge_8": float((valid_scores >= 8.0).mean()) if n_valid > 0 else 0.0,
    }

    print(f"\n{'='*60}")
    print(f"  {label} Generation Summary")
    print(f"{'='*60}")
    print(f"  Total: {n_total}, Valid: {n_valid} ({stats['validity']:.1%})")
    print(f"  Unique: {n_unique} ({stats['uniqueness']:.1%})")
    print(f"  pIC50: {stats['mean_pIC50']:.3f} +/- {stats['std_pIC50']:.3f}")
    print(f"  Max pIC50: {stats['max_pIC50']:.3f}")
    print(f"  >= 6.0: {stats['frac_ge_6']:.1%}, >= 7.0: {stats['frac_ge_7']:.1%}, >= 8.0: {stats['frac_ge_8']:.1%}")
    print(f"{'='*60}\n")

    return stats, scores_arr


# ==================================================================
# Main
# ==================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO molecular generation with multiple model backends"
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
    model_backend = args.model

    # Adjust results dir to include model name
    if model_backend == "reinvent_mol2mol":
        results_dir = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_ppo"
    else:
        results_dir = PROJECT_ROOT / "results" / "paper_evaluation" / f"reinvent4_ppo_{model_backend}"
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"[ppo] Starting PPO molecular generation pipeline (backend: {model_backend})")
    print(f"[ppo] Results: {results_dir}")
    print()

    # ------------------------------------------------------------------
    # 1. Load FiLMDelta scorer
    # ------------------------------------------------------------------
    print("[ppo] Loading FiLMDelta scorer...")
    film_model, film_scaler, film_anchors, film_pIC50 = load_film_model()
    print(f"[ppo] FiLMDelta loaded: {len(film_pIC50)} anchors")

    # ------------------------------------------------------------------
    # 2. Load ZAP70 source molecules
    # ------------------------------------------------------------------
    src_pool = load_zap70_sources()

    # ------------------------------------------------------------------
    # 3. Load model backends (agent + reference)
    # ------------------------------------------------------------------
    print(f"[ppo] Loading {model_backend} agent...")
    agent = create_backend(model_backend, DEVICE, mode="training")

    print(f"[ppo] Loading {model_backend} reference (frozen)...")
    ref_model = create_backend(model_backend, DEVICE, mode="inference")
    for p in ref_model.network_parameters():
        p.requires_grad = False

    d_model_val = agent.d_model()
    print(f"[ppo] Model d_model={d_model_val}")

    # ------------------------------------------------------------------
    # 4. Initialize critic (adapted to backend's d_model)
    # ------------------------------------------------------------------
    critic = SmilesCritic(d_model=d_model_val, hidden_dim=128).to(DEVICE)
    print(f"[ppo] Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")

    # ------------------------------------------------------------------
    # 5. Optimizers
    # ------------------------------------------------------------------
    actor_optimizer = torch.optim.Adam(agent.network_parameters(), lr=ACTOR_LR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # ------------------------------------------------------------------
    # 6. PPO training loop
    # ------------------------------------------------------------------
    print(f"\n[ppo] Starting PPO training: {TOTAL_ROLLOUTS} rollouts x {BATCH_SIZE} batch")
    print(f"[ppo] PPO epochs: {PPO_EPOCHS}, clip: {CLIP_EPS}, KL target: {KL_TARGET}")
    print()

    beta_kl = BETA_KL_INIT
    training_log = []
    all_rewards = []

    for rollout_idx in range(TOTAL_ROLLOUTS):
        t0 = time.time()

        # --- Rollout ---
        rollout_data = run_rollout(
            agent, ref_model, critic, src_pool,
            film_model, film_scaler, film_anchors, film_pIC50,
            DEVICE,
        )

        if rollout_data is None:
            print(f"[ppo] Rollout {rollout_idx+1}: no valid samples, skipping")
            continue

        # --- PPO Update ---
        stats, beta_kl = ppo_update(
            agent, critic, actor_optimizer, critic_optimizer,
            rollout_data, DEVICE, beta_kl,
        )

        # --- Logging ---
        raw_scores = rollout_data["raw_scores"]
        valid_scores = [s for s in raw_scores if not np.isnan(s)]
        mean_reward = np.mean(valid_scores) if valid_scores else float("nan")
        max_reward = np.max(valid_scores) if valid_scores else float("nan")
        all_rewards.extend(valid_scores)

        log_entry = {
            "rollout": rollout_idx + 1,
            "n_valid": rollout_data["n_valid"],
            "n_total": rollout_data["n_total"],
            "mean_pIC50": float(mean_reward),
            "max_pIC50": float(max_reward),
            **stats,
            "time_s": time.time() - t0,
        }
        training_log.append(log_entry)

        if (rollout_idx + 1) % 10 == 0 or rollout_idx == 0:
            print(
                f"  Rollout {rollout_idx+1:4d}/{TOTAL_ROLLOUTS} | "
                f"valid {rollout_data['n_valid']:3d}/{rollout_data['n_total']:3d} | "
                f"pIC50 {mean_reward:.3f} (max {max_reward:.3f}) | "
                f"P_loss {stats['policy_loss']:.4f} | "
                f"V_loss {stats['value_loss']:.4f} | "
                f"KL {stats['kl']:.4f} | "
                f"beta {stats['beta_kl']:.4f} | "
                f"{time.time()-t0:.1f}s"
            )

        # Periodic checkpoint
        if (rollout_idx + 1) % 50 == 0:
            if model_backend == "reinvent_mol2mol":
                ckpt_path = results_dir / f"agent_rollout_{rollout_idx+1}.pt"
            else:
                ckpt_path = results_dir / f"agent_rollout_{rollout_idx+1}"
            agent.save(str(ckpt_path))
            print(f"  [checkpoint] Saved agent to {ckpt_path}")

        gc.collect()

    # Save final agent
    if model_backend == "reinvent_mol2mol":
        agent.save(str(results_dir / "agent_final.pt"))
    else:
        agent.save(str(results_dir / "agent_final"))

    # Save training log
    log_path = results_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n[ppo] Training log saved to {log_path}")

    # ------------------------------------------------------------------
    # 7. Final generation
    # ------------------------------------------------------------------
    print(f"\n[ppo] Generating {FINAL_SAMPLE_SIZE} molecules from trained policy...")
    final_smiles, final_inputs = generate_final(
        agent, src_pool, DEVICE, n_samples=FINAL_SAMPLE_SIZE,
    )

    ppo_stats, ppo_scores = evaluate_generation(
        final_smiles, film_model, film_scaler, film_anchors, film_pIC50, label="PPO"
    )

    # Save generated SMILES
    smiles_path = results_dir / "ppo_generated.smi"
    with open(smiles_path, "w") as f:
        for smi in final_smiles:
            f.write(smi + "\n")
    print(f"[ppo] Generated SMILES saved to {smiles_path}")

    # Save scores
    scores_path = results_dir / "ppo_scores.json"
    with open(scores_path, "w") as f:
        json.dump(
            [{"smiles": s, "pIC50": float(sc)} for s, sc in zip(final_smiles, ppo_scores)],
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 8. Comparison with DAP / DPO baselines (if results exist)
    # ------------------------------------------------------------------
    comparison = {"ppo": ppo_stats, "model_backend": model_backend}

    # Check for DAP results from run_reinvent4_generation.py
    dap_dir = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4" / "mol2mol"
    dap_smiles_file = dap_dir / "generated_smiles.smi"
    if dap_smiles_file.exists():
        print("[ppo] Found DAP baseline results, scoring for comparison...")
        dap_smiles = [line.strip() for line in open(dap_smiles_file) if line.strip()]
        if dap_smiles:
            dap_stats, _ = evaluate_generation(
                dap_smiles[:FINAL_SAMPLE_SIZE],
                film_model, film_scaler, film_anchors, film_pIC50,
                label="DAP (Mol2Mol)",
            )
            comparison["dap"] = dap_stats

    # Check for DPO results
    dpo_dir = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_dpo"
    dpo_smiles_file = dpo_dir / "dpo_generated.smi"
    if dpo_smiles_file.exists():
        print("[ppo] Found DPO baseline results, scoring for comparison...")
        dpo_smiles = [line.strip() for line in open(dpo_smiles_file) if line.strip()]
        if dpo_smiles:
            dpo_stats, _ = evaluate_generation(
                dpo_smiles[:FINAL_SAMPLE_SIZE],
                film_model, film_scaler, film_anchors, film_pIC50,
                label="DPO",
            )
            comparison["dpo"] = dpo_stats

    # Save comparison
    comparison_path = results_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[ppo] Comparison saved to {comparison_path}")

    # ------------------------------------------------------------------
    # 9. Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  PPO Generation Complete (backend: {model_backend})")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  Rollouts: {TOTAL_ROLLOUTS}, Molecules generated: {TOTAL_ROLLOUTS * BATCH_SIZE}")
    print(f"  Final sample: {len(final_smiles)} molecules")
    print(f"  Mean pIC50: {ppo_stats['mean_pIC50']:.3f}")
    print(f"  Max pIC50: {ppo_stats['max_pIC50']:.3f}")
    print(f"  Potent (>=7.0): {ppo_stats['frac_ge_7']:.1%}")
    for label, stats in comparison.items():
        if isinstance(stats, dict) and "label" in stats and label != "ppo":
            print(f"  {stats['label']}: mean={stats['mean_pIC50']:.3f}, max={stats['max_pIC50']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
