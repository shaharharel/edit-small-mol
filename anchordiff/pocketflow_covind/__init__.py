"""Covalent-aware extension to PocketFlow for ZAP70 Cys346 inhibitor design.

Files:
  inpaint.py   - warhead-seeded autoregressive inpainting (vanilla ckpt)
  train_dc.py  - finetune with C-arm + D-arm soft loss
  sample.py    - generate from finetuned ckpt with the same seed
  warhead_seed.py - shared helper: build a 5-atom acrylamide seed Ligand dict
  cov_token.py  - 290-d covalent token (ported from anchordiff.covind v2.5)
"""
