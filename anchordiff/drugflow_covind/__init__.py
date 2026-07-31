"""DrugFlow + Covalent (C-arm chemistry token, D-arm position pin, joint
atom+bond inpainting via Markov-bridge logit biasing).

Three deliverables live here:
  - inpaint.py   : soft warhead atom-position + atom-type + bond-type pinning
                   wrapped around DrugFlow.sample
  - train_dc.py  : finetune wrapper (D-arm jitter + soft pose loss, C-arm token
                   biasing pocket features) on the CovBinder CSV
  - sample.py    : sample 25 ZAP70 mols from finetuned ckpt

All three reuse the proven ported pieces from anchordiff/covind:
  - covalent_token_v2_5 (290-d C-arm token)
  - cov_adapter_v2_5    (token → pocket-feature bias)
  - local_frame         (Cys-SG/CB-anchored SE(3) frame — derived from input atoms)
"""
