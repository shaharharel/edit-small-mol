"""L2 EXTENDED SCAFFOLD-ANCHOR INPAINT — scaffold-anchor extension sweep
implementing the medchem reviewer's two highest-leverage recommendations
from the L2 C5 cohort review:

  (1) Linker-locked anchor (anchor_id="L") — forces the direct amide
      `c–C(=O)–[*]` instead of the C5 cohort's ethylene-amide
      `c–CH2–CH2–NH–C(=O)–...`. This recovers ~0.7-1.4 kcal/mol entropy and
      places the next-emitted H-bond donor at ~5 A from the warhead (in line
      with MM-GBSA winners) instead of ~7.5 A.

  (2) Heteroaryl-stub anchors (H1 / H2 / H3) — pre-populate the kinase
      hinge-binder (aminopyridine, oxadiazole, thiazole) so the model only
      decodes the distal capping group. H1/H2 directly seed the chassis of
      the C5-cohort MM-GBSA winners.

Anchors (each cohort: 30 mols @ T=1.0):

  L  : C=CC(=O)N1Cc2cccc(C(=O)[*])c2C1
       15 heavy atoms, 31 tokens. [*] on the non-aromatic carbonyl C.
  H1 : C=CC(=O)N1Cc2cccc(C(=O)Nc3ccc([*])nc3)c2C1
       22 heavy atoms, 42 tokens. [*] on the 4-position of 2-aminopyridine.
  H2 : C=CC(=O)N1Cc2cccc(C(=O)N3CCC(c4nnc([*])o4)CC3)c2C1
       26 heavy atoms, 50 tokens. [*] on the 5-position of 1,3,4-oxadiazole.
  H3 : C=CC(=O)N1Cc2cccc(Cc3csc([*])n3)c2C1
       19 heavy atoms, 36 tokens. [*] on the 2-position of the thiazole.

All four anchors decode cleanly with the FSMILES vocab (verified at import
time). All four pin atom 0 (terminal CH2= of the acrylamide) at
cb_pos_target with atom0->atom1 aligned to the Burgi-Dunitz attack vector.

Run target (Mac CPU, ~10 min/anchor):

  python experiments/run_lingo3dmol_l2_extended_anchor.py \
      --pocket_pdb data/lingo3dmol_smoke/zap70_pocket_cys346.pdb \
      --anchor_id L \
      --output data/lingo3dmol_L2_extended_L/samples_T10.sdf \
      --gennums 30 --min_acceptable 15 --tempture 1.0
"""
from __future__ import annotations
import os, sys, time, argparse, json
from pathlib import Path
import numpy as np

# Install CPU shim FIRST, before any Lingo3DMol model import
ROOT = Path(__file__).resolve().parent.parent
ORIG_CWD = Path(os.getcwd()).resolve()
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "external" / "Lingo3DMol"))
os.chdir(str(ROOT / "external" / "Lingo3DMol"))


def _abs(p):
    p = Path(p)
    if p.is_absolute():
        return str(p.resolve())
    cand = (ORIG_CWD / p).resolve()
    if cand.exists():
        return str(cand)
    cand2 = (ROOT / p).resolve()
    if cand2.exists():
        return str(cand2)
    return str(cand)


import lingo3dmol_cpu_shim  # noqa: F401

import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from util.fragmol_frag_zyh import FragmolUtil
from model.transformer_v1_res_mp1 import TransformerModel as TransformerModel_contact, topkp_random
from model.transformer_v1_res_fac2 import TransformerModel
from dataloader.dataloader_case_nci_res_merge import mydataset as testdataset
from torch.utils.data import DataLoader
from inference.cube_collision_check import CollisionCheck

from run_lingo3dmol_l0_smoke import (
    changepos2, bonusVSpenalty2, get_partial_to_warehouse,
    molecular_workflow, write_product, go_factory,
    _open_incremental_writer, _flush_summary,
)
from anchor_geometry import (
    rotate_scaffold_to_bd, assert_bd_angle, measured_bd_angle,
)


_FU = FragmolUtil()
_V = _FU.vocab_c2i_v1_decode_new


# =====================================================================
# ANCHOR DEFINITIONS
# =====================================================================
# For each anchor:
#   smiles        : canonical SMILES with [*] (open valence)
#   scaffold_smi  : same molecule WITHOUT [*] (for ETKDG embedding)
#   n_heavy       : heavy-atom count of scaffold_smi (== len(atom_index_map))
#   token_labels  : FSMILES token sequence including 'start_0' .. 'sep_0'
#   atom_index_map: {token_position_in_sequence: heavy_atom_index_in_scaffold_smi}

ANCHORS = {
    # -----------------------------------------------------------------
    # Anchor L (Linker-locked): C=CC(=O)N1Cc2cccc(C(=O)[*])c2C1
    # Atom indices (15 heavy atoms in scaffold without [*]):
    #   0:C(CH2=) 1:C 2:C 3:O 4:N(isoind) 5:C 6:c-junc 7:c 8:c 9:c 10:c(sub)
    #   11:C(carbonyl2) 12:O 13:c-junc 14:C(sp3)
    # -----------------------------------------------------------------
    "L": {
        "smiles": "C=CC(=O)N1Cc2cccc(C(=O)[*])c2C1",
        "scaffold_smi": "C=CC(=O)N1Cc2cccc(C(=O))c2C1",  # invalid: bare C(=O)
        # We embed the FULL anchor WITHOUT the [*] dummy via RDKit by
        # parsing the *-substituted SMILES and removing the * atom.
        "n_heavy": 15,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',                       # 0,1
            'C_0', '(_0', '=_0', 'O_0', ')_0',         # 2,3
            'N_5', '1_0',                              # 4
            'C_5',                                     # 5
            'c_5', '2_0',                              # 6 (junction)
            'c_6', 'c_6', 'c_6',                       # 7,8,9
            'c_6',                                     # 10 (C5)
            '(_0',
                'C_0', '(_0', '=_0', 'O_0', ')_0',     # 11, 12
                '[*]_0',
            ')_0',
            'c_5', '2_0',                              # 13 (junction)
            'C_5', '1_0',                              # 14 (sp3 CH)
            'sep_0',
        ],
        # token position -> heavy-atom index in the scaffold_smi (without [*])
        "atom_index_map": {
            1: 0,   # C_0
            3: 1,   # C_0
            4: 2,   # C_0 (carbonyl1)
            7: 3,   # O_0
            9: 4,   # N_5
            11: 5,  # C_5
            12: 6,  # c_5 (junction)
            14: 7,  # c_6
            15: 8,  # c_6
            16: 9,  # c_6
            17: 10, # c_6 (C5)
            19: 11, # C_0 (carbonyl2)
            22: 12, # O_0 (carbonyl2)
            26: 13, # c_5 (junction back)
            28: 14, # C_5 (sp3 CH)
        },
        "star_attach_aromatic": False,  # [*] sits on sp2 carbonyl C
    },
    # -----------------------------------------------------------------
    # Anchor H1 (aminopyridine): C=CC(=O)N1Cc2cccc(C(=O)Nc3ccc([*])nc3)c2C1
    # Atom indices (22 heavy atoms in scaffold without [*]):
    #   0:C 1:C 2:C 3:O 4:N(isoind) 5:C 6:c-junc 7:c 8:c 9:c 10:c(sub)
    #   11:C(carbonyl2) 12:O 13:N(amide,non-ring) 14:c(pyr-attach-N) 15:c 16:c
    #   17:c(C-4 with [*]) 18:n 19:c(close ring3) 20:c-junc(close ring2) 21:C(sp3)
    # -----------------------------------------------------------------
    "H1": {
        "smiles": "C=CC(=O)N1Cc2cccc(C(=O)Nc3ccc([*])nc3)c2C1",
        "scaffold_smi": None,  # built by removing [*] from smiles
        "n_heavy": 22,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',                       # 0,1
            'C_0', '(_0', '=_0', 'O_0', ')_0',         # 2,3
            'N_5', '1_0',                              # 4
            'C_5',                                     # 5
            'c_5', '2_0',                              # 6
            'c_6', 'c_6', 'c_6',                       # 7,8,9
            'c_6',                                     # 10
            '(_0',
                'C_0', '(_0', '=_0', 'O_0', ')_0',     # 11, 12
                'N_0',                                 # 13 (amide N, non-ring)
                'c_6', '3_0',                          # 14 (pyr C attached to N)
                'c_6', 'c_6',                          # 15, 16
                'c_6',                                 # 17 (C-4)
                '(_0', '[*]_0', ')_0',
                'n_6',                                 # 18
                'c_6', '3_0',                          # 19 (close ring3)
            ')_0',
            'c_5', '2_0',                              # 20
            'C_5', '1_0',                              # 21
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 11: 5,
            12: 6, 14: 7, 15: 8, 16: 9, 17: 10,
            19: 11, 22: 12,
            24: 13,                # N_0 amide
            25: 14,                # c_6 (pyr-1)
            27: 15, 28: 16,        # c_6, c_6
            29: 17,                # c_6 (C-4)
            33: 18,                # n_6
            34: 19,                # c_6 (close ring3)
            37: 20,                # c_5 (junction back)
            39: 21,                # C_5 (sp3 CH)
        },
        "star_attach_aromatic": True,  # [*] sits on aromatic pyridine C
    },
    # -----------------------------------------------------------------
    # Anchor H2 (oxadiazole): C=CC(=O)N1Cc2cccc(C(=O)N3CCC(c4nnc([*])o4)CC3)c2C1
    # Atom indices (26 heavy atoms in scaffold without [*]):
    #   0:C 1:C 2:C 3:O 4:N(isoind) 5:C 6:c-junc 7:c 8:c 9:c 10:c(sub)
    #   11:C(carbonyl2) 12:O 13:N(pip,ring) 14:C 15:C 16:C(branch)
    #   17:c(oxa-1) 18:n 19:n 20:c([*]) 21:o(close ring4) 22:C 23:C(close pip ring3)
    #   24:c-junc(close ring2) 25:C(sp3)
    # -----------------------------------------------------------------
    "H2": {
        "smiles": "C=CC(=O)N1Cc2cccc(C(=O)N3CCC(c4nnc([*])o4)CC3)c2C1",
        "scaffold_smi": None,
        "n_heavy": 26,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_5', '1_0',
            'C_5',
            'c_5', '2_0',
            'c_6', 'c_6', 'c_6',
            'c_6',
            '(_0',
                'C_0', '(_0', '=_0', 'O_0', ')_0',         # 11, 12
                'N_6', '3_0',                              # 13 (pip N, 6-ring)
                'C_6', 'C_6', 'C_6',                       # 14, 15, 16
                '(_0',
                    'c_5', '4_0',                          # 17 (oxa C)
                    'n_5', 'n_5',                          # 18, 19
                    'c_5',                                 # 20 (has [*])
                    '(_0', '[*]_0', ')_0',
                    'o_5', '4_0',                          # 21 (close ring4)
                ')_0',
                'C_6', 'C_6', '3_0',                       # 22, 23 (close pip ring3)
            ')_0',
            'c_5', '2_0',                                  # 24
            'C_5', '1_0',                                  # 25
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 11: 5,
            12: 6, 14: 7, 15: 8, 16: 9, 17: 10,
            19: 11, 22: 12,
            24: 13,                # N_6 (pip)
            26: 14, 27: 15, 28: 16,# C_6 C_6 C_6
            30: 17,                # c_5 (oxa-1)
            32: 18, 33: 19,        # n_5 n_5
            34: 20,                # c_5 (has [*])
            38: 21,                # o_5 (close ring4)
            41: 22, 42: 23,        # C_6 C_6 (pip close)
            45: 24,                # c_5 (junction back)
            47: 25,                # C_5 (sp3 CH)
        },
        "star_attach_aromatic": True,  # [*] sits on aromatic oxadiazole C
    },
    # -----------------------------------------------------------------
    # Anchor H3 (thiazole): C=CC(=O)N1Cc2cccc(Cc3csc([*])n3)c2C1
    # Atom indices (19 heavy atoms in scaffold without [*]):
    #   0:C 1:C 2:C 3:O 4:N 5:C 6:c-junc 7:c 8:c 9:c 10:c(sub)
    #   11:C(methylene,non-ring) 12:c(thi-1) 13:c 14:s 15:c([*]) 16:n(close ring3)
    #   17:c-junc(close ring2) 18:C(sp3)
    # -----------------------------------------------------------------
    "H3": {
        "smiles": "C=CC(=O)N1Cc2cccc(Cc3csc([*])n3)c2C1",
        "scaffold_smi": None,
        "n_heavy": 19,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_5', '1_0',
            'C_5',
            'c_5', '2_0',
            'c_6', 'c_6', 'c_6',
            'c_6',
            '(_0',
                'C_0',                                  # 11 (methylene)
                'c_5', '3_0',                           # 12 (thi-1)
                'c_5',                                  # 13
                's_5',                                  # 14
                'c_5',                                  # 15 (has [*])
                '(_0', '[*]_0', ')_0',
                'n_5', '3_0',                           # 16 (close ring3)
            ')_0',
            'c_5', '2_0',                               # 17
            'C_5', '1_0',                               # 18
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 11: 5,
            12: 6, 14: 7, 15: 8, 16: 9, 17: 10,
            19: 11,                # C_0 methylene
            20: 12,                # c_5 (thi-1)
            22: 13,                # c_5
            23: 14,                # s_5
            24: 15,                # c_5 ([*] attaches here)
            28: 16,                # n_5 (close ring3)
            31: 17,                # c_5 (junction back)
            33: 18,                # C_5 (sp3 CH)
        },
        "star_attach_aromatic": True,  # [*] sits on aromatic thiazole C
    },
    # =================================================================
    # ZAP70 multi-chassis ensemble (Phase A-mined + published kinase chassis)
    # All six decode-verified; atom 0 = vinyl CH2 (terminal).
    # See data/zap70_chassis_families.json for provenance.
    # =================================================================
    # ZAP_C1: meta aniline-acrylamide — 11 ChEMBL ZAP70 actives, median pIC50 7.04
    # Scaffold (no [*]): C=CC(=O)Nc1ccccc1 — 11 heavy atoms
    "ZAP_C1": {
        "smiles": "C=CC(=O)Nc1cccc([*])c1",
        "scaffold_smi": None,
        "n_heavy": 11,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',                       # 0, 1
            'C_0', '(_0', '=_0', 'O_0', ')_0',         # 2, 3
            'N_0',                                     # 4 (amide N, non-ring)
            'c_6', '1_0',                              # 5 (ring open)
            'c_6', 'c_6', 'c_6',                       # 6, 7, 8
            'c_6',                                     # 9 (has [*])
            '(_0', '[*]_0', ')_0',
            'c_6', '1_0',                              # 10 (close ring)
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 10: 5,
            12: 6, 13: 7, 14: 8, 15: 9, 19: 10,
        },
        "star_attach_aromatic": True,
    },
    # ZAP_C2: ortho aniline-acrylamide — 8 ChEMBL ZAP70 actives
    # Scaffold: C=CC(=O)Nc1ccccc1 — 11 heavy atoms
    "ZAP_C2": {
        "smiles": "C=CC(=O)Nc1ccccc1[*]",
        "scaffold_smi": None,
        "n_heavy": 11,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_0',
            'c_6', '1_0',                              # 5
            'c_6', 'c_6', 'c_6', 'c_6',                # 6, 7, 8, 9
            'c_6', '1_0',                              # 10 (close, has [*])
            '[*]_0',
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 10: 5,
            12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
        },
        "star_attach_aromatic": True,
    },
    # ZAP_C3: BTK-ibrutinib chassis — piperidine-N-acrylamide, [*] at C2
    # Scaffold: C=CC(=O)N1CCCCC1 — 10 heavy atoms
    "ZAP_C3": {
        "smiles": "C=CC(=O)N1CCCCC1[*]",
        "scaffold_smi": None,
        "n_heavy": 10,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_6', '1_0',                              # 4 (pip N, 6-ring)
            'C_6', 'C_6', 'C_6', 'C_6',                # 5, 6, 7, 8
            'C_6', '1_0',                              # 9 (close, has [*])
            '[*]_0',
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4,
            11: 5, 12: 6, 13: 7, 14: 8, 15: 9,
        },
        "star_attach_aromatic": False,  # sp3 piperidine C
    },
    # ZAP_C4: PRN694-style pyrrolidine-N-acrylamide, [*] at C2
    # Scaffold: C=CC(=O)N1CCCC1 — 9 heavy atoms
    "ZAP_C4": {
        "smiles": "C=CC(=O)N1CCCC1[*]",
        "scaffold_smi": None,
        "n_heavy": 9,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_5', '1_0',                              # 4 (pyr N, 5-ring)
            'C_5', 'C_5', 'C_5',                       # 5, 6, 7
            'C_5', '1_0',                              # 8 (close, has [*])
            '[*]_0',
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4,
            11: 5, 12: 6, 13: 7, 14: 8,
        },
        "star_attach_aromatic": False,  # sp3 pyrrolidine C
    },
    # ZAP_C5: BMX/FGFR4 pyridine-amino-acrylamide chassis
    # Scaffold: C=CC(=O)Nc1cccnc1 — 11 heavy atoms
    "ZAP_C5": {
        "smiles": "C=CC(=O)Nc1cnccc1[*]",
        "scaffold_smi": None,
        "n_heavy": 11,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_0',
            'c_6', '1_0',                              # 5
            'c_6', 'n_6', 'c_6', 'c_6',                # 6, 7 (pyridine N), 8, 9
            'c_6', '1_0',                              # 10 (close, has [*])
            '[*]_0',
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 10: 5,
            12: 6, 13: 7, 14: 8, 15: 9, 16: 10,
        },
        "star_attach_aromatic": True,
    },
    # ZAP_C6: C4-piperidine-N-acrylamide (vs C2 of ZAP_C3) — geometric diversifier
    # Scaffold: C=CC(=O)N1CCCCC1 — 10 heavy atoms (same as C3, different [*] position)
    "ZAP_C6": {
        "smiles": "C=CC(=O)N1CCC([*])CC1",
        "scaffold_smi": None,
        "n_heavy": 10,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_6', '1_0',                              # 4
            'C_6', 'C_6',                              # 5, 6
            'C_6',                                     # 7 (has [*])
            '(_0', '[*]_0', ')_0',
            'C_6', 'C_6', '1_0',                       # 8, 9 (close)
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4,
            11: 5, 12: 6, 13: 7, 17: 8, 18: 9,
        },
        "star_attach_aromatic": False,  # sp3 piperidine C4
    },
    # =================================================================
    # Cross-target generalization anchors (Nature Methods gap)
    # =================================================================
    # BTK_C481: ibrutinib-class chassis — acrylamide + C4-piperidine, [*] at C4
    # Identical structurally to ZAP_C6 (piperidine-N-acrylamide with C4 attach
    # point) but reserved as a dedicated ID for the BTK Cys481 ATP-pocket
    # cross-target run. Scaffold without [*]: C=CC(=O)N1CCCCC1 — 10 heavy atoms.
    "BTK_C481": {
        "smiles": "C=CC(=O)N1CCC([*])CC1",
        "scaffold_smi": None,
        "n_heavy": 10,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_6', '1_0',                              # 4
            'C_6', 'C_6',                              # 5, 6
            'C_6',                                     # 7 (has [*])
            '(_0', '[*]_0', ')_0',
            'C_6', 'C_6', '1_0',                       # 8, 9 (close)
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4,
            11: 5, 12: 6, 13: 7, 17: 8, 18: 9,
        },
        "star_attach_aromatic": False,  # sp3 piperidine C4
    },
    # KRAS_G12C: sotorasib-class chassis — acrylamide + piperazine, [*] at the
    # distal piperazine N (sotorasib's aryl-amide arm attaches there).
    # Scaffold (without [*]): O=C(C=C)N1CCNCC1 — 10 heavy atoms.
    # Atom indices in scaffold:
    #   0:C(CH2=) 1:C 2:C(carbonyl) 3:O 4:N(pip-1, ring open) 5:C 6:C
    #   7:N(pip-2, has [*]) 8:C 9:C(close pip ring)
    "KRAS_G12C": {
        "smiles": "C=CC(=O)N1CCN([*])CC1",
        "scaffold_smi": None,
        "n_heavy": 10,
        "token_labels": [
            'start_0',
            'C_0', '=_0', 'C_0',
            'C_0', '(_0', '=_0', 'O_0', ')_0',
            'N_6', '1_0',                              # 4 (pip-1, opens ring)
            'C_6', 'C_6',                              # 5, 6
            'N_6',                                     # 7 (pip-2, has [*])
            '(_0', '[*]_0', ')_0',
            'C_6', 'C_6', '1_0',                       # 8, 9 (close)
            'sep_0',
        ],
        "atom_index_map": {
            1: 0, 3: 1, 4: 2, 7: 3, 9: 4,
            11: 5, 12: 6, 13: 7, 17: 8, 18: 9,
        },
        "star_attach_aromatic": False,  # sp3 piperazine N substituent attaches via sp3 N
    },
}


def _scaffold_smi_no_star(anchor_smi: str) -> str:
    """Strip [*] atom from an anchor SMILES to produce the scaffold."""
    mol = Chem.MolFromSmiles(anchor_smi)
    assert mol is not None, f"failed to parse anchor SMILES {anchor_smi!r}"
    rw = Chem.RWMol(mol)
    star_atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "*"]
    assert len(star_atoms) == 1, f"expected exactly 1 [*] atom in {anchor_smi!r}"
    rw.RemoveAtom(star_atoms[0])
    out = Chem.MolToSmiles(rw.GetMol())
    return out


def _sanity_check_anchor(anchor_id: str, spec: dict) -> dict:
    """Decode the prefix tokens and verify they round-trip to anchor SMILES."""
    labels = spec["token_labels"]
    toks = [_V[t] for t in labels]
    batch = np.array([toks + [0] * (100 - len(toks))])
    pos = np.zeros((1, 100, 3), dtype=np.float32)
    smi, _ss, _mols = _FU.decode3d(batch, pos)
    got = smi[0] if smi else None
    canon_got = Chem.MolToSmiles(Chem.MolFromSmiles(got)) if got else None
    canon_exp = Chem.MolToSmiles(Chem.MolFromSmiles(spec["smiles"]))
    ok = canon_got == canon_exp
    print(f"[L2 ext anchor {anchor_id}] decode sanity: {got}")
    print(f"[L2 ext anchor {anchor_id}] canon_got:     {canon_got}")
    print(f"[L2 ext anchor {anchor_id}] canon_exp:     {canon_exp}")
    print(f"[L2 ext anchor {anchor_id}] tokens={len(toks)} MATCH={ok}")
    if not ok:
        return {"ok": False, "tokens": toks, "labels": labels,
                "canon_got": canon_got, "canon_exp": canon_exp}
    # verify [*] attachment is on aromatic or non-aromatic as declared
    decoded = Chem.MolFromSmiles(got)
    star = next((a for a in decoded.GetAtoms() if a.GetSymbol() == "*"), None)
    assert star is not None, "no [*] in decoded prefix"
    nbrs = list(star.GetNeighbors())
    assert len(nbrs) == 1, f"expected 1 neighbor for [*], got {len(nbrs)}"
    attach = nbrs[0]
    is_arom_attach = attach.GetIsAromatic()
    print(f"[L2 ext anchor {anchor_id}] [*] attaches to: {attach.GetSymbol()} "
          f"aromatic={is_arom_attach}")
    if spec["star_attach_aromatic"] != is_arom_attach:
        return {"ok": False, "reason": "aromaticity_mismatch",
                "expected_aromatic": spec["star_attach_aromatic"],
                "actual_aromatic": is_arom_attach,
                "tokens": toks, "labels": labels}
    # verify atom_index_map covers all heavy atoms in scaffold_smi
    scaffold_smi = spec.get("scaffold_smi") or _scaffold_smi_no_star(spec["smiles"])
    scaff = Chem.MolFromSmiles(scaffold_smi)
    n_scaff = scaff.GetNumAtoms()
    assert n_scaff == spec["n_heavy"], (
        f"scaffold heavy-atom mismatch: declared {spec['n_heavy']}, got {n_scaff}"
    )
    mapped_atom_indices = sorted(spec["atom_index_map"].values())
    expected_indices = list(range(spec["n_heavy"]))
    assert mapped_atom_indices == expected_indices, (
        f"atom_index_map values {mapped_atom_indices} != expected {expected_indices}"
    )
    return {"ok": True, "tokens": toks, "labels": labels,
            "n_tokens": len(toks), "scaffold_smi": scaffold_smi,
            "canon_got": canon_got}


_SCAFFOLD_CACHE_DIR = ROOT / "data" / "lingo3dmol_scaffold_cache"


def _scaffold_cache_path(anchor_id, scaffold_smi):
    """Deterministic cache filename based on anchor id + canonical scaffold."""
    import hashlib
    canon = Chem.MolToSmiles(Chem.MolFromSmiles(scaffold_smi))
    h = hashlib.sha1(canon.encode()).hexdigest()[:10]
    return _SCAFFOLD_CACHE_DIR / f"{anchor_id}_{h}.npy"


def _build_scaffold_xyz(anchor_json_path, anchor_spec, sanity, anchor_id=None):
    """ETKDG-embed the scaffold (without [*]) then translate+rotate so atom 0
    sits at cb_pos_target and the angle SG-atom0-atom1 equals the
    Burgi-Dunitz angle (107 deg) at the electrophilic Cbeta carbon.

    This replaces the previous (buggy) recipe that aligned atom0->atom1
    COLINEAR with the attack vector — that produced angle(SG, atom0, atom1)
    = 180 deg, which is geometrically wrong for a Michael addition.

    **CROSS-MACHINE DETERMINISM FIX (2026-06-01, H1 bug)**:
    ETKDGv3 + MMFFOptimizeMolecule produce slightly different conformers
    across RDKit versions (e.g. 2025.09 on Mac vs 2026.03 on T4). For
    extended scaffolds like H1, a 0.5 A drift in a distal atom (atom 17,
    pyridine C4) lands the atom in an occupied maskcoc voxel on T4 while
    Mac's coord remains in a free voxel. ALL decoded mols then fail
    bonusVSpenalty2 since they share the prefix-atom 17 position. To make
    H1/H2/H3 reproducible across machines we cache the receptor-frame
    rotated xyz in data/lingo3dmol_scaffold_cache/<anchor_id>_<hash>.npy
    so the FIRST machine to compute it (Mac) seeds the deterministic
    coords for downstream runs (T4).

    Returns (n_heavy, 3) np.float64 array in receptor (A) frame.
    """
    with open(anchor_json_path) as f:
        anchor = json.load(f)
    cb_target = np.array(anchor["cb_pos_target"], dtype=np.float64)
    av = np.array(anchor["anchor_attack_vector"], dtype=np.float64)
    sg_pos = np.array(anchor["sg_pos"], dtype=np.float64)
    perp_in_plane = np.array(anchor.get("frame_e2_in_plane",
                                        [0.0, 0.0, 0.0]), dtype=np.float64)
    if np.linalg.norm(perp_in_plane) < 1e-6:
        perp_in_plane = None
    bd_deg = float(anchor.get("burgi_dunitz_angle_deg", 107.0))

    scaffold_smi = sanity["scaffold_smi"]
    cache_p = None
    if anchor_id is not None:
        cache_p = _scaffold_cache_path(anchor_id, scaffold_smi)
        if cache_p.exists():
            try:
                rotated = np.load(cache_p)
                if (rotated.shape[0] == anchor_spec["n_heavy"]
                        and rotated.shape[1] == 3
                        and np.allclose(rotated[0], cb_target, atol=1e-3)):
                    measured = assert_bd_angle(rotated, sg_pos, bd_deg=bd_deg, tol_deg=10.0)
                    print(f"[L2 ext anchor] LOADED cached scaffold xyz from {cache_p.name} "
                          f"(BD angle={measured:.2f} deg)")
                    return rotated
                else:
                    print(f"[L2 ext anchor] cached scaffold xyz at {cache_p} is stale; rebuilding")
            except Exception as e:
                print(f"[L2 ext anchor] failed to load cached xyz ({e}); rebuilding")

    # Embed the scaffold WITHOUT the [*] dummy.
    mol = Chem.MolFromSmiles(scaffold_smi)
    assert mol is not None, f"failed to parse scaffold {scaffold_smi!r}"
    assert mol.GetNumAtoms() == anchor_spec["n_heavy"], (
        f"scaffold atom count {mol.GetNumAtoms()} != n_heavy {anchor_spec['n_heavy']}"
    )
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        params.randomSeed = -1
        res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        # fall back to 2D layout, then add fake z=0
        AllChem.Compute2DCoords(mol)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception as e:
        print(f"[L2 ext anchor] MMFF relax failed (continuing): {e}")
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )

    rotated = rotate_scaffold_to_bd(
        coords, cb_target=cb_target, av=av,
        perp_in_plane=perp_in_plane, bd_deg=bd_deg,
    )

    # Sanity assertion: angle(SG, atom0, atom1) must be ~bd_deg.
    measured = assert_bd_angle(rotated, sg_pos, bd_deg=bd_deg, tol_deg=5.0)
    print(f"[L2 ext anchor] BD angle at atom0 = {measured:.2f} deg "
          f"(target {bd_deg:.1f} deg)")

    if cache_p is not None:
        try:
            _SCAFFOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_p, rotated)
            print(f"[L2 ext anchor] cached scaffold xyz -> {cache_p.name}")
        except Exception as e:
            print(f"[L2 ext anchor] failed to cache scaffold xyz: {e}")
    return rotated


def _build_initial_partial(anchor_json_path, pocket_center, anchor_spec, sanity, anchor_id=None):
    """Build partial_product = [[token_list], [coords_per_token]]."""
    xyz = _build_scaffold_xyz(anchor_json_path, anchor_spec, sanity, anchor_id=anchor_id)
    grid = (xyz - pocket_center) / 0.1 + 119.5
    grid = grid.astype(np.float32)
    # Clip grid coords into model's [0, 239] voxel range to prevent
    # `get_partial_to_warehouse fail: index out of range` for extended chassis.
    grid = np.clip(grid, 0.0, 239.0)

    coords_per_token = []
    last_atom_grid = grid[0].tolist()
    tokens = sanity["tokens"]
    atom_map = anchor_spec["atom_index_map"]
    for tok_idx in range(len(tokens)):
        if tok_idx in atom_map:
            last_atom_grid = grid[atom_map[tok_idx]].tolist()
        coords_per_token.append(list(last_atom_grid))

    return [
        [list(tokens)],
        [list(coords_per_token)],
    ], xyz, grid


def _apply_anchor_mode(initial_partial, anchor_spec, anchor_mode):
    """Apply an anchor relaxation strategy on top of the FULL initial_partial.

    Anchor modes (see "Lingo3DMol anchoring ablation"):
      - FULL            : unchanged (pin 50 prefix tokens + coords) — baseline
      - BC_ONLY         : pin only [start_0, C_0] (=> atom 0 = beta-C); both
                          tokens get the beta-C voxel coord. start_this_step=2.
      - BC_PLUS_4       : pin acrylamide core (atoms 0..4 = C=C-C(=O)-N);
                          token prefix runs to and including 'N_5'. coords =
                          per-atom voxel for atom tokens, last-atom coord for
                          structural tokens (mirrors FULL).
      - SMI_ONLY        : keep full 50-token code prefix; set all per-token
                          coords to the [-1,-1,-1] sentinel so the model picks
                          coords for every atom. This is closest to LibInvent.
      - BC_NOCONSTRAINT : like BC_ONLY for token codes, but coord rows are
                          [-1,-1,-1] -> model also picks the beta-C voxel.
                          Effectively the lightest anchoring we can run.

    Returns: (new_initial_partial, info_dict)
    """
    tokens = list(initial_partial[0][0])
    coords = list(initial_partial[1][0])
    n_full = len(tokens)
    atom_map = anchor_spec["atom_index_map"]

    SENT = [-1.0, -1.0, -1.0]

    if anchor_mode == "FULL":
        new_tokens, new_coords = tokens, coords
    elif anchor_mode == "BC_ONLY":
        # [start_0, C_0] pinned with C_0 voxel; rest dropped.
        new_tokens = tokens[:2]
        new_coords = coords[:2]
    elif anchor_mode == "BC_PLUS_4":
        # Atom indices for the acrylamide core: 0,1,2,3,4 (C=C-C(=O)-N).
        # Take all token positions up to and including the token mapped to atom 4.
        last_tok = max(t for t, a in atom_map.items() if a <= 4)
        new_tokens = tokens[: last_tok + 1]
        new_coords = coords[: last_tok + 1]
    elif anchor_mode == "BC_PLUS_8":
        # Atom indices 0..8: C=C-C(=O)-N + first 4 ring/junction atoms of the
        # isoindoline scaffold (covers acrylamide + amide N + CH2 + 3 aromatic
        # ring atoms). Pins ~9 heavy atoms while leaving the distal
        # oxadiazole-piperidine recognition arm free for the model to decode.
        # For H2 anchor: token positions 0..15 (atom 8 = c at token 15).
        last_tok = max(t for t, a in atom_map.items() if a <= 8)
        new_tokens = tokens[: last_tok + 1]
        new_coords = coords[: last_tok + 1]
    elif anchor_mode == "SMI_ONLY":
        # Tokens stay; all per-token coords -> sentinel -1.
        new_tokens = tokens
        new_coords = [list(SENT) for _ in tokens]
    elif anchor_mode == "BC_NOCONSTRAINT":
        # 2-token code pin (start_0, C_0); coords -> sentinel.
        new_tokens = tokens[:2]
        new_coords = [list(SENT), list(SENT)]
    else:
        raise ValueError(f"unknown anchor_mode {anchor_mode!r}")

    info = {
        "anchor_mode": anchor_mode,
        "n_tokens_pinned": len(new_tokens),
        "n_tokens_full": n_full,
        "coords_use_sentinel": anchor_mode in ("SMI_ONLY", "BC_NOCONSTRAINT"),
    }
    new_partial = [[list(new_tokens)], [list(new_coords)]]
    return new_partial, info


def run(args):
    args.pocket_pdb  = _abs(args.pocket_pdb)
    args.output      = _abs(args.output)
    args.anchor_json = _abs(args.anchor_json)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    anchor_id = args.anchor_id
    assert anchor_id in ANCHORS, f"unknown anchor_id {anchor_id!r}; valid: {list(ANCHORS)}"
    anchor_spec = ANCHORS[anchor_id]
    print(f"[L2 ext anchor {anchor_id}] anchor SMILES: {anchor_spec['smiles']}")
    print(f"[L2 ext anchor {anchor_id}] pocket={args.pocket_pdb}")
    print(f"[L2 ext anchor {anchor_id}] anchor_json={args.anchor_json}")
    print(f"[L2 ext anchor {anchor_id}] gennums={args.gennums} min_acceptable={args.min_acceptable} T={args.tempture}")

    sanity = _sanity_check_anchor(anchor_id, anchor_spec)
    if not sanity["ok"]:
        print(f"[L2 ext anchor {anchor_id}] SANITY FAILED — aborting this anchor")
        summary = {
            "anchor_id": anchor_id,
            "anchor_smiles": anchor_spec["smiles"],
            "status": "sanity_failed",
            "sanity": sanity,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        # write empty summary so the orchestrator sees we tried
        empty_sdf = Path(args.output)
        empty_sdf.parent.mkdir(parents=True, exist_ok=True)
        empty_sdf.write_text("")
        summary_path = out_dir / (Path(args.output).stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        return summary
    print(f"[L2 ext anchor {anchor_id}] sanity PASS ({sanity['n_tokens']} tokens, "
          f"scaffold={sanity['scaffold_smi']})")

    caption_contact = TransformerModel_contact()
    dict_ = torch.load(args.contact_path, map_location="cpu", weights_only=False)
    caption_contact.load_state_dict(dict_, strict=False)
    caption_contact = nn.DataParallel(caption_contact)
    caption_contact.eval()

    caption = TransformerModel()
    # BUG FIX 2026-05-31 (QA round-2 A1): unwrap dict-wrapped FT checkpoints.
    # Train script saves {"model": state_dict, "args": ...}; sampling needs the inner state_dict.
    _raw_ckpt = torch.load(args.caption_path, map_location="cpu", weights_only=False)
    _sd = _raw_ckpt["model"] if isinstance(_raw_ckpt, dict) and "model" in _raw_ckpt else _raw_ckpt
    _info = caption.load_state_dict(_sd, strict=False)
    print(f"[load caption] from {args.caption_path}: missing={len(_info.missing_keys)} unexpected={len(_info.unexpected_keys)}")
    assert len(_info.missing_keys) < 10, f"load_state_dict missing too many params: {_info.missing_keys[:5]}"
    caption = nn.DataParallel(caption)
    caption.eval()

    line = f",,{args.pocket_pdb}"
    testset = testdataset([line])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False,
                            pin_memory=False, num_workers=0)

    for batch in testloader:
        coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
        coc = CollisionCheck(args.pocket_pdb, args.coc_dis, center=center)
        pocket_center = center[0].cpu().numpy()
        break

    initial_partial, scaffold_xyz, scaffold_grid = _build_initial_partial(
        args.anchor_json, pocket_center, anchor_spec, sanity, anchor_id=anchor_id
    )
    anchor_mode = getattr(args, "anchor_mode", "FULL") or "FULL"
    initial_partial, anchor_mode_info = _apply_anchor_mode(
        initial_partial, anchor_spec, anchor_mode
    )
    print(f"[L2 ext anchor {anchor_id}] anchor_mode={anchor_mode}  "
          f"pinned_tokens={anchor_mode_info['n_tokens_pinned']}/"
          f"{anchor_mode_info['n_tokens_full']}  "
          f"coords_sentinel={anchor_mode_info['coords_use_sentinel']}")
    print(f"[L2 ext anchor {anchor_id}] pocket_center={pocket_center.tolist()}")
    print(f"[L2 ext anchor {anchor_id}] scaffold xyz (A):")
    for i, p in enumerate(scaffold_xyz):
        print(f"    atom {i:>2}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
    print(f"[L2 ext anchor {anchor_id}] scaffold grid (voxel) range: "
          f"min={scaffold_grid.min():.2f} max={scaffold_grid.max():.2f}")
    if (scaffold_grid.min() < 0) or (scaffold_grid.max() >= 240):
        print(f"[L2 ext anchor {anchor_id}] WARNING: grid coords outside [0,240) — "
              f"generation may fail")
    print(f"[L2 ext anchor {anchor_id}] seeded {len(initial_partial[0][0])} tokens + "
          f"{len(initial_partial[1][0])} per-token coords; "
          f"atom coords = {len(anchor_spec['atom_index_map'])}")

    results = []
    start = time.time()
    sample_num = args.gen_frag_set
    warehouse = [[], []]

    writer, summary_path = _open_incremental_writer(args.output)
    summary = {
        "n_sampled": 0,
        "elapsed_sec": 0,
        "output_sdf": args.output,
        "pocket_pdb": args.pocket_pdb,
        "status": "running",
        "gennums_target": args.gennums,
        "min_acceptable": args.min_acceptable,
        "plan": f"L2_extended_anchor_{anchor_id}",
        "anchor_id": anchor_id,
        "anchor_smiles": anchor_spec["smiles"],
        "scaffold_smi": sanity["scaffold_smi"],
        "scaffold_prefix_tokens": sanity["tokens"],
        "scaffold_prefix_labels": sanity["labels"],
        "scaffold_xyz_receptor": scaffold_xyz.tolist(),
        "scaffold_grid_voxel": scaffold_grid.tolist(),
        "pocket_center_receptor": pocket_center.tolist(),
        "tempture": args.tempture,
        "star_attach_aromatic": anchor_spec["star_attach_aromatic"],
        "anchor_mode": anchor_mode,
        "anchor_mode_info": anchor_mode_info,
    }
    _flush_summary(summary_path, summary)

    try:
        while len(results) < args.gennums and (time.time() - start) < args.max_run_seconds:
            for batch in testloader:
                coords, residue, atom_type, mask, center, index, contact_prob, contact_scaffold_prob = batch
                with torch.no_grad():
                    index = index.repeat(sample_num)
                    center = center.repeat(sample_num, 1)
                    if contact_prob.shape[-1] == 0 or contact_scaffold_prob.shape[-1] == 0:
                        model_cp, model_csp = caption_contact(
                            coords=coords, residue=residue, atom_type=atom_type,
                            src_mask=mask, isTrain=args.isTrain)
                    if contact_prob.shape[-1] == 0:
                        contact_prob = model_cp
                    if contact_scaffold_prob.shape[-1] == 0:
                        contact_scaffold_prob = model_csp

                    contact_prob0 = torch.where(contact_prob > args.nci_thrs, 2, 0)
                    contact_scaffold_prob1 = torch.where(contact_scaffold_prob > 0.9, 1, 0)
                    contact_prob1 = contact_prob0 + contact_scaffold_prob1
                    contact_prob = contact_prob.repeat(sample_num, 1)
                    residue_use = residue.repeat(sample_num, 1)
                    residue_mask = torch.where(residue_use != 5, 1.0, 0.0)
                    src_mask_repeat = mask.squeeze(1).repeat(sample_num, 1)
                    contact_prob = contact_prob.masked_fill(src_mask_repeat == 0, 0)
                    contact_prob = contact_prob.masked_fill(residue_mask == 0, 0)
                    contact_prob = torch.softmax(contact_prob * 5, dim=-1)
                    contact_idx = topkp_random(contact_prob, top_k=args.topk, top_p=0.9, thred=0.0)
                    factory_args = [coords, residue, mask, atom_type, center, caption,
                                    contact_idx, contact_prob1, coc, contact_scaffold_prob1]
                    seeded_partial = [
                        [list(initial_partial[0][0])],
                        [list(initial_partial[1][0])],
                    ]
                    molecular_workflow(0, warehouse, seeded_partial, factory_args, results, args,
                                       writer=writer, summary_path=summary_path,
                                       summary=summary, start_time=start)
                    if len(results) >= args.gennums:
                        break
            if len(results) >= args.gennums:
                break
            elapsed_so_far = time.time() - start
            if (len(results) >= args.min_acceptable
                    and elapsed_so_far >= 0.6 * args.max_run_seconds):
                print(f"[L2 ext anchor {anchor_id}] min_acceptable={args.min_acceptable} met "
                      f"with n={len(results)}, exiting early")
                break
    finally:
        try:
            writer.close()
        except Exception:
            pass

    elapsed = time.time() - start
    status = "ok" if len(results) >= args.min_acceptable else "insufficient"
    print(f"[L2 ext anchor {anchor_id}] generated {len(results)} mols in {elapsed:.1f}s "
          f"status={status}")
    print(f"[L2 ext anchor {anchor_id}] wrote SDF: {args.output}")

    summary.update({"n_sampled": len(results), "elapsed_sec": elapsed, "status": status})
    _flush_summary(summary_path, summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pocket_pdb", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--anchor_id", required=True, choices=list(ANCHORS.keys()),
                   help="Which anchor cohort to sample (L / H1 / H2 / H3)")
    p.add_argument("--anchor_json", default="data/lingo3dmol_anchor_zap70_cys346.json")
    p.add_argument("--contact_path", default="checkpoint/contact.pkl")
    p.add_argument("--caption_path", default="checkpoint/gen_mol.pkl")
    p.add_argument("--gennums", type=int, default=30)
    p.add_argument("--min_acceptable", type=int, default=15)
    p.add_argument("--gen_frag_set", type=int, default=10)
    p.add_argument("--prod_time", type=int, default=3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--nci_thrs", type=float, default=0.7)
    p.add_argument("--coc_dis", type=float, default=0.5)
    p.add_argument("--frag_len_add", type=int, default=15)
    p.add_argument("--tempture", type=float, default=1.0)
    p.add_argument("--max_run_seconds", type=int, default=1500)
    p.add_argument("--isTrain", action="store_true")
    p.add_argument("--USE_THRESHOLD", action="store_true", default=True)
    p.add_argument("--isMultiSample", action="store_true", default=True)
    p.add_argument("--isGuideSample", action="store_true", default=True)
    p.add_argument("--OnceMolGen", action="store_true")
    p.add_argument("--anchor_mode", default="FULL",
                   choices=["FULL", "BC_ONLY", "BC_PLUS_4", "BC_PLUS_8",
                            "SMI_ONLY", "BC_NOCONSTRAINT"],
                   help="Anchoring strategy (Lingo3DMol anchoring ablation). "
                        "FULL=baseline 50 prefix tokens + coords; BC_ONLY="
                        "pin only [start,C_beta]; BC_PLUS_4=pin acrylamide "
                        "core C=C-C(=O)-N; SMI_ONLY=token codes pinned, "
                        "coords float; BC_NOCONSTRAINT=2-token code pin, "
                        "coords float.")
    args = p.parse_args()
    run(args)
