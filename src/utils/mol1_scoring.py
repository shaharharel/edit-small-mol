"""
Common scoring utilities for the Mol 1 expansion tiers.

Provides:
  - SAScore (Ertl & Schuffenhauer, via RDKit Contrib)
  - Warhead-intact SMARTS check (acrylamide preserved)
  - PAINS filter
  - 3D shape Tanimoto (RDKit O3A alignment + ShapeTanimotoDist)
  - Warhead vector deviation (angle of C=C–C(=O)–N vs seed)
  - Tc to Mol 1
  - max_Tc_train + mean_top10_Tc_train (training set novelty)

All scoring functions are vectorisable; main entry point is `score_dataframe`.
"""

import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, rdShapeHelpers, rdMolAlign
RDLogger.DisableLog('rdApp.*')

# SAScore (Ertl)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
try:
    from rdkit.Chem import RDConfig
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer  # type: ignore
    HAVE_SAS = True
except Exception:
    HAVE_SAS = False


MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]-C(=O)-[N;!H2]"  # acrylamide bound to a non-primary N
WARHEAD_VEC_PATTERN = "[CH2]=[CH]-C(=O)-[N]"  # for vector extraction


def get_warhead_pattern():
    return Chem.MolFromSmarts(WARHEAD_SMARTS)


def warhead_intact(mol_or_smi) -> bool:
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return False
    return mol.HasSubstructMatch(get_warhead_pattern())


def sa_score(mol_or_smi) -> float:
    if not HAVE_SAS:
        return float('nan')
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return float('nan')
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return float('nan')


_PAINS_FC = None
def _get_pains():
    global _PAINS_FC
    if _PAINS_FC is None:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_FC = FilterCatalog.FilterCatalog(params)
    return _PAINS_FC


def pains_alerts(mol_or_smi) -> int:
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return -1
    fc = _get_pains()
    return len(fc.GetMatches(mol))


def descriptors(mol_or_smi):
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "QED": Chem.QED.qed(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "Rings": Descriptors.RingCount(mol),
    }


def morgan_fp(mol_or_smi, radius=2, n_bits=2048):
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def tanimoto(a_fp, b_fp) -> float:
    return DataStructs.TanimotoSimilarity(a_fp, b_fp)


# ── 3D scoring ────────────────────────────────────────────────────────────────

_SEED_MOL3D = None
def get_seed_3d():
    global _SEED_MOL3D
    if _SEED_MOL3D is None:
        m = Chem.AddHs(Chem.MolFromSmiles(MOL1_SMILES))
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xCAFE
        AllChem.EmbedMolecule(m, params)
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        _SEED_MOL3D = m
    return _SEED_MOL3D


def embed_3d(mol_or_smi):
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xCAFE
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


_SEED_CONFS = None
N_CONFS_DEFAULT = 4

def get_seed_conformer_ensemble(n_confs=N_CONFS_DEFAULT):
    """Generate + minimise N low-energy conformers of Mol 1 once per process."""
    global _SEED_CONFS
    if _SEED_CONFS is not None and _SEED_CONFS.GetNumConformers() >= n_confs:
        return _SEED_CONFS
    m = Chem.AddHs(Chem.MolFromSmiles(MOL1_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xCAFE
    cids = AllChem.EmbedMultipleConfs(m, numConfs=n_confs, params=params)
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(m, confId=cid, maxIters=200)
        except Exception:
            pass
    _SEED_CONFS = m
    return _SEED_CONFS


def embed_3d_ensemble(mol_or_smi, n_confs=N_CONFS_DEFAULT):
    if isinstance(mol_or_smi, str):
        mol = Chem.MolFromSmiles(mol_or_smi)
    else:
        mol = mol_or_smi
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xCAFE
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if not cids:
        return None
    for cid in cids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid, maxIters=200)
        except Exception:
            pass
    return mol


def shape_tanimoto_seed(mol_or_smi) -> float:
    """3D shape Tanimoto vs Mol 1 seed via O3A alignment + Gaussian volume Tc.
    Uses conformer ensemble (best alignment over confs)."""
    seed_ens = get_seed_conformer_ensemble()
    cand_ens = embed_3d_ensemble(mol_or_smi)
    if cand_ens is None:
        return float('nan')
    try:
        seed_props = AllChem.MMFFGetMoleculeProperties(seed_ens)
        cand_props = AllChem.MMFFGetMoleculeProperties(cand_ens)
        if seed_props is None or cand_props is None:
            return float('nan')
        best = float('nan')
        for sc in range(seed_ens.GetNumConformers()):
            for cc in range(cand_ens.GetNumConformers()):
                try:
                    o3a = rdMolAlign.GetO3A(cand_ens, seed_ens, cand_props, seed_props,
                                            refCid=sc, prbCid=cc)
                    o3a.Align()
                    tc = 1.0 - float(rdShapeHelpers.ShapeTanimotoDist(seed_ens, cand_ens,
                                                                      confId1=sc, confId2=cc))
                    if not np.isnan(tc) and (np.isnan(best) or tc > best):
                        best = tc
                except Exception:
                    continue
        return best
    except Exception:
        return float('nan')


def esp_sim_seed(mol_or_smi) -> float:
    """ESP-Sim Tanimoto (combined shape + electrostatic) vs Mol 1 seed.
    Uses ESP-Sim's default Gasteiger charges + Gaussian-volume."""
    try:
        from espsim import GetEspSim, GetShapeSim
    except ImportError:
        return float('nan')
    seed_ens = get_seed_conformer_ensemble()
    cand_ens = embed_3d_ensemble(mol_or_smi)
    if cand_ens is None:
        return float('nan')
    try:
        # Use a single best-aligned conformer pair
        seed_props = AllChem.MMFFGetMoleculeProperties(seed_ens)
        cand_props = AllChem.MMFFGetMoleculeProperties(cand_ens)
        if seed_props is None or cand_props is None:
            return float('nan')
        o3a = rdMolAlign.GetO3A(cand_ens, seed_ens, cand_props, seed_props)
        o3a.Align()
        return float(GetEspSim(cand_ens, seed_ens))
    except Exception:
        return float('nan')


def warhead_vector_deviation(mol_or_smi) -> float:
    """MIN angle (degrees) between seed's C=C–C(=O)–N vector and candidate's,
    over a conformer ensemble after O3A alignment (kills rotamer-choice noise).
    Returns nan if warhead missing or alignment fails.
    """
    seed_ens = get_seed_conformer_ensemble()
    cand_ens = embed_3d_ensemble(mol_or_smi)
    if cand_ens is None:
        return float('nan')

    pattern = Chem.MolFromSmarts(WARHEAD_VEC_PATTERN)
    seed_match = seed_ens.GetSubstructMatch(pattern)
    cand_match = cand_ens.GetSubstructMatch(pattern)
    if not seed_match or not cand_match:
        return float('nan')

    try:
        seed_props = AllChem.MMFFGetMoleculeProperties(seed_ens)
        cand_props = AllChem.MMFFGetMoleculeProperties(cand_ens)
        if seed_props is None or cand_props is None:
            return float('nan')
    except Exception:
        return float('nan')

    best_angle = float('inf')
    for sc in range(seed_ens.GetNumConformers()):
        for cc in range(cand_ens.GetNumConformers()):
            try:
                o3a = rdMolAlign.GetO3A(cand_ens, seed_ens, cand_props, seed_props,
                                        refCid=sc, prbCid=cc)
                o3a.Align()
                seed_conf = seed_ens.GetConformer(sc)
                cand_conf = cand_ens.GetConformer(cc)
                s_beta = np.array(seed_conf.GetAtomPosition(seed_match[0]))
                s_n = np.array(seed_conf.GetAtomPosition(seed_match[3]))
                c_beta = np.array(cand_conf.GetAtomPosition(cand_match[0]))
                c_n = np.array(cand_conf.GetAtomPosition(cand_match[3]))
                v_seed = s_n - s_beta
                v_cand = c_n - c_beta
                cos = np.dot(v_seed, v_cand) / (np.linalg.norm(v_seed) * np.linalg.norm(v_cand) + 1e-9)
                cos = max(-1.0, min(1.0, cos))
                ang = float(np.degrees(np.arccos(cos)))
                if ang < best_angle:
                    best_angle = ang
            except Exception:
                continue
    return best_angle if best_angle != float('inf') else float('nan')


# ── Batch entry point ─────────────────────────────────────────────────────────

def score_dataframe(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    train_smiles: Optional[list] = None,
    compute_3d: bool = True,
    pIC50_predictor=None,
) -> pd.DataFrame:
    """Augment a DataFrame of candidates with all scoring columns.

    Args:
        df: must contain `smiles_col`.
        train_smiles: optional list of training SMILES; enables max_Tc_train + mean_top10.
        compute_3d: enable shape Tc + warhead vector (slower, ~50ms/mol).
        pIC50_predictor: optional callable(list[str]) -> list[float] for pIC50.
    """
    out = df.copy()
    smis = out[smiles_col].tolist()
    mols = [Chem.MolFromSmiles(s) for s in smis]

    # 2D / fast
    out["valid"] = [m is not None for m in mols]
    out["warhead_intact"] = [warhead_intact(m) if m else False for m in mols]
    out["SAScore"] = [sa_score(m) if m else float('nan') for m in mols]
    out["PAINS_alerts"] = [pains_alerts(m) if m else -1 for m in mols]
    descs = [descriptors(m) if m else None for m in mols]
    for k in ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "QED", "HeavyAtoms", "Rings"]:
        out[k] = [d[k] if d else float('nan') for d in descs]

    # Tc to Mol 1
    seed_fp = morgan_fp(MOL1_SMILES)
    cand_fps = [morgan_fp(m) if m else None for m in mols]
    out["Tc_to_Mol1"] = [tanimoto(seed_fp, fp) if fp else float('nan') for fp in cand_fps]

    # Tc to training set
    if train_smiles is not None:
        train_fps = [morgan_fp(s) for s in train_smiles]
        train_fps = [fp for fp in train_fps if fp is not None]
        max_tc, mean_top10 = [], []
        for fp in cand_fps:
            if fp is None:
                max_tc.append(float('nan')); mean_top10.append(float('nan')); continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
            max_tc.append(float(sims.max()))
            top = np.sort(sims)[-10:]
            mean_top10.append(float(top.mean()))
        out["max_Tc_train"] = max_tc
        out["mean_top10_Tc_train"] = mean_top10

    # 3D — with explicit garbage collection between molecules to avoid memory growth
    if compute_3d:
        import gc as _gc
        shape_tcs, esp_sims, wh_devs = [], [], []
        for i, m in enumerate(mols):
            if m is None:
                shape_tcs.append(float('nan')); esp_sims.append(float('nan')); wh_devs.append(float('nan'))
                continue
            shape_tcs.append(shape_tanimoto_seed(m))
            esp_sims.append(esp_sim_seed(m))
            wh_devs.append(warhead_vector_deviation(m))
            if (i + 1) % 5 == 0:
                _gc.collect()
        out["shape_Tc_seed"] = shape_tcs
        out["esp_sim_seed"] = esp_sims
        out["warhead_dev_deg"] = wh_devs

    # FiLMDelta pIC50
    if pIC50_predictor is not None:
        out["pIC50"] = pIC50_predictor(smis)

    return out


def load_film_predictor():
    """Return a callable(smiles_list) → np.ndarray of pIC50 anchor-mean predictions.
    Uses the cached reinvent4_film_model.pt checkpoint (FiLMDelta on 280 ZAP70 anchors).
    """
    import torch
    from sklearn.preprocessing import StandardScaler
    from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

    model_path = Path(__file__).resolve().parents[2] / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = ckpt["scaler_mean"]
    scaler.scale_ = ckpt["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    anchor_embs = ckpt["anchor_embs"]
    anchor_pIC50 = np.asarray(ckpt["anchor_pIC50"]).astype(np.float64)
    n_anchors = anchor_embs.shape[0]

    def predict(smiles_list):
        scores = np.full(len(smiles_list), np.nan, dtype=np.float64)
        valid_idx, fps = [], []
        for i, s in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(s) if isinstance(s, str) else s
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr); valid_idx.append(i)
        if not fps:
            return scores
        embs = torch.FloatTensor(scaler.transform(np.array(fps, dtype=np.float32)))
        with torch.no_grad():
            for k, orig in enumerate(valid_idx):
                tgt = embs[k:k+1].expand(n_anchors, -1)
                deltas = model(anchor_embs, tgt).numpy().flatten()
                scores[orig] = float(np.mean(anchor_pIC50 + deltas))
        return scores

    return predict


def load_zap70_train_smiles():
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    return smiles_df['smiles'].tolist()
