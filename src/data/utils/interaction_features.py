"""Protein-ligand interaction fingerprints from AutoDock Vina PDBQT pose files.

Extracts geometric interaction features (H-bonds, hydrophobic contacts,
aromatic stacking, ionic interactions) from docked poses without requiring
ProLIF, PLIP, or BioPython — uses only numpy/scipy and standard library.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# ── AutoDock4 atom type classification ──────────────────────────────────────

# H-bond donors: types that carry or can donate H
_HBOND_DONOR_TYPES = {"HD", "HS"}
# H-bond acceptors: O/N/S lone-pair bearing types
_HBOND_ACCEPTOR_TYPES = {"OA", "NA", "SA", "OS", "NS"}
# Hydrophobic / carbon types
_HYDROPHOBIC_TYPES = {"C", "A"}  # A = aromatic carbon in AD4
# Aromatic carbon
_AROMATIC_TYPES = {"A"}
# Positively charged (metal-like or guanidinium N)
_POS_CHARGED_TYPES = {"N", "NA"}
# Negatively charged (carboxylate O)
_NEG_CHARGED_TYPES = {"OA"}
# Charged residue names
_POS_CHARGED_RESIDUES = {"ARG", "LYS", "HIS"}
_NEG_CHARGED_RESIDUES = {"ASP", "GLU"}


# ── Feature names ───────────────────────────────────────────────────────────

_FEATURE_NAMES = [
    "n_contact_residues",
    "n_hbond_donor_contacts",
    "n_hbond_acceptor_contacts",
    "n_hbonds_total",
    "n_hydrophobic_contacts",
    "n_aromatic_contacts",
    "n_ionic_contacts",
    "mean_contact_distance",
    "min_contact_distance",
    "max_contact_distance",
    "std_contact_distance",
    "burial_fraction",
    "n_ligand_atoms_buried",
    "vina_score",
    "vina_inter",
    "vina_intra",
    "vina_unbound",
]

INTERACTION_FEAT_DIM = len(_FEATURE_NAMES)


def get_interaction_feature_names() -> List[str]:
    """Return the feature names for the interaction fingerprint vector."""
    return list(_FEATURE_NAMES)


# ── PDBQT parsing ──────────────────────────────────────────────────────────


def extract_vina_energies(pose_pdbqt_path: str) -> Dict:
    """Parse REMARK lines from a Vina pose PDBQT to extract energy terms.

    Args:
        pose_pdbqt_path: Path to a pose PDBQT file produced by Vina.

    Returns:
        Dictionary with keys: vina_score, vina_inter, vina_intra,
        vina_unbound, all_scores (list of per-mode scores).
        Values are float or NaN on parse failure.
    """
    result = {
        "vina_score": np.nan,
        "vina_inter": np.nan,
        "vina_intra": np.nan,
        "vina_unbound": np.nan,
        "all_scores": [],
    }
    try:
        with open(pose_pdbqt_path, "r") as fh:
            current_model = 0
            for line in fh:
                line = line.rstrip()
                if line.startswith("MODEL"):
                    current_model += 1
                elif line.startswith("REMARK VINA RESULT:"):
                    # Format: REMARK VINA RESULT:    -9.301      0.000      0.000
                    parts = line.split(":")[1].split()
                    if parts:
                        score = float(parts[0])
                        result["all_scores"].append(score)
                        if current_model <= 1:
                            # Best pose (MODEL 1)
                            result["vina_score"] = score
                elif line.startswith("REMARK INTER:"):
                    if current_model <= 1:
                        val = line.split(":")[1].strip()
                        result["vina_inter"] = float(val)
                elif line.startswith("REMARK INTRA:"):
                    if current_model <= 1:
                        val = line.split(":")[1].strip()
                        result["vina_intra"] = float(val)
                elif line.startswith("REMARK UNBOUND:"):
                    if current_model <= 1:
                        val = line.split(":")[1].strip()
                        result["vina_unbound"] = float(val)
    except Exception as e:
        logger.warning(f"Failed to parse Vina energies from {pose_pdbqt_path}: {e}")

    return result


def parse_pdbqt_atoms(
    pdbqt_path: str, model: int = 1
) -> List[Dict]:
    """Parse ATOM/HETATM records from a PDBQT file for a given MODEL.

    PDBQT uses fixed-width PDB columns with an extra AD4 atom type field
    at columns 77-79 (0-indexed: 77:79).

    For receptor files without MODEL records, all atoms are returned
    when model=1.

    Args:
        pdbqt_path: Path to a PDBQT file.
        model: Which MODEL to extract (1-based). Default 1 (best pose).

    Returns:
        List of atom dicts with keys: x, y, z, atom_name, atom_type,
        residue_name, residue_num, chain.
    """
    atoms = []
    current_model = 0
    has_model_records = False

    try:
        with open(pdbqt_path, "r") as fh:
            for line in fh:
                if line.startswith("MODEL"):
                    has_model_records = True
                    current_model += 1
                    continue
                if line.startswith("ENDMDL"):
                    if current_model == model:
                        break
                    continue

                # For files without MODEL records, treat everything as model 1
                if not has_model_records:
                    current_model = 1

                if current_model != model:
                    continue

                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        # PDB fixed-width columns (1-indexed in spec, 0-indexed here)
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain = line[21:22].strip() if len(line) > 21 else ""
                        residue_num_str = line[22:26].strip()
                        residue_num = int(residue_num_str) if residue_num_str else 0
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        # AD4 atom type is the last non-whitespace field
                        # In PDBQT it's typically at columns 77+ but can vary
                        # Safest: take the last whitespace-separated token
                        atom_type = line.split()[-1] if len(line) > 60 else "C"

                        atoms.append({
                            "x": x,
                            "y": y,
                            "z": z,
                            "atom_name": atom_name,
                            "atom_type": atom_type,
                            "residue_name": residue_name,
                            "residue_num": residue_num,
                            "chain": chain,
                        })
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping malformed ATOM line: {e}")
                        continue
    except Exception as e:
        logger.warning(f"Failed to parse PDBQT atoms from {pdbqt_path}: {e}")

    return atoms


# ── Interaction fingerprint ─────────────────────────────────────────────────


def _atoms_to_coords(atoms: List[Dict]) -> np.ndarray:
    """Extract Nx3 coordinate array from atom list."""
    if not atoms:
        return np.empty((0, 3), dtype=np.float64)
    return np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=np.float64)


def _atoms_to_types(atoms: List[Dict]) -> List[str]:
    """Extract atom type list."""
    return [a["atom_type"] for a in atoms]


def compute_interaction_fingerprint(
    ligand_pose_path: str,
    receptor_pdbqt_path: str,
    cutoff: float = 6.0,
) -> np.ndarray:
    """Compute a fixed-size protein-ligand interaction fingerprint.

    For the best pose (MODEL 1), identifies receptor residues in contact
    with the ligand and classifies interaction types based on distance
    thresholds and AD4 atom type heuristics.

    Args:
        ligand_pose_path: Path to ligand pose PDBQT (Vina output).
        receptor_pdbqt_path: Path to receptor PDBQT.
        cutoff: Distance cutoff in Angstroms for defining contacts.

    Returns:
        Feature vector of shape (INTERACTION_FEAT_DIM,) as float32.
        Returns NaN-filled vector on failure.
    """
    nan_vec = np.full(INTERACTION_FEAT_DIM, np.nan, dtype=np.float32)

    # Parse atoms
    lig_atoms = parse_pdbqt_atoms(ligand_pose_path, model=1)
    rec_atoms = parse_pdbqt_atoms(receptor_pdbqt_path, model=1)

    if not lig_atoms or not rec_atoms:
        logger.warning(
            f"No atoms parsed: ligand={len(lig_atoms)}, receptor={len(rec_atoms)}"
        )
        return nan_vec

    # Coordinate arrays
    lig_coords = _atoms_to_coords(lig_atoms)
    rec_coords = _atoms_to_coords(rec_atoms)

    # All-pairs distance matrix: (n_lig, n_rec)
    dist_matrix = cdist(lig_coords, rec_coords)

    # Identify contact receptor atoms (any lig atom within cutoff)
    rec_in_contact = np.any(dist_matrix < cutoff, axis=0)  # shape (n_rec,)

    if not np.any(rec_in_contact):
        logger.warning("No receptor atoms within cutoff")
        return nan_vec

    # Get unique contact residues
    contact_residue_ids = set()
    for j in range(len(rec_atoms)):
        if rec_in_contact[j]:
            a = rec_atoms[j]
            contact_residue_ids.add((a["chain"], a["residue_num"], a["residue_name"]))

    n_contact_residues = len(contact_residue_ids)

    # Per-atom-pair interaction classification
    lig_types = _atoms_to_types(lig_atoms)
    rec_types = _atoms_to_types(rec_atoms)

    n_hbond_donor = 0
    n_hbond_acceptor = 0
    n_hydrophobic = 0
    n_aromatic = 0
    n_ionic = 0

    # Minimum distances for each ligand atom to any receptor atom
    min_dist_per_lig = np.min(dist_matrix, axis=1)  # shape (n_lig,)
    # All contact distances (lig-rec pairs within cutoff)
    contact_distances = dist_matrix[dist_matrix < cutoff]

    # Classify interactions using vectorized distance thresholds
    # For efficiency, iterate only over contact receptor atoms
    contact_rec_indices = np.where(rec_in_contact)[0]

    for j in contact_rec_indices:
        r_type = rec_types[j]
        r_atom = rec_atoms[j]
        r_resname = r_atom["residue_name"]

        # Get distances from this receptor atom to all ligand atoms
        dists_to_lig = dist_matrix[:, j]  # shape (n_lig,)

        for i in range(len(lig_atoms)):
            d = dists_to_lig[i]
            if d >= cutoff:
                continue

            l_type = lig_types[i]

            # H-bond donor contact: ligand HD near receptor acceptor, or
            # receptor HD near ligand acceptor
            if d < 3.5:
                if l_type in _HBOND_DONOR_TYPES and r_type in _HBOND_ACCEPTOR_TYPES:
                    n_hbond_donor += 1
                if r_type in _HBOND_DONOR_TYPES and l_type in _HBOND_ACCEPTOR_TYPES:
                    n_hbond_acceptor += 1

            # Hydrophobic contact: C-C within 4.5A
            if d < 4.5:
                if l_type in _HYDROPHOBIC_TYPES and r_type in _HYDROPHOBIC_TYPES:
                    n_hydrophobic += 1

            # Aromatic stacking: aromatic C near aromatic C within 5.5A
            if d < 5.5:
                if l_type in _AROMATIC_TYPES and r_type in _AROMATIC_TYPES:
                    n_aromatic += 1

            # Ionic interactions: charged residue atoms near opposite-charge
            # ligand atoms within 5.0A
            if d < 5.0:
                lig_could_be_pos = l_type in _POS_CHARGED_TYPES
                lig_could_be_neg = l_type in _NEG_CHARGED_TYPES
                rec_is_pos_res = r_resname in _POS_CHARGED_RESIDUES
                rec_is_neg_res = r_resname in _NEG_CHARGED_RESIDUES

                if (lig_could_be_neg and rec_is_pos_res) or (
                    lig_could_be_pos and rec_is_neg_res
                ):
                    n_ionic += 1

    # Burial: fraction of ligand atoms with any protein neighbor < 4.0A
    burial_mask = min_dist_per_lig < 4.0
    burial_fraction = float(np.mean(burial_mask))
    n_buried = int(np.sum(burial_mask))

    # Contact distance statistics
    if len(contact_distances) > 0:
        mean_contact_dist = float(np.mean(contact_distances))
        min_contact_dist = float(np.min(contact_distances))
        max_contact_dist = float(np.max(contact_distances))
        std_contact_dist = float(np.std(contact_distances))
    else:
        mean_contact_dist = np.nan
        min_contact_dist = np.nan
        max_contact_dist = np.nan
        std_contact_dist = np.nan

    # Vina energies
    energies = extract_vina_energies(ligand_pose_path)

    # Assemble feature vector
    features = np.array(
        [
            float(n_contact_residues),
            float(n_hbond_donor),
            float(n_hbond_acceptor),
            float(n_hbond_donor + n_hbond_acceptor),
            float(n_hydrophobic),
            float(n_aromatic),
            float(n_ionic),
            mean_contact_dist,
            min_contact_dist,
            max_contact_dist,
            std_contact_dist,
            burial_fraction,
            float(n_buried),
            energies["vina_score"],
            energies["vina_inter"],
            energies["vina_intra"],
            energies["vina_unbound"],
        ],
        dtype=np.float32,
    )

    return features


# ── Batch processing ────────────────────────────────────────────────────────


def compute_all_interaction_features(
    poses_dir: str,
    receptor_path: str,
    mol_ids: List,
    pose_filename_template: str = "mol_{mol_id}_pose.pdbqt",
    cache_path: Optional[str] = None,
) -> Dict:
    """Compute interaction features for multiple docked molecules.

    Args:
        poses_dir: Directory containing pose PDBQT files.
        receptor_path: Path to receptor PDBQT file.
        mol_ids: List of molecule identifiers (used to construct filenames).
        pose_filename_template: Filename template with {mol_id} placeholder.
        cache_path: Optional .npz path to cache/load results.

    Returns:
        Dictionary mapping mol_id -> feature vector (np.ndarray of shape
        (INTERACTION_FEAT_DIM,)).
    """
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached = dict(data["features"].item())
            # Verify all requested mol_ids are present
            if all(mid in cached for mid in mol_ids):
                logger.info(f"Loaded {len(cached)} interaction features from cache")
                return {mid: cached[mid] for mid in mol_ids}
            else:
                logger.info("Cache incomplete, recomputing...")
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")

    # Parse receptor once (expensive)
    logger.info(f"Parsing receptor from {receptor_path}")
    rec_atoms = parse_pdbqt_atoms(receptor_path, model=1)
    if not rec_atoms:
        logger.error("Failed to parse receptor atoms")
        nan_vec = np.full(INTERACTION_FEAT_DIM, np.nan, dtype=np.float32)
        return {mid: nan_vec.copy() for mid in mol_ids}

    rec_coords = _atoms_to_coords(rec_atoms)
    rec_types = _atoms_to_types(rec_atoms)

    results = {}
    n_success = 0
    n_fail = 0

    for mol_id in mol_ids:
        fname = pose_filename_template.format(mol_id=mol_id)
        pose_path = os.path.join(poses_dir, fname)

        if not os.path.exists(pose_path):
            logger.debug(f"Pose file not found: {pose_path}")
            results[mol_id] = np.full(INTERACTION_FEAT_DIM, np.nan, dtype=np.float32)
            n_fail += 1
            continue

        try:
            feat = compute_interaction_fingerprint(
                pose_path, receptor_path, cutoff=6.0
            )
            results[mol_id] = feat
            if not np.all(np.isnan(feat)):
                n_success += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.warning(f"Failed for mol_id={mol_id}: {e}")
            results[mol_id] = np.full(INTERACTION_FEAT_DIM, np.nan, dtype=np.float32)
            n_fail += 1

    logger.info(
        f"Interaction features: {n_success} success, {n_fail} failed "
        f"out of {len(mol_ids)} molecules"
    )

    # Save cache
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, features=results)
            logger.info(f"Saved interaction feature cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# SE(3)-Invariant Geometric Features from Docked Poses
# ══════════════════════════════════════════════════════════════════════════


def get_pocket_residues(
    receptor_pdbqt_path: str,
    ref_ligand_poses_dir: str,
    n_samples: int = 50,
    contact_cutoff: float = 6.0,
    min_contact_fraction: float = 0.10,
    pose_filename_glob: str = "*_pose.pdbqt",
    cache_path: Optional[str] = None,
) -> List[Tuple[str, int, str]]:
    """Determine the fixed set of binding-pocket residues.

    Pocket residues are those receptor residues contacted (within
    ``contact_cutoff`` A) by at least ``min_contact_fraction`` of a sample of
    docked ligand poses.  This defines the fixed-size per-residue contact
    vector used by :func:`compute_geometric_features`.

    Args:
        receptor_pdbqt_path: Path to receptor PDBQT.
        ref_ligand_poses_dir: Directory with docked ligand pose PDBQTs.
        n_samples: Max number of ligand poses to sample for pocket definition.
        contact_cutoff: Distance cutoff (A) for residue contact.
        min_contact_fraction: A residue must be contacted by at least this
            fraction of sampled ligands to be included.
        pose_filename_glob: Glob pattern for pose files.
        cache_path: Optional path to cache the pocket residue list as JSON.

    Returns:
        Sorted list of (chain, residue_num, residue_name) tuples defining
        the pocket.
    """
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        try:
            import json
            with open(cache_path, "r") as fh:
                data = json.load(fh)
            pocket = [tuple(r) for r in data["pocket_residues"]]
            logger.info(f"Loaded {len(pocket)} pocket residues from cache")
            return pocket
        except Exception as e:
            logger.warning(f"Failed to load pocket cache {cache_path}: {e}")

    rec_atoms = parse_pdbqt_atoms(receptor_pdbqt_path, model=1)
    if not rec_atoms:
        logger.error("Failed to parse receptor for pocket detection")
        return []

    rec_coords = _atoms_to_coords(rec_atoms)

    # Collect pose files
    import glob
    pose_files = sorted(
        glob.glob(os.path.join(ref_ligand_poses_dir, pose_filename_glob))
    )
    if not pose_files:
        logger.warning(f"No pose files found in {ref_ligand_poses_dir}")
        return []

    # Sub-sample if too many
    rng = np.random.RandomState(42)
    if len(pose_files) > n_samples:
        pose_files = list(rng.choice(pose_files, size=n_samples, replace=False))

    # Count contacts per residue across sampled ligands
    residue_contact_counts: Dict[Tuple[str, int, str], int] = {}
    n_valid = 0

    for pf in pose_files:
        lig_atoms = parse_pdbqt_atoms(pf, model=1)
        if not lig_atoms:
            continue
        n_valid += 1

        lig_coords = _atoms_to_coords(lig_atoms)
        # Min distance from each receptor atom to any ligand atom
        dists = cdist(rec_coords, lig_coords)  # (n_rec, n_lig)
        min_dists = np.min(dists, axis=1)  # (n_rec,)

        contacted = set()
        for j in range(len(rec_atoms)):
            if min_dists[j] < contact_cutoff:
                key = (
                    rec_atoms[j]["chain"],
                    rec_atoms[j]["residue_num"],
                    rec_atoms[j]["residue_name"],
                )
                contacted.add(key)

        for key in contacted:
            residue_contact_counts[key] = residue_contact_counts.get(key, 0) + 1

    if n_valid == 0:
        logger.warning("No valid ligand poses found for pocket detection")
        return []

    threshold = max(1, int(n_valid * min_contact_fraction))
    pocket = sorted(
        k for k, v in residue_contact_counts.items() if v >= threshold
    )

    logger.info(
        f"Pocket definition: {len(pocket)} residues (from {n_valid} poses, "
        f"threshold={threshold})"
    )

    # Save cache
    if cache_path:
        try:
            import json
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as fh:
                json.dump({"pocket_residues": pocket, "n_poses": n_valid}, fh)
            logger.info(f"Saved pocket residue cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save pocket cache: {e}")

    return pocket


def _compute_gyration_tensor(coords: np.ndarray) -> np.ndarray:
    """Compute 3x3 gyration tensor from Nx3 coordinates."""
    centroid = np.mean(coords, axis=0)
    shifted = coords - centroid
    return (shifted.T @ shifted) / len(coords)


def compute_geometric_features(
    ligand_pose_path: str,
    receptor_pdbqt_path: str,
    pocket_residues: List[Tuple[str, int, str]],
    contact_cutoff: float = 4.5,
    pocket_cutoff: float = 6.0,
) -> np.ndarray:
    """Compute SE(3)-invariant geometric features from a docked pose.

    All features are computed in the fixed receptor frame, making them
    SE(3)-invariant by construction.

    Feature groups (concatenated):
        1. Per-residue contact counts (len(pocket_residues) dims): number
           of ligand heavy atoms within ``contact_cutoff`` of each pocket
           residue.
        2. Centroid displacement (3 dims): x, y, z of ligand centroid
           relative to pocket centroid (in receptor frame).
        3. Occupancy descriptors (3 dims): ligand span, ligand-pocket
           overlap fraction, pocket fill fraction.
        4. Score spread (1 dim): std of all Vina mode scores.
        5. Principal axis alignment (3 dims): cosine between ligand and
           pocket principal axes, asphericity, compactness (Rg).

    Args:
        ligand_pose_path: Path to ligand pose PDBQT (Vina output).
        receptor_pdbqt_path: Path to receptor PDBQT.
        pocket_residues: Fixed list of (chain, resnum, resname) tuples
            from :func:`get_pocket_residues`.
        contact_cutoff: Distance cutoff (A) for per-residue contacts and
            ligand-pocket overlap.
        pocket_cutoff: Distance cutoff (A) for identifying pocket atoms
            from receptor.

    Returns:
        Feature vector of shape (n_pocket + 10,) as float32.
        Returns NaN-filled vector on failure.
    """
    n_pocket = len(pocket_residues)
    feat_dim = n_pocket + 10  # 3 centroid + 3 occupancy + 1 score + 3 axis
    nan_vec = np.full(feat_dim, np.nan, dtype=np.float32)

    # Parse ligand atoms (best pose, model 1)
    lig_atoms = parse_pdbqt_atoms(ligand_pose_path, model=1)
    if not lig_atoms:
        return nan_vec

    lig_coords = _atoms_to_coords(lig_atoms)
    n_lig = len(lig_atoms)

    # Parse receptor atoms
    rec_atoms = parse_pdbqt_atoms(receptor_pdbqt_path, model=1)
    if not rec_atoms:
        return nan_vec

    rec_coords = _atoms_to_coords(rec_atoms)

    # Build residue-to-atom-indices lookup for receptor
    residue_to_rec_idx: Dict[Tuple[str, int, str], List[int]] = {}
    for j, a in enumerate(rec_atoms):
        key = (a["chain"], a["residue_num"], a["residue_name"])
        if key not in residue_to_rec_idx:
            residue_to_rec_idx[key] = []
        residue_to_rec_idx[key].append(j)

    # ── 1. Per-residue contact counts ──
    per_residue_contacts = np.zeros(n_pocket, dtype=np.float32)
    pocket_atom_indices = []

    for p_i, res_key in enumerate(pocket_residues):
        atom_indices = residue_to_rec_idx.get(res_key, [])
        pocket_atom_indices.extend(atom_indices)
        if not atom_indices:
            continue
        res_coords = rec_coords[atom_indices]
        # Distance from each ligand atom to closest atom in this residue
        dists = cdist(lig_coords, res_coords)  # (n_lig, n_res_atoms)
        min_dists = np.min(dists, axis=1)  # (n_lig,)
        per_residue_contacts[p_i] = float(np.sum(min_dists < contact_cutoff))

    # ── 2. Centroid displacement ──
    lig_centroid = np.mean(lig_coords, axis=0)  # (3,)

    # Pocket centroid: mean of all pocket residue atom coordinates
    if pocket_atom_indices:
        unique_pocket_idx = sorted(set(pocket_atom_indices))
        pocket_coords = rec_coords[unique_pocket_idx]
        pocket_centroid = np.mean(pocket_coords, axis=0)
    else:
        pocket_centroid = np.mean(rec_coords, axis=0)
        pocket_coords = rec_coords

    centroid_disp = lig_centroid - pocket_centroid  # (3,) in receptor frame

    # ── 3. Occupancy descriptors ──
    # 3a. Ligand span: max pairwise distance between ligand atoms
    if n_lig >= 2:
        lig_dists = cdist(lig_coords, lig_coords)
        ligand_span = float(np.max(lig_dists))
    else:
        ligand_span = 0.0

    # 3b. Ligand-pocket overlap: fraction of ligand atoms within cutoff
    #     of any protein atom
    all_dists = cdist(lig_coords, rec_coords)  # (n_lig, n_rec)
    min_dist_to_protein = np.min(all_dists, axis=1)  # (n_lig,)
    overlap_fraction = float(
        np.mean(min_dist_to_protein < contact_cutoff)
    )

    # 3c. Pocket fill fraction: fraction of pocket residues contacted
    pocket_fill = float(np.mean(per_residue_contacts > 0)) if n_pocket > 0 else 0.0

    # ── 4. Score spread ──
    energies = extract_vina_energies(ligand_pose_path)
    all_scores = energies.get("all_scores", [])
    if len(all_scores) >= 2:
        score_spread = float(np.std(all_scores))
    else:
        score_spread = 0.0

    # ── 5. Principal axis alignment ──
    # 5a. Ligand principal axis (eigenvector of largest eigenvalue of gyration tensor)
    if n_lig >= 3:
        lig_gyr = _compute_gyration_tensor(lig_coords)
        lig_eigenvalues, lig_eigenvectors = np.linalg.eigh(lig_gyr)
        # Sorted ascending; principal axis = last eigenvector
        lig_principal = lig_eigenvectors[:, -1]

        # 5b. Pocket principal axis
        if len(pocket_coords) >= 3:
            pocket_gyr = _compute_gyration_tensor(pocket_coords)
            pocket_eigenvalues, pocket_eigenvectors = np.linalg.eigh(pocket_gyr)
            pocket_principal = pocket_eigenvectors[:, -1]
        else:
            pocket_principal = np.array([1.0, 0.0, 0.0])
            pocket_eigenvalues = np.array([1.0, 1.0, 1.0])

        # Cosine between principal axes (absolute value: direction is arbitrary)
        cos_alignment = float(
            np.abs(np.dot(lig_principal, pocket_principal))
        )

        # 5c. Asphericity from gyration tensor eigenvalues
        # asphericity = (lam3 - 0.5*(lam1+lam2)) / (lam1+lam2+lam3)
        lam = np.sort(lig_eigenvalues)
        total = np.sum(lam)
        if total > 1e-10:
            asphericity = float((lam[2] - 0.5 * (lam[0] + lam[1])) / total)
        else:
            asphericity = 0.0

        # 5d. Compactness: radius of gyration
        rg = float(np.sqrt(np.sum(lig_eigenvalues)))
    else:
        cos_alignment = 0.0
        asphericity = 0.0
        rg = 0.0

    # ── Assemble feature vector ──
    features = np.concatenate([
        per_residue_contacts,                          # n_pocket dims
        centroid_disp.astype(np.float32),              # 3 dims
        np.array([ligand_span, overlap_fraction, pocket_fill],
                 dtype=np.float32),                    # 3 dims
        np.array([score_spread], dtype=np.float32),    # 1 dim
        np.array([cos_alignment, asphericity, rg],
                 dtype=np.float32),                    # 3 dims
    ])

    return features.astype(np.float32)


def get_geometric_feature_names(
    pocket_residues: List[Tuple[str, int, str]],
) -> List[str]:
    """Return human-readable names for each geometric feature dimension.

    Args:
        pocket_residues: The pocket residue list used for feature computation.

    Returns:
        List of feature name strings, one per dimension.
    """
    names = []
    for chain, resnum, resname in pocket_residues:
        names.append(f"contact_{resname}{resnum}_{chain}")
    names.extend([
        "centroid_disp_x", "centroid_disp_y", "centroid_disp_z",
        "ligand_span", "overlap_fraction", "pocket_fill_fraction",
        "score_spread",
        "principal_axis_cosine", "asphericity", "radius_of_gyration",
    ])
    return names


def compute_all_geometric_features(
    poses_dir: str,
    receptor_path: str,
    mol_ids: List,
    pocket_residues: Optional[List[Tuple[str, int, str]]] = None,
    pose_filename_template: str = "{mol_id}_pose.pdbqt",
    contact_cutoff: float = 4.5,
    pocket_cutoff: float = 6.0,
    cache_path: Optional[str] = None,
) -> Dict:
    """Batch computation of geometric features with caching.

    If ``pocket_residues`` is None, they are auto-detected from the poses
    directory using :func:`get_pocket_residues`.

    Args:
        poses_dir: Directory containing pose PDBQT files.
        receptor_path: Path to receptor PDBQT file.
        mol_ids: List of molecule identifiers.
        pocket_residues: Pre-computed pocket residue list, or None to
            auto-detect.
        pose_filename_template: Filename template with ``{mol_id}``.
        contact_cutoff: Distance cutoff for per-residue contacts.
        pocket_cutoff: Distance cutoff for pocket atom identification.
        cache_path: Optional .npz path to cache/load results.

    Returns:
        Dictionary mapping mol_id -> feature vector (np.ndarray).
    """
    # Try loading from cache
    if cache_path and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached = dict(data["features"].item())
            if all(mid in cached for mid in mol_ids):
                logger.info(
                    f"Loaded {len(cached)} geometric features from cache"
                )
                return {mid: cached[mid] for mid in mol_ids}
            else:
                logger.info("Geometric cache incomplete, recomputing...")
        except Exception as e:
            logger.warning(f"Failed to load geometric cache {cache_path}: {e}")

    # Auto-detect pocket if not provided
    if pocket_residues is None:
        pocket_cache = None
        if cache_path:
            pocket_cache = cache_path.replace(".npz", "_pocket.json")
        pocket_residues = get_pocket_residues(
            receptor_path,
            poses_dir,
            n_samples=50,
            contact_cutoff=pocket_cutoff,
            cache_path=pocket_cache,
        )

    if not pocket_residues:
        logger.error("No pocket residues detected; cannot compute geometric features")
        return {}

    n_pocket = len(pocket_residues)
    feat_dim = n_pocket + 10
    logger.info(
        f"Computing geometric features: {n_pocket} pocket residues, "
        f"{feat_dim} total dims"
    )

    results = {}
    n_success = 0
    n_fail = 0

    for mol_id in mol_ids:
        fname = pose_filename_template.format(mol_id=mol_id)
        pose_path = os.path.join(poses_dir, fname)

        if not os.path.exists(pose_path):
            results[mol_id] = np.full(feat_dim, np.nan, dtype=np.float32)
            n_fail += 1
            continue

        try:
            feat = compute_geometric_features(
                pose_path,
                receptor_path,
                pocket_residues,
                contact_cutoff=contact_cutoff,
                pocket_cutoff=pocket_cutoff,
            )
            results[mol_id] = feat
            if not np.all(np.isnan(feat)):
                n_success += 1
            else:
                n_fail += 1
        except Exception as e:
            logger.warning(f"Geometric features failed for mol_id={mol_id}: {e}")
            results[mol_id] = np.full(feat_dim, np.nan, dtype=np.float32)
            n_fail += 1

    logger.info(
        f"Geometric features: {n_success} success, {n_fail} failed "
        f"out of {len(mol_ids)} molecules"
    )

    # Save cache
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, features=results)
            logger.info(f"Saved geometric feature cache to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save geometric cache: {e}")

    return results
