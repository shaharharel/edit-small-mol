"""Warhead-seeded autoregressive inpainting for ZAP70 Cys346.

Usage (T4):
  cd ~/PocketFlow
  python ~/edit-small-mol/anchordiff/pocketflow_covind/inpaint.py \
      -pkt ~/edit-small-mol/anchordiff/pockets/zap70_cys346/pocket.pdb \
      --warhead_sdf ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf \
      --ckpt ~/PocketFlow/ckpt/ZINC-pretrained-255000.pt \
      -n 25 --name zap70_inpaint --root_path ~/runs/pocketflow_inpaint

Differences vs vanilla `main_generate.py`:
  1. Replaces `Ligand.empty_dict()` with a populated dict containing the
     5 acrylamide atoms (loaded from --warhead_sdf).
  2. Subclasses `Generate` to override `run()` so the autoregressive loop
     starts at atom_idx = n_seed_atoms (i.e. focal detection runs on the
     existing ligand context, NOT on protein surface). This forces growth
     to extend FROM the Cβ outward — bond chemistry is decided by the model.

No post-hoc bond fixing. The 5 seed bonds (C=C, C-C, C=O, C-N) are added to
`ligand_context_bond_index` and PocketFlow's autoregressive bond predictor
extends from there.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import torch
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# Make PocketFlow importable
POCKETFLOW_ROOT = os.path.expanduser("~/PocketFlow")
sys.path.insert(0, POCKETFLOW_ROOT)

from pocket_flow import PocketFlow, Generate  # type: ignore
from pocket_flow.utils import (  # type: ignore
    Protein, ComplexData, torchify_dict,
    FeaturizeProteinAtom, FeaturizeLigandAtom, FocalMaker, AtomComposer,
    RefineData, LigandCountNeighbors, mask_node, verify_dir_exists,
)

# import warhead seed (use local-dir import to avoid package-init dependency)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from warhead_seed import build_seed_or_empty  # type: ignore


def _safe_generate(self, data, num_gen=100, rec_name='recptor', with_print=True, root_path='gen_results'):
    """Bug-fix wrapper for Generate.generate: PocketFlow's own code references
    `mol` before it's been assigned when self.run() returns a falsy value.
    Same behavior as the original otherwise.
    """
    import time as _time
    from rdkit import Chem as _Chem
    from pocket_flow.utils import verify_dir_exists as _vdir, substructure as _substr
    date = _time.strftime("%Y-%m-%d-%H-%M-%S", _time.localtime())
    out_dir = root_path + '/' + rec_name + '/' + date + '/'
    self.out_dir = out_dir
    _vdir(out_dir)
    valid_mol = []
    smiles_list = []
    valid_counter = 0
    for i in range(num_gen):
        data_clone = data.clone().detach()
        mol = None
        try:
            out = self.run(data_clone)
        except Exception as e:
            print(f"  [gen {i}] run error: {type(e).__name__}: {str(e)[:120]}")
            out = None
        if out:
            mol, _ = out
        del data_clone
        if mol is not None:
            mol.SetProp('_Name', f'No_{valid_counter}-{out_dir}')
            smi = _Chem.MolToSmiles(mol)
            if with_print:
                print(smi)
            with open(out_dir + 'generated.sdf', 'a') as sdf_writer:
                sdf_writer.write(_Chem.MolToMolBlock(mol) + '\n$$$$\n')
            with open(out_dir + 'generated.smi', 'a') as smi_writer:
                smi_writer.write(smi + '\n')
            smiles_list.append(smi)
            valid_mol.append(mol)
            valid_counter += 1
    n_tot = max(1, num_gen)
    print(f"\nvalid={len(smiles_list)}  unique={len(set(smiles_list))}")
    print(f"Validity: {len(smiles_list)/n_tot:.4f}")
    if len(smiles_list):
        print(f"Unique: {len(set(smiles_list))/len(smiles_list):.4f}")
    out_stat = {
        'Validity': len(smiles_list) / n_tot,
        'Unique': (len(set(smiles_list)) / max(1, len(smiles_list))) if smiles_list else 0.0,
    }
    try:
        out_stat['ring_size'] = _substr([valid_mol])
    except Exception:
        pass
    with open(out_dir + 'metrics.dir', 'w') as fw:
        fw.write(str(out_stat))


class SeededGenerate(Generate):
    """Subclass of `Generate` that starts the autoregressive loop at
    `atom_idx = n_seed_atoms` instead of 0.

    PocketFlow's existing `choose_focal` already handles `atom_idx != 0`
    by selecting from `idx_ligand_ctx_in_cpx` — which is exactly what we
    want once the warhead is in the ligand context. So we only need to
    bump the loop start.
    """

    def __init__(self, *a, n_seed_atoms: int = 0,
                 cb_seed_idx: int | None = None,
                 cb_focal_bias: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.n_seed_atoms = int(n_seed_atoms)
        # Index of Cβ atom inside the seed (i.e. position in
        # ligand_context_element / idx_ligand_ctx_in_cpx). If None, no bias.
        self.cb_seed_idx = None if cb_seed_idx is None else int(cb_seed_idx)
        self.cb_focal_bias = float(cb_focal_bias)

    # bind safe generator
    generate = _safe_generate

    def choose_focal(self, h_cpx, cpx_index, idx_ligand_ctx_in_cpx, data, atom_idx):
        """Override that adds a +cb_focal_bias log-bias to the Cβ atom's focal
        logit so autoregressive growth actually starts FROM the warhead Cβ.

        Replicates Generate.__choose_focal's logic (it's name-mangled, so we
        re-implement it here) but injects the bias on focal_pred (logit space)
        BEFORE sigmoid/argmax.

        Cβ identification: it sits at position ``self.cb_seed_idx`` inside the
        ligand-context array. Once the position has no free valence, the
        post-`__choose_focal` valence_mask in the parent removes it, so the
        bias naturally decays as growth proceeds. We mirror that valence
        gating below.
        """
        import torch as _t

        # protein-only first-atom path: no warhead context yet, no bias to apply.
        # (In seeded mode we ALWAYS have atom_idx >= n_seed_atoms > 0, so this
        # branch is for safety only.)
        if atom_idx == 0:
            return super().choose_focal(h_cpx, cpx_index, idx_ligand_ctx_in_cpx, data, atom_idx)

        # Replicate __choose_focal(ctx_idx=idx_ligand_ctx_in_cpx) inline so we can bias
        focal_pred = self.model.focal_net(h_cpx, idx_ligand_ctx_in_cpx)  # (n_lig, 1)

        # Inject Cβ bias. cb_seed_idx is a position in idx_ligand_ctx_in_cpx
        # (= position in ligand_context_element). focal_pred is parallel to
        # idx_ligand_ctx_in_cpx, so we index by cb_seed_idx directly.
        bias_applied = False
        if (
            self.cb_seed_idx is not None
            and self.cb_focal_bias != 0.0
            and self.cb_seed_idx < focal_pred.size(0)
        ):
            # Only apply when Cβ still has free valence — otherwise the
            # downstream valence_mask drops it and the bias is wasted.
            max_v = data.max_atom_valence[self.cb_seed_idx].item()
            cur_v = data.ligand_context_valence[self.cb_seed_idx].item()
            if cur_v < max_v:
                focal_pred = focal_pred.clone()
                focal_pred[self.cb_seed_idx] = focal_pred[self.cb_seed_idx] + self.cb_focal_bias
                bias_applied = True

        focal_prob = _t.sigmoid(focal_pred).view(-1)
        # In atom_idx != 0 path the parent uses choose_max=1 always
        max_idx = focal_pred.argmax()
        focal_idx_ = idx_ligand_ctx_in_cpx[max_idx].view(-1)
        focal_prob = focal_prob[max_idx].view(-1)

        # Telemetry: print whenever the bias actually flipped the argmax to Cβ
        if bias_applied and atom_idx == self.n_seed_atoms:
            chosen_pos = int(max_idx.item())
            tag = "CB" if chosen_pos == self.cb_seed_idx else f"pos{chosen_pos}"
            print(f"  [choose_focal] atom_idx={atom_idx} bias_applied -> focal={tag}")

        # Reproduce parent's valence-check tail (only the atom_idx != 0 branch)
        focal_valence_check = False
        if data.ligand_context_element.size(0) > 3 and atom_idx != 0:
            max_valence = data.max_atom_valence[focal_idx_]
            valence_in_ligand_context_focal = data.ligand_context_valence[focal_idx_]
            valence_mask = max_valence > valence_in_ligand_context_focal
            focal_idx_ = focal_idx_[valence_mask]
            focal_prob = focal_prob[valence_mask]
            focal_valence_check = valence_mask.sum() == 0
            if focal_valence_check:
                return False
            self.counter += 1
        if focal_valence_check:
            return False
        return focal_idx_, focal_prob

    def run(self, data):
        # Mirrors Generate.run but enters the loop at atom_idx = n_seed_atoms.
        import torch as _t
        from torch_geometric.nn import knn as _knn, radius as _radius  # noqa
        from rdkit import Chem as _Chem
        from pocket_flow.utils import (  # noqa
            get_tri_edges, add_ligand_atom_to_data, data2mol, modify, check_alert_structures,
        )
        from pocket_flow.gdbp_model import embed_compose

        data.max_atom_valence = _t.empty(0, dtype=_t.long)
        # max_atom_valence has to be populated for the n_seed_atoms entries we
        # already inserted via the ligand dict
        if self.n_seed_atoms > 0 and hasattr(data, "ligand_context_element"):
            mv = {1:1,6:4,7:3,8:2,9:1,15:5,16:6,17:1,35:1,53:1}
            for el in data.ligand_context_element[:self.n_seed_atoms].tolist():
                data.max_atom_valence = _t.cat([
                    data.max_atom_valence, _t.LongTensor([mv.get(int(el), 4)])
                ])
        data = data.to(self.device)

        with _t.no_grad():
            self.prior_node = _t.distributions.normal.Normal(
                _t.zeros([len(self.atom_type_map)]).cuda(data.cpx_pos.device),
                self.temperature[0] * _t.ones([len(self.atom_type_map)]).cuda(data.cpx_pos.device),
            )
            self.prior_edge = _t.distributions.normal.Normal(
                _t.zeros([self.num_bond_type]).cuda(data.cpx_pos.device),
                self.temperature[1] * _t.ones([self.num_bond_type]).cuda(data.cpx_pos.device),
            )

            rw_mol = _Chem.RWMol()
            # Pre-populate rw_mol with the seed atoms + bonds so SMILES output
            # contains them. (PocketFlow's `data2mol` rebuilds from `data`
            # anyway, so rw_mol is mostly fallback path.)
            if self.n_seed_atoms > 0 and hasattr(data, "ligand_context_element"):
                for el in data.ligand_context_element[:self.n_seed_atoms].tolist():
                    rw_mol.AddAtom(_Chem.Atom(int(el)))
                # bonds: ligand_context_bond_index is 2xE, both directions; dedup
                seen = set()
                for k in range(data.ligand_context_bond_index.size(1)):
                    i = int(data.ligand_context_bond_index[0, k])
                    j = int(data.ligand_context_bond_index[1, k])
                    bt = int(data.ligand_context_bond_type[k])
                    if i >= self.n_seed_atoms or j >= self.n_seed_atoms:
                        continue
                    if (min(i,j), max(i,j)) in seen:
                        continue
                    seen.add((min(i,j), max(i,j)))
                    if bt in self.bond_type_map:
                        try:
                            rw_mol.AddBond(i, j, self.bond_type_map[bt])
                        except Exception:
                            pass

            self.counter = 0
            for atom_idx in range(self.n_seed_atoms, self.max_atom_num):
                data = data.to(self.device)
                h_cpx = embed_compose(
                    data.cpx_feature.float(), data.cpx_pos, data.idx_ligand_ctx_in_cpx,
                    data.idx_protein_in_cpx, self.model.ligand_atom_emb,
                    self.model.protein_atom_emb, self.model.emb_dim,
                )
                h_cpx = self.model.encoder(
                    node_attr=h_cpx, pos=data.cpx_pos,
                    edge_index=data.cpx_edge_index, edge_feature=data.cpx_edge_feature,
                )
                self.resample_edge_faild = False
                self.check_node = True
                self.resample_node = 0
                while self.check_node:
                    if self.resample_node > 50:
                        break
                    # Since atom_idx >= n_seed_atoms > 0, choose_focal uses
                    # ligand-context idx — focus picked among warhead atoms.
                    focal_out = self.choose_focal(
                        h_cpx, data.idx_protein_in_cpx, data.idx_ligand_ctx_in_cpx,
                        data, atom_idx,
                    )
                    if focal_out is False:
                        break
                    focal_idx, focal_prob = focal_out
                    new_atom_type, focal_idx = self.atom_generate(
                        h_cpx, focal_idx, focal_prob, atom_idx,
                    )
                    atom_type_emb = self.model.atom_type_embedding(new_atom_type).view(-1, self.hidden_channels)
                    new_pos_to_add = self.pos_generate(
                        h_cpx, atom_type_emb, focal_idx, data.cpx_pos, atom_idx,
                    )
                    if new_pos_to_add is False:
                        self.resample_node += 1
                        continue
                    rw_mol.AddAtom(_Chem.Atom(self.atom_type_map[new_atom_type]))
                    bond_out = self.bond_generate(
                        h_cpx, data, new_pos_to_add, atom_type_emb, atom_idx, rw_mol,
                    )
                    if bond_out is not False:
                        rw_mol, new_edge_idx, new_bond_type_to_add = bond_out
                        has_alert = check_alert_structures(
                            rw_mol, ['[O]-[O]', '[N]-[O,Br,Cl,I,F,P]',
                                     '[S,P]-[Br,Cl,I,F]', '[P]-[O]-[P]',
                                     '[Br,Cl,I,F]-[Br,Cl,I,F]'],
                        )
                        if has_alert:
                            for ix in range(new_edge_idx.size(1)):
                                _, j = new_edge_idx[:, ix].tolist()
                                rw_mol.RemoveBond(atom_idx, j)
                            rw_mol.RemoveAtom(atom_idx)
                            self.resample_node += 1
                            continue
                        else:
                            break
                    if self.resample_edge_faild:
                        break
                if self.resample_edge_faild or self.resample_node > 50 or focal_out is False:
                    break
                data = data.to('cpu')
                data = add_ligand_atom_to_data(
                    data, new_pos_to_add.to('cpu'), new_atom_type.to('cpu'),
                    new_edge_idx.to('cpu'), new_bond_type_to_add.to('cpu'),
                    type_map=self.atom_type_map,
                )
                data = self.transform(data)
        try:
            mol = data2mol(data)
            modified_mol = modify(mol, max_double_in_6ring=self.max_double_in_6ring)
            return modified_mol, mol
        except Exception:
            mol_ = rw_mol.GetMol()
            print("Invalid mol:", _Chem.MolToSmiles(mol_))
            try:
                mol = data2mol(data)
                return None
            except Exception:
                return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-pkt", "--pocket", required=True)
    ap.add_argument("--warhead_sdf", required=True,
                    help="5-atom acrylamide SDF placed at canonical Cys346 position")
    ap.add_argument("--ckpt", default=os.path.expanduser("~/PocketFlow/ckpt/ZINC-pretrained-255000.pt"))
    ap.add_argument("-n", "--num_gen", type=int, default=25)
    ap.add_argument("--name", default="zap70_inpaint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--atom_temperature", type=float, default=1.0)
    ap.add_argument("--bond_temperature", type=float, default=1.0)
    ap.add_argument("--max_atom_num", type=int, default=40)
    ap.add_argument("--focus_threshold", type=float, default=0.3,
                    help="Lowered from 0.5 → 0.3 per QA verdict (Day-2 fix): the focal classifier was trained on ZINC; with warhead-seeded context it under-confidence-rated focal candidates, causing 13/25 mols to terminate early.")
    ap.add_argument("--choose_max", type=str, default="1")
    ap.add_argument("--min_dist_inter_mol", type=float, default=3.0)
    ap.add_argument("--bond_length_range", default="(1.0,2.0)")
    ap.add_argument("--max_double_in_6ring", type=int, default=0)
    ap.add_argument("--cb_focal_bias", type=float, default=0.0,
                    help="Add +bias (in logit space) to Cβ atom's focal score "
                         "so growth actually starts FROM the warhead. 0 = off. "
                         "Recommended: ~2.0 (raises sigmoid prob by 0.88 at logit=0).")
    ap.add_argument("--root_path", default=os.path.expanduser("~/runs/pocketflow_inpaint"))
    args = ap.parse_args()

    choose_max = args.choose_max.lower() in {"1", "true", "yes", "t", "y"}
    if isinstance(args.bond_length_range, str):
        args.bond_length_range = eval(args.bond_length_range)

    print(f"[inpaint] pocket={args.pocket}")
    print(f"[inpaint] warhead={args.warhead_sdf}")
    print(f"[inpaint] ckpt={args.ckpt}")

    # parse pocket
    pro_dict = Protein(args.pocket).get_atom_dict(removeHs=True, get_surf=True)

    # SEED warhead ligand dict
    lig_dict, cb_seed_idx = build_seed_or_empty(args.warhead_sdf)
    n_seed = len(lig_dict.element)
    print(f"[inpaint] seeded ligand with n={n_seed} warhead atoms, Cβ focus idx={cb_seed_idx}")

    data = ComplexData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pro_dict),
        ligand_dict=torchify_dict(lig_dict),
    )

    # Standard PocketFlow transform pipeline
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6, 7, 8, 9, 15, 16, 17, 35, 53])
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)

    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    # IMPORTANT: PocketFlow's mask_node signature is (data, context_idx, masked_idx).
    # We want the 5 seed atoms in CONTEXT (visible) and nothing in MASKED (no
    # next-atom-prediction target at the start — Generate.run will autoregress
    # from the seed forward).
    context_idx = torch.arange(data.ligand_pos.size(0))  # all seed atoms in context
    data = mask_node(data, context_idx, torch.empty([0], dtype=torch.long),
                     num_atom_type=9, y_pos_std=0.)
    data = atom_composer.run(data)

    # Load model
    print("[inpaint] loading model ...")
    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = ckpt["config"]
    model = PocketFlow(config).to(args.device)
    model.load_state_dict(ckpt["model"])

    print("[inpaint] generating ...")
    gen = SeededGenerate(
        model, atom_composer.run,
        temperature=[args.atom_temperature, args.bond_temperature],
        atom_type_map=[6, 7, 8, 9, 15, 16, 17, 35, 53],
        num_bond_type=4, max_atom_num=args.max_atom_num,
        focus_threshold=args.focus_threshold,
        max_double_in_6ring=args.max_double_in_6ring,
        min_dist_inter_mol=args.min_dist_inter_mol,
        bond_length_range=args.bond_length_range,
        choose_max=choose_max, device=args.device,
        n_seed_atoms=n_seed,
        cb_seed_idx=cb_seed_idx,
        cb_focal_bias=args.cb_focal_bias,
    )
    t0 = time.time()
    gen.generate(data, num_gen=args.num_gen, rec_name=args.name,
                 with_print=True, root_path=args.root_path)
    print(f"[inpaint] done in {time.time()-t0:.1f}s — output dir: {gen.out_dir}")
    return gen.out_dir


if __name__ == "__main__":
    main()
