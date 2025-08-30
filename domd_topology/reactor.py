import cgi
from itertools import permutations
from typing import Any

import networkx as nx
import tqdm
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from misc.logger import logger
from ._mapping import process_reactants, atom_map, bond_map


def reaction_mol_mapping(reactions: list[tuple]) -> dict[int, set]:
    reaction_hash = {}
    for r in reactions:
        reaction_indices = r[1:]
        for rid in reaction_indices:
            if reaction_hash.get(rid) is None:
                reaction_hash[rid] = set()
            reaction_hash[rid].add(r)
    return reaction_hash


class Reaction(object):
    def __init__(self, name, cg_reactant_list, smarts, prod_idx=None):
        self.cg_reactant_list = cg_reactant_list
        self.reaction_name = name
        self.reaction = rdChemReactions.ReactionFromSmarts(smarts)
        self.smarts = smarts
        self.prod_idx = prod_idx
        self.reaction_maps = {}

    def build_reaction_maps(self, cg_reactants, molecules):
        if self.reaction_maps.get(cg_reactants) is None:
            self.reaction_maps[cg_reactants] = []
            reactants = process_reactants(molecules)
            products = self.reaction.RunReactants(reactants)
            if len(products) == 0:
                raise ValueError(f"Reaction {self.smarts} does not run on CG reactants {cg_reactants}")
            for product in products:
                amap, reacting_atoms = atom_map(product, self.reaction)
                bmap = bond_map(reactants, product, self.reaction)
                self.reaction_maps[cg_reactants].append((reacting_atoms, amap, bmap))


def allowed_p(reacted_atoms, cg_reactants, reaction):
    for reaction_map in reaction.reaction_maps.get(cg_reactants):
        allowed = True
        for ri in reaction_map[0]:
            if set.intersection(set(reaction_map[0][ri]), reacted_atoms[ri]):
                allowed = False
        if allowed:
            return reaction_map, reaction.prod_idx
    return None, None  # if no available reaction is chosen.


class Reactor(object):
    def __init__(self, reactants_meta: dict[str, dict[str, str]], reaction_templates: dict[str, dict[str, Any]]):
        self.reactants_meta = reactants_meta
        # self.cg_molecules = None
        # self.aa_molecules = []
        # self.meta = []
        self.reaction_templates = {}
        for reaction_name in reaction_templates:
            _info = reaction_templates[reaction_name]
            self.reaction_templates[reaction_name] = Reaction(
                reaction_name, _info['cg_reactant_list'], _info['smarts'], _info.get("prod_idx")
            )

    def process(self, cg_molecules: list[nx.Graph], reactions: list) -> tuple[list[Chem.RWMol], list[nx.Graph]]:
        aa_molecules = []
        meta = []
        reaction_hash = reaction_mol_mapping(reactions)
        for _i, cg_mol in enumerate(cg_molecules):
            logger.info(f"Generating top for CG molecule {_i} with residue num of {len(cg_mol)}")
            aa_mol = Chem.RWMol()
            mol_meta = nx.Graph()
            global_count = 0
            mol_reactions = set()
            for node in cg_mol.nodes:
                atom_idx = {}
                for r in reaction_hash[node]:
                    mol_reactions.add(r)
                reactant = cg_mol.nodes[node]
                reactant_molecule = Chem.MolFromSmiles(reactant['smiles'])
                for atom_id in range(reactant_molecule.GetNumAtoms()):
                    atom = reactant_molecule.GetAtomWithIdx(atom_id)
                    aa_mol.AddAtom(atom)
                    atom_idx[atom_id] = atom_id + global_count
                for bond in reactant_molecule.GetBonds():
                    aa_mol.AddBond(
                        bond.GetBeginAtomIdx() + global_count,
                        bond.GetEndAtomIdx() + global_count,
                        bond.GetBondType()
                    )
                global_count += reactant_molecule.GetNumAtoms()
                mol_meta.add_node(node, atom_idx=atom_idx, reacting_map={}, rm_atoms=set())
            if len(cg_mol.nodes) == 1:
                for m in mol_meta.nodes:
                    molecule = mol_meta.nodes[m]
                    for idx in molecule['atom_idx'].values():
                        atom = aa_mol.GetAtomWithIdx(idx)
                        atom.SetIntProp('global_res_id', int(m))
                        atom.SetProp('res_name', str(cg_mol.nodes[m].get('type')))
                        logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                        if cg_mol.nodes[m].get('local_res_id') is None:
                            logger.warning(f"No local_res_id found in cg_molecule!")
                            atom.SetIntProp('res_id', -1)
                        else:
                            logger.debug(f"local_res_id for res {m}: {cg_mol.nodes[m]['local_res_id']}")
                            atom.SetIntProp('res_id', cg_mol.nodes[m]['local_res_id'])
                aa_molecules.append(aa_mol)
                meta.append(mol_meta)
                continue
            for edge in cg_mol.edges:
                mol_meta.add_edge(*edge)
            r__id = 0
            for r in mol_reactions:
                r__id += 1
                reaction_name = r[0]
                _reactant_idx = r[1:]
                rxn_tpls = self.reaction_templates.get(reaction_name)
                if rxn_tpls is None:
                    raise ValueError(f"Reaction {r} is not defined in reaction_info!")

                _all_orders = list(permutations(_reactant_idx))
                _reactants_tuple = tuple([cg_mol.nodes[_]['type'] for _ in _reactant_idx])
                reactants_order = reactants_tuple = None
                for _order in _all_orders:
                    _tuple = tuple([cg_mol.nodes[_]['type'] for _ in _order])
                    if _tuple in rxn_tpls.cg_reactant_list:
                        reactants_order = _order
                        reactants_tuple = _tuple
                if not reactants_order:
                    raise ValueError(f"Reaction {r} for reactants ({_reactants_tuple}) is not defined!")

                _molecules = []
                for t in reactants_tuple:
                    _molecules.append(Chem.MolFromSmiles(self.reactants_meta[t]['smiles']))
                rxn_tpls.build_reaction_maps(reactants_tuple, _molecules)

                if len(reactants_tuple) == 2:
                    if reactants_tuple[0] == reactants_tuple[1]:
                        di_same = True
                        reactants_order = sorted(reactants_order)
                reactants = [mol_meta.nodes[_] for _ in reactants_order]
                key = tuple(sorted(_reactant_idx))
                reacted_atoms = {}

                for ri in range(len(reactants)):
                    reacted_atoms[ri] = set()

                for ri, rt in enumerate(reactants):  # keep reactant order
                    for k in rt['reacting_map']:
                        for at in rt['reacting_map'][k]:
                            reacted_atoms[ri].add(at)

                reaction_map, product_idx = allowed_p(reacted_atoms, reactants_tuple, rxn_tpls)
                if not reaction_map:
                    if not reaction_map:
                        raise (ValueError(
                            f"{r} with order {_reactant_idx}, {_reactants_tuple} can not react! This error happens while"
                            f"the reacted atoms in one bead have been reacted more than once. We reconmand you use another"
                            f"reaction template to avoid this."))

                amap, bmap = reaction_map[1], reaction_map[2]  # store the reacted atoms.
                for ri, rt in enumerate(reactants):
                    if rt['reacting_map'].get(key) is None:
                        rt['reacting_map'][key] = set()
                    for at in reaction_map[0][ri]:
                        rt['reacting_map'][key].add(at)

                for atom in amap:
                    if product_idx is not None:
                        if atom.product_id not in product_idx:
                            reactant = reactants[atom.reactant_id]
                            reactant['rm_atoms'].add(reactant['atom_idx'][atom.reactant_atom_id])
                for b in bmap:
                    if b.status == 'deleted':
                        reactant = reactants[b.reactants_id[0]]
                        bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                        aa_mol.RemoveBond(bi, bj)
                    if b.status == 'changed':
                        reactant = reactants[b.reactants_id[0]]
                        bi = reactant['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant['atom_idx'][b.reactant_atoms_id[1]]
                        bond = aa_mol.GetBondBetweenAtoms(bi, bj)
                        bond.SetBondType(b.bond_type)
                    if b.status == 'new':
                        reactant0 = reactants[b.reactants_id[0]]
                        reactant1 = reactants[b.reactants_id[1]]
                        bi = reactant0['atom_idx'][b.reactant_atoms_id[0]]
                        bj = reactant1['atom_idx'][b.reactant_atoms_id[1]]
                        aa_mol.AddBond(bi, bj, b.bond_type)

            rm_all = []
            for m in mol_meta.nodes:
                molecule = mol_meta.nodes[m]
                for idx in molecule['atom_idx'].values():
                    atom = aa_mol.GetAtomWithIdx(idx)
                    atom.SetIntProp('global_res_id', int(m))
                    atom.SetProp('res_name', str(cg_mol.nodes[m]['type']))
                    logger.debug(f"global_res_id for atom {idx} in residue {m} is {m}")
                    if cg_mol.nodes[m].get('local_res_id') is None:
                        logger.warning(f"No local_res_id found in cg_molecule!")
                        atom.SetIntProp('res_id', -1)
                    else:
                        logger.debug(f"local_res_id for res {m}: {cg_mol.nodes[m]['local_res_id']}")
                        atom.SetIntProp('res_id', cg_mol.nodes[m]['local_res_id'])
                rm_all.extend(list(molecule['rm_atoms']))

            rm_all = sorted(list(set(rm_all)), reverse=True)
            for bi in rm_all:
                aa_mol.RemoveAtom(bi)
            aa_molecules.append(aa_mol)
            meta.append(mol_meta)
        return aa_molecules, meta
