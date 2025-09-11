# Copyright (c) 2024 Rui Shi, Mingyang Li, Hujun Qian
# Licensed under the PolyForm Noncommercial License v1.0.0
#  https://polyformproject.org/licenses/noncommercial/1.0.0/
# Commercial Licensing
#  For commercial use or to enter into a commercial license agreement, please contact: lmy23@mails.jlu.edu.cn

from collections import namedtuple

from rdkit.Chem import rdChemReactions

atom_info = namedtuple('atom_info',
                       ('reactant_id', 'reactant_atom_id', 'product_id', 'product_atom_id', 'atomic_number'))
bond_info = namedtuple('bond_info',
                       ('product_id', 'product_atoms_id', 'reactants_id', 'reactant_atoms_id', 'bond_type', 'status'))


def process_reactants(reactants):
    for r_idx, m in enumerate(reactants):
        for atom in m.GetAtoms():
            atom.SetIntProp('reactant_idx', r_idx)
    return reactants


def map_reacting_atoms(reaction):
    reacting_map = {}
    for r_idx in range(reaction.GetNumReactantTemplates()):
        # print(r_idx, reaction.GetNumReactantTemplates())
        rt = reaction.GetReactantTemplate(r_idx)
        for atom in rt.GetAtoms():
            if atom.GetAtomMapNum():
                reacting_map[atom.GetAtomMapNum()] = r_idx
    return reacting_map


def map_atoms(products, reacting_map):
    amap = []
    ra_dict = {}
    for ip, p in enumerate(products):
        p_idx = ip
        for a in p.GetAtoms():
            p_aidx = a.GetIdx()
            old_mapno = a.GetPropsAsDict().get('old_mapno')
            r_aidx = a.GetPropsAsDict().get("react_atom_idx")
            if old_mapno is not None:  # reacting atoms
                r_idx = reacting_map[old_mapno]
                if ra_dict.get(r_idx) is None:
                    ra_dict[r_idx] = []
                ra_dict[r_idx].append(r_aidx)
                amap.append(atom_info(r_idx, r_aidx, p_idx, p_aidx, a.GetAtomicNum()))
    return amap, ra_dict


def atom_map(products, reaction):
    reacting_map = map_reacting_atoms(reaction)
    amap, reacting_atoms = map_atoms(products, reacting_map)
    return amap, reacting_atoms


def bond_map(reactants: list, products: list, reaction: rdChemReactions.ChemicalReaction) -> list:
    amap, reacting_atoms = atom_map(products, reaction)
    res = []
    for ir, r in enumerate(reactants):
        for bond in r.GetBonds():  # exist in reactants, but not in production
            a_pid = b_pid = None
            a, b, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
            for atom in amap:
                if atom.reactant_id == ir:
                    if atom.reactant_atom_id == a:
                        a_pid = atom.product_id
                        a_paid = atom.product_atom_id
                    if atom.reactant_atom_id == b:
                        b_pid = atom.product_id
                        b_paid = atom.product_atom_id
            if a_pid is None or b_pid is None:
                continue
            if a_pid != b_pid:
                res.append(bond_info(a_pid, (a_paid, b_paid), (ir, ir), (a, b), t, 'deleted'))
            if a_pid == b_pid:
                p_bond = products[a_pid].GetBondBetweenAtoms(a_paid, b_paid)
                if p_bond is None:
                    res.append(bond_info(a_pid, (a_paid, b_paid), (ir, ir), (a, b), t, 'deleted'))

    for ip, p in enumerate(products):
        for bond in p.GetBonds():
            a_rid = b_rid = None
            a, b, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
            for atom in amap:
                if atom.product_id == ip:
                    if atom.product_atom_id == a:
                        a_rid = atom.reactant_id
                        a_raid = atom.reactant_atom_id
                    if atom.product_atom_id == b:
                        b_rid = atom.reactant_id
                        b_raid = atom.reactant_atom_id
            if a_rid is None or b_rid is None:
                continue
            if a_rid != b_rid:
                res.append(bond_info(ip, (a, b), (a_rid, b_rid), (a_raid, b_raid), t, 'new'))
            if a_rid == b_rid:
                r_bond = reactants[a_rid].GetBondBetweenAtoms(a_raid, b_raid)
                if r_bond is None:
                    res.append(bond_info(ip, (a, b), (a_rid, b_rid), (a_raid, b_raid), t, 'new'))
                else:
                    if r_bond.GetBondType() != t:
                        res.append(bond_info(ip, (a, b), (a_rid, b_rid), (a_raid, b_raid), t, 'changed'))
    return res

