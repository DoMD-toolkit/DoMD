from typing import Union, Any

import networkx as nx
import numpy as np
from rdkit import Chem

from domd_forcefield.forcefield import ForceField


def assemble_opls(molecules: list[Union[Chem.Mol, Chem.RWMol]],
                  forcefields: list[ForceField],
                  conformers: list[Chem.Conformer]) -> Any:
    counter = 0
    cg_counter = 0
    system = nx.Graph()
    all_forcefields = {'atoms': {}, 'bonds': {}, 'angles': {}, 'dihedrals': {}, 'impropers': {}}
    # The force field data is data[atom_ind] = atom_type, data[(bond_i, bond_j)] = bond, etc.
    xyz = []
    local_graphs = []
    for m, f, c in zip(molecules, forcefields, conformers):
        molg = nx.Graph()
        for atom in m.GetAtoms():
            molg.add_node(atom.GetIdx(),
                            global_idx=atom.GetIdx() + counter,
                            symbol=atom.GetSymbol(),
                            bond_type=f.all_params['atoms'][atom.GetIdx()].bond_type,
                            res_id=atom.GetIntProp('global_res_id'),
                            res_name=atom.GetProp('res_name'))
            system.add_node(atom.GetIdx() + counter,
                            symbol=atom.GetSymbol(),
                            bond_type=f.all_params['atoms'][atom.GetIdx()].bond_type,
                            res_id=atom.GetIntProp('global_res_id'),
                            res_name = atom.GetProp('res_name'))
        for bond in m.GetBonds():
            system.add_edge(bond.GetBeginAtomIdx() + counter, bond.GetEndAtomIdx() + counter)
            molg.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        local_graphs.append(molg)
        for sect in all_forcefields.keys():
            for k in f.all_params[sect]:
                if sect == 'atoms':
                    key = k + counter
                else:
                    key = tuple([_ + counter for _ in k])
                all_forcefields[sect][key] = f.all_params[sect][k]
        xyz.append(c.GetPositions())
        counter += m.GetNumAtoms()
    return system, np.vstack(xyz), all_forcefields, local_graphs
