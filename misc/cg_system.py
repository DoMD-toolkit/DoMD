from typing import Tuple, Dict, List

import networkx as nx

from misc.logger import logger, DuplicateFilter


def read_cg_topology(cg_system, residues: Dict) -> Tuple[nx.Graph, List[nx.Graph]]:
    r"""
    :param cg_system: a cg_system is an object and cg_system.data contains
    'position', 'bond', 'type', ...
    :param residues: dict, the residue meta, {monomer_type: {'smiles':...,}}
    :return: list of cg_molecules, each cg_molecule is a Graph, its node is the
    global id of residue, and a local residue id in molecule is generated automatically
    the Graph node contains 'type', 'smiles', 'x' and 'local_res_id'.
    """
    cg_sys = nx.Graph()
    for bond in cg_system.data['bond']:
        bond_type, i, j = str(bond[0]), int(bond[1]), int(bond[2])
        type_i, type_j = cg_system.data['type'][i], cg_system.data['type'][j]
        ri = residues.get(type_i) or {}
        rj = residues.get(type_j) or {}
        if not (ri or rj):
            with DuplicateFilter(logger):
                logger.warning(f"The residues {residues} do not contain "
                               f"residue information for type {type_i} or {type_j}, "
                               f"this is usually for manually operations.")
        cg_sys.add_node(i, type=type_i,
                        smiles=ri.get('smiles'),
                        x=cg_system.data['position'][i] * 10
                        )
        cg_sys.add_node(j, type=type_j,
                        smiles=rj.get('smiles'),
                        x=cg_system.data['position'][j] * 10
                        )
        cg_sys.add_edge(i, j, bond_type=bond_type)
    # singe = nx.Graph()
    for n in range(cg_system.data['type'].__len__()):
        if cg_sys.nodes.get(n) is None:
            t = cg_system.data['type'][n]
            r = residues.get(t) or {}
            cg_sys.add_node(n, type=t, smiles=r.get('smiles'),
                            x=cg_system.data['position'][n] * 10)
    cg_molecules = [cg_sys.subgraph(c).copy() for c in nx.connected_components(cg_sys)]
    for cg_mol in cg_molecules:
        for res_id, node in enumerate(cg_mol):
            cg_mol.nodes[node]["local_res_id"] = res_id
    return cg_sys, cg_molecules
