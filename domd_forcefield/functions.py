from typing import Union

import networkx as nx
import rdkit
from rdkit import Chem

from misc.logger import logger


def get_submol_rad_n(mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol],
                     radius: int, atom: Chem.Atom) -> Union[None, tuple[Chem.Mol, dict[int, int], dict, str]]:
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
    if not env:
        return
    amap = {}
    sub_mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smi = Chem.MolToSmiles(sub_mol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)
    return sub_mol, amap, env, sub_smi


def submol2graph(mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol],
                 atom: Chem.Atom, radius: int) -> Union[None, tuple[nx.Graph, str]]:
    g = nx.Graph()
    sa_prop = {}
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom.GetIdx(), useHs=True)
    if not env:
        return
    amap = {}
    sub_mol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smi = Chem.MolToSmiles(sub_mol, rootedAtAtom=amap[atom.GetIdx()], canonical=False)  # , allHsExplicit=True)
    nm = 0
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 1:
            nm += nbr.GetAtomicNum()
    h = atom.GetHybridization()
    for a in amap:
        atom_ = mol.GetAtomWithIdx(a)
        h_ = atom_.GetHybridization()
        nm_ = 0
        for nbr in atom_.GetNeighbors():
            if nbr.GetAtomicNum() != 1:
                nm_ += nbr.GetAtomicNum()

        # sa_prop[amap[a]] = (atom.GetAtomicNum(), int(atom.GetIsAromatic()),
        #                    int(atom.IsInRing()), h.real, h.imag, atom_.GetAtomicNum(),
        #                    int(atom_.GetIsAromatic()), int(atom_.IsInRing()), nm, h_.real, h_.imag)
        sa_prop[amap[a]] = (atom.GetAtomicNum(), int(atom.GetIsAromatic()), int(atom.IsInRing()),
                            atom_.GetAtomicNum(), int(atom_.GetIsAromatic()), int(atom_.IsInRing()))

    for bond in sub_mol.GetBonds():
        ai = bond.GetBeginAtom()
        aj = bond.GetEndAtom()
        g.add_node(ai.GetIdx(), desc=sa_prop[ai.GetIdx()])
        g.add_node(aj.GetIdx(), desc=sa_prop[aj.GetIdx()])
        g.add_edge(ai.GetIdx(), aj.GetIdx(), bo=f'{bond.GetBondTypeAsDouble():.1f}')
    return g, sub_smi


def atom_stats(mol: Union[Chem.RWMol, rdkit.Chem.rdchem.Mol], radius: int = 3) -> dict[int, list]:
    ret = {}
    for atom in mol.GetAtoms():
        res = None
        for n in range(radius, 0, -1):  # 2 is ok for atom_types, make it larger for charge databases
            res = submol2graph(mol, atom, n)
            if res is not None:
                g, sub_smi = res
                break
        if res is not None:
            env_hash = nx.weisfeiler_lehman_graph_hash(g, node_attr='desc', edge_attr='bo')
            ret[atom.GetIdx()] = [atom.GetIdx(), atom.GetSymbol(), sub_smi, env_hash]
        else:
            logger.error(f"Atom {atom.GetIdx()}, {atom.GetSymbol()} "
                         f"in mol ({Chem.MolToSmiles(mol)}) "
                         f"with {mol.GetNumAtoms()} atoms can not be hashed ")
            raise ValueError(f"Atom {atom.GetIdx()}, {atom.GetSymbol()} "
                             f"in mol ({Chem.MolToSmiles(mol)}) "
                             f"with {mol.GetNumAtoms()} atoms can not be hashed ")
    return ret


def bonded_hash(atom_env_hashes: list[str]) -> str:
    r"""For any boned type, i.e., bonds, angles or torsions, L and L[::-1] are enough
    :param atom_env_hashes:
    :return:
    """
    sorted_vals = ''.join(atom_env_hashes) + ''.join(atom_env_hashes[::-1])
    # return hashlib.sha256(sorted_vals.encode()).hexdigest()
    return sorted_vals


def improper_hash(center_hash: str) -> str:
    sorted_vals = center_hash  # center is enough for improper
    # return hashlib.sha256(sorted_vals.encode()).hexdigest()
    return center_hash
