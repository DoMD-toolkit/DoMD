from typing import Union

from rdkit import Chem


def divide_into_molecules(aa_system):
    res = []
    for m in Chem.rdmolops.GetMolFrags(aa_system, asMols=True):
        n = Chem.RWMol()
        for atom in m.GetAtoms():
            n.AddAtom(atom)
        for bond in m.GetBonds():
            n.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType())
        res.append(n)
    return res


def set_molecule_id_for_h(molecule: Union[Chem.RWMol, Chem.Mol]) -> Union[Chem.RWMol, Chem.Mol]:
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 1:
            for nbr_atom in atom.GetNeighbors():
                if nbr_atom.GetAtomicNum() == 1:
                    nbr_atom.SetIntProp("res_id", atom.GetIntProp("res_id"))
                    nbr_atom.SetIntProp("global_res_id", atom.GetIntProp("global_res_id"))
                    nbr_atom.SetProp('res_name', atom.GetProp('res_name'))
    return molecule
