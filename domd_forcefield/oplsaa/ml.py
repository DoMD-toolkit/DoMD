import sys
#sys.path.append('e:\\downloads\\article\\high_throughput_system\\software\\DoMD\\')
import math
from typing import Any, Union

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import GetUFFBondStretchParams, GetUFFAngleBendParams, GetUFFTorsionParams
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from domd_forcefield.forcefield import CustomRule
from domd_forcefield.oplsaa.database import OplsAtom, OplsBonded
from misc.logger import logger

en = {
    "H": 2.300, "He": 4.160,
    "Li": 0.912, "Be": 1.576,
    "B": 2.051, "C": 2.544,
    "N": 3.066, "O": 3.610,
    "F": 4.193, "Ne": 4.787,
    "Na": 0.869, "Mg": 1.293,
    "Al": 1.613, "Si": 1.916,
    "P": 2.253, "S": 2.589,
    "Cl": 2.869, "Ar": 3.242,
    "K": 0.734, "Ca": 1.034,
    "Sc": 1.19, "Ti": 1.38,
    "V": 1.53, "Cr": 1.65,
    "Mn": 1.75, "Fe": 1.80,
    "Co": 1.84, "Ni": 1.88,
    "Cu": 1.85, "Zn": 1.588,
    "Ga": 1.756, "Ge": 1.994,
    "As": 2.211, "Se": 2.424,
    "Br": 2.685, "Kr": 2.966,
    "Rb": 0.706, "Sr": 0.963,
    "Y": 1.12, "Zr": 1.32,
    "Nb": 1.41, "Mo": 1.47,
    "Tc": 1.51, "Ru": 1.54,
    "Rh": 1.56, "Pd": 1.58,
    "Ag": 1.87, "Cd": 1.521,
    "In": 1.656, "Sn": 1.824,
    "Sb": 1.984, "Te": 2.158,
    "I": 2.359, "Xe": 2.582,
    "Cs": 0.659, "Ba": 0.881,
    "Lu": 1.09, "Hf": 1.16,
    "Ta": 1.34, "W": 1.47,
    "Re": 1.60, "Os": 1.65,
    "Ir": 1.68, "Pt": 1.72,
    "Au": 1.92, "Hg": 1.765,
    "Tl": 1.789, "Pb": 1.854,
    "Bi": 2.01, "Po": 2.19,
    "At": 2.39, "Rn": 2.60,
    "Fr": 0.67, "Ra": 0.89
}


def envien(atom: Chem.Atom, rdmol: Union[Chem.Mol, Chem.RWMol]) -> float:
    bs = []
    nei_en = []
    nei_bias = []
    for nei in atom.GetNeighbors():
        nei_en.append(en[nei.GetSymbol()])
        bs.append(rdmol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetBondTypeAsDouble())
        bias = 0
        for nnei in nei.GetNeighbors():
            if nnei.GetIdx() == atom.GetIdx():
                continue
            bias += 0.1 * en[nnei.GetSymbol()]
        nei_en[-1] += bias
        # print(bias)
    bs = np.array(bs)
    nei_en = np.array(nei_en)
    z = bs.sum()
    if z == 0:
        return 0
    bs = bs / z

    return (bs * nei_en).sum()


def getneimasssum(atom: Chem.Atom) -> float:
    s = 0
    for nei in atom.GetNeighbors():
        s += nei.GetMass()
    return s


def mol2torch_graph(molecule: Union[Chem.Mol, Chem.RWMol]) -> tuple[Data, Data, Data]:
    ComputeGasteigerCharges(molecule, nIter=120)
    g = nx.Graph()
    nds = set()
    for idx, bond in enumerate(molecule.GetBonds()):
        ai = bond.GetBeginAtom()
        aj = bond.GetEndAtom()
        gei = float(ai.GetProp('_GasteigerCharge'))
        if math.isnan(gei):
            gei = 0
        gej = float(aj.GetProp('_GasteigerCharge'))
        if math.isnan(gej):
            gej = 0
        eni = en[ai.GetSymbol()]
        enj = en[aj.GetSymbol()]
        neni = float(envien(ai, molecule))
        nenj = float(envien(aj, molecule))
        hi = ai.GetHybridization()
        hj = aj.GetHybridization()
        if ai.GetIdx() not in nds:
            nds.add(ai.GetIdx())
            xf = np.array([100 * gei, ai.GetAtomicNum(), int(ai.GetIsAromatic()) * 10, int(ai.IsInRing()) * 10,
                           ai.GetFormalCharge() * 5, 2 * hi.real + hi.imag, ai.GetExplicitValence() * 5,
                           getneimasssum(ai), eni * 5, neni * 5], dtype=float)
            g.add_node(ai.GetIdx(), x_f=xf, orig_idx=ai.GetIdx())
        if aj.GetIdx() not in nds:
            nds.add(aj.GetIdx())
            xf = np.array([100 * gej, aj.GetAtomicNum(), int(aj.GetIsAromatic()) * 10, int(aj.IsInRing()) * 10,
                           aj.GetFormalCharge() * 5, 2 * hj.real + hj.imag, aj.GetExplicitValence() * 5,
                           getneimasssum(aj), enj * 5, nenj * 5], dtype=float)
            g.add_node(aj.GetIdx(), x_f=xf, orig_idx=aj.GetIdx())
        if not bond.GetBondTypeAsDouble():
            break_flag = 1
            continue
        else:
            break_flag = 0
        cc = GetUFFBondStretchParams(molecule, ai.GetIdx(), aj.GetIdx())
        if cc is None:
            k, bl = 10000, 0.25
        else:
            k, bl = cc
        g.add_edge(ai.GetIdx(), aj.GetIdx(), bo=torch.tensor(np.asarray([bond.GetBondTypeAsDouble(), bl]), dtype=float),
                   bidx=idx)
    # evaluate
    data_molg = from_networkx(g)
    #for i in g.nodes:
    #    an_ = data_molg.x_f[i][1]
    #    if an_ != molecule.GetAtomWithIdx(i).GetAtomicNum():
    #        logger.warning(f"In mol2torch_graph, atom {i} with {molecule.GetAtomWithIdx(i).GetSymbol()} assigned atomic number {an_} "
    #                       f"not equal to rdkit atomic number {molecule.GetAtomWithIdx(i).GetAtomicNum()}. Use rdkit atomic number.")
            #an_ = molecule.GetAtomWithIdx(i).GetAtomicNum()
    #        raise

    aidx = 0
    didx = 0
    bond_g = nx.Graph()
    ang_g = nx.Graph()
    bond_set = set()
    ang_set = set()
    for i, j in g.edges:
        idx = g.edges[(i, j)]['bidx']
        xfi = g.nodes[i]['x_f']
        xfj = g.nodes[j]['x_f']
        if idx not in bond_set:
            bond_set.add(idx)
            bond_g.add_node(idx, b_f=np.concatenate((xfi, xfj), axis=0), bead_idx=(i, j))
        ang_nij = set()
        for n in g.neighbors(i):
            if n == j:
                continue
            xfn = g.nodes[n]['x_f']
            nidx = g.edges[(i, n)]['bidx']
            if nidx not in bond_set:
                bond_set.add(nidx)
                bond_g.add_node(nidx, b_f=np.concatenate((xfi, xfn), axis=0), bead_idx=(i, n))
            cc = GetUFFAngleBendParams(molecule, n, i, j)
            if cc is None:
                k0, an = 1.5, 109.5
            else:
                k0, an = cc
            # bond_g.add_edge(idx,nidx, ao=torch.tensor([round(k0,3),an], dtype=float), aidx=aidx,idx=(n,i,j))
            if (n, i, j) not in ang_set and (j, i, n) not in ang_set:
                bond_g.add_edge(idx, nidx, ao=torch.tensor(np.asarray([round(k0, 3), an]), dtype=float), aidx=aidx,
                                idx=(n, i, j))
                ang_set.add((n, i, j))
                ang_g.add_node(aidx, a_f=np.concatenate((xfn, xfi, xfj)), bead_idx=(n, i, j))
                ang_nij.add(aidx)
                aidx += 1
        ang_nji = set()
        for n in g.neighbors(j):
            if n == i:
                continue
            xfn = g.nodes[n]['x_f']
            nidx = g.edges[(j, n)]['bidx']
            if nidx not in bond_set:
                bond_set.add(nidx)
                bond_g.add_node(nidx, b_f=np.concatenate((xfj, xfn), axis=0), bead_idx=(j, n))
            cc = GetUFFAngleBendParams(molecule, n, j, i)
            if cc is None:
                k0, an = 1.5, 109.5
            else:
                k0, an = cc
            # bond_g.add_edge(idx,nidx, ao=torch.tensor([round(k0,3),an], dtype=float), aidx=aidx,idx=(n,j,i))
            if (n, j, i) not in ang_set and (i, j, n) not in ang_set:
                bond_g.add_edge(idx, nidx, ao=torch.tensor(np.asarray([round(k0, 3), an]), dtype=float), aidx=aidx,
                                idx=(n, j, i))
                ang_set.add((n, j, i))
                ang_g.add_node(aidx, a_f=np.concatenate((xfn, xfj, xfi)), bead_idx=(n, j, i))
                ang_nji.add(aidx)
                aidx += 1
    beadidx_aidx_hash = {}
    for n in ang_g.nodes:
        i, j, k = ang_g.nodes[n]['bead_idx']
        beadidx_aidx_hash[(i, j, k)] = n
        beadidx_aidx_hash[(k, j, i)] = n
    didx = 0
    for i, j in g.edges:
        for ni in g.neighbors(i):
            if ni == j:
                continue
            for nj in g.neighbors(j):
                if nj == i:
                    continue
                nij_idx = beadidx_aidx_hash[(ni, i, j)]
                nji_idx = beadidx_aidx_hash[(nj, j, i)]
                k0 = GetUFFTorsionParams(molecule, ni, i, j, nj)
                if k0 is None:
                    k0 = 1.5
                ang_g.add_edge(nij_idx, nji_idx, do=round(k0, 3), didx=didx, idx=(ni, i, j, nj))
                didx += 1
    fuck_flag = False
    idx_di = set()
    for e in ang_g.edges:
        ni, i, j, nj = ang_g.edges[e]['idx']
        # WTF is this?
        #if (13072, 13073, 13065, 28224) == (ni, i, j, nj) or (13072, 13073, 13065, 28224) == (nj, j, i, ni):
        #    print('fucking insane')
        idx_di.add((ni, i, j, nj))
        idx_di.add((nj, j, i, ni))
    for b in molecule.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # for ni in g.neighbors(i):
        for nbri in molecule.GetAtomWithIdx(i).GetNeighbors():
            ni = nbri.GetIdx()
            if ni == j:
                continue
            # for nj in g.neighbors(j):
            for nbrj in molecule.GetAtomWithIdx(j).GetNeighbors():
                nj = nbrj.GetIdx()
                if nj == i:
                    continue
                if (ni, i, j, nj) not in idx_di:
                    # print(ni, i, j, nj)
                    fuck_flag = True
    idx_an = set()
    for n in ang_g.nodes:
        ni, i, nj = ang_g.nodes[n]['bead_idx']
        idx_an.add((ni, i, nj))
        idx_an.add((nj, i, ni))
    for i in g.nodes:
        if g.degree(i) >= 2:
            neis = g.neighbors(i)
            for ni in neis:
                for nj in neis:
                    if ni == nj:
                        continue
                    if (ni, i, nj) not in idx_an:
                        # print(ni, i, nj)
                        fuck_flag = True
    if fuck_flag:
        raise 'fuckkq'
    return from_networkx(g), from_networkx(bond_g), from_networkx(ang_g)


class OplsMlRule(CustomRule):
    def __init__(self, atom_model: callable, charge_model: callable,
                 bond_model: callable, angle_model: callable,
                 dihedral_model: callable, improper_model:callable,
                 molecule: Union[Chem.Mol, Chem.RWMol, None] = None, name="ML4ALL"):
        r"""
        This is an example that the ML method can output all parameters at once.
        Since in the opls_engine, parameters are found per-atom, bond, angle, dihedral and improper.
        Use the model-pre-process to get all parameters at once and in process() function,
        take out the parameters by query. The bonded parameters, i.e., torsions, was give as dict[(i,j,k,l), parameters]
        therefore, the query function below finds self.dihedrals[query] or dihedrals[query[::-1]]
        :param *_model: a callable object, returns all parameters at once.
        :param molecule: the molecule given in rdkit molecule
        :param name: the name of the rule
        """
        super().__init__(name)
        self.charge_model = charge_model
        self.atom_model = atom_model
        self.bond_model = bond_model
        self.angle_model = angle_model
        self.dihedral_model = dihedral_model
        self.improper_model = improper_model
        self.molecule = molecule
        self.atoms = None
        self.bonds = None
        self.angles = None
        self.dihedrals = None
        self.charge = None
        if self.molecule is not None:
            logger.info("Initialize the ML model, generating parameters...")
            self.preprocess()
        else:
            logger.warning("No molecule given, the preprocessing will be skipped, and initialized in first call.")

    def set_molecule(self, molecule: Union[Chem.Mol, Chem.RWMol]):
        self.molecule = molecule
        logger.info(f"Set molecule to {self.molecule}, preprocesing...")
        self.preprocess()

    def preprocess(self):
        if self.molecule is None:
            logger.error("Molecule not set, plase use set_molecule() method.")
            return
        # Here are the self-defined method for pre-processing the molecule
        mol_graph, bond_graph, ang_graph = mol2torch_graph(self.molecule)
        self.charge = self.charge_model(mol_graph)
        self.atoms = self.atom_model(mol_graph)
        self.bonds = self.bond_model(mol_graph, self.molecule)
        #print(self.bonds.keys())
        self.angles = self.angle_model(bond_graph)
        self.dihedrals = self.dihedral_model(ang_graph)
        self.impropers = self.improper_model(mol_graph)
        logger.info(f"ML parameters generated with {len(self.atoms)} atoms, {len(self.charge)} charges,"
                    f" {len(self.bonds)} bonds, {len(self.angles)} angles and {len(self.dihedrals)} dihedrals. "
                    f"The molecule contains {self.molecule.GetNumAtoms()} atoms, {self.molecule.GetNumBonds()} bonds")

    def process(self, molecule: Union[Chem.Mol, Chem.RWMol], query: Union[int, tuple[int], tuple[int,str]]) -> Any:
        r"""
        :param molecule: The target molecule, should be as same as self.molecule.
        :param query: query (molecule, atom_idx) for atom, (idx, jdx) for bond, etc.
        :return: opls_atom, bond, angle, dihedral, improper.
        """
        if self.molecule is None:
            self.set_molecule(molecule)
        if isinstance(query, int):
            _atom = self.atoms[query]
            #print(_atom[0], _atom[1], _atom)
            _charge = self.charge[query]
            _rd_atom = self.molecule.GetAtomWithIdx(query)
            _rd_atom1 = molecule.GetAtomWithIdx(query)
            assert _rd_atom.GetAtomicNum() == _rd_atom1.GetAtomicNum(), ("The molecule used to initiate"
                                                                         "ML and applied is not same!")
            _symbol = _rd_atom.GetSymbol()
            _mass = _rd_atom.GetMass()
            _atomic_num = _rd_atom.GetAtomicNum()
            #print(_atom[0], _atom[1], _charge)
            an_ = nb_an[_atom]
            #if an_ != _atomic_num:
            #    logger.warning(f"In custom finder {self.name}, atom {query} with {_symbol} assigned atomic number {an_} "
            #                   f"not equal to rdkit atomic number {_atomic_num}. Use rdkit atomic number.")
                #an_ = _atomic_num
            #    raise
            return OplsAtom(name=f"{_symbol}_ML", bond_type=f"{_symbol}_ML", smarts=_symbol,
                            element=_symbol, hash=_symbol, charge=_charge, epsilon=_atom[0], sigma=_atom[1], ptype='A',
                            mass=_mass, atomic_num=_atomic_num, desc='ML model')
        elif len(query) == 2 and query[1] == 'imp':  # for improper
            query = query[0]
            _imp = self.impropers[query]
            #_atom = self.atoms[query]
            #_charge = self.charge[query]
            _rd_atom = self.molecule.GetAtomWithIdx(query)
            _rd_atom1 = molecule.GetAtomWithIdx(query)
            assert _rd_atom.GetAtomicNum() == _rd_atom1.GetAtomicNum(), ("The molecule used to initiate"
                                                                         "ML and applied is not same!")
            _symbol = _rd_atom.GetSymbol()
            _mass = _rd_atom.GetMass()
            _atomic_num = _rd_atom.GetAtomicNum()
            #print(_atom[0], _atom[1], _charge)
            name = f'{_symbol}-SP2-imp'
            return OplsBonded(name=name, hash=name, ftype=4, param=f'[{_imp[1]}, {_imp[2]}, {_imp[3]}]',
                              type='bond', idx=None, is_rule=False)
        elif len(query) == 2:  # for bonds
            i, j = query
            _rd_atom_i = molecule.GetAtomWithIdx(i)
            _rd_atom_j = molecule.GetAtomWithIdx(j)
            _rd_atom_i1 = self.molecule.GetAtomWithIdx(_rd_atom_i.GetIdx())
            _rd_atom_j1 = self.molecule.GetAtomWithIdx(_rd_atom_j.GetIdx())
            assert _rd_atom_i.GetAtomicNum() == _rd_atom_i1.GetAtomicNum() and \
                   _rd_atom_j.GetAtomicNum() == _rd_atom_j1.GetAtomicNum(), \
                ("The molecule used to initiate"
                 "ML and applied is not same!")
            name = f'{molecule.GetAtomWithIdx(i).GetSymbol()}-{molecule.GetAtomWithIdx(j).GetSymbol()}'
            if self.bonds.get((i, j)) is not None:
                bp = self.bonds[(i, j)]
            elif self.bonds.get((j, i)) is not None:
                bp = self.bonds[(j, i)]
            else:
                logger.error(f"In custom finder {self.name}, bond {query} with {name} not found.")
                return
            return OplsBonded(name=name, hash=name, ftype=1, param=f'"[{bp[1]}, {bp[2]}]"',
                              type='bond', idx=None, is_rule=False)
        elif len(query) == 3:  # For atom and bonds are already checked.
            i, j, k = query
            _rd_atom_i = molecule.GetAtomWithIdx(i)
            _rd_atom_j = molecule.GetAtomWithIdx(j)
            _rd_atom_k = molecule.GetAtomWithIdx(k)
            name = f'{_rd_atom_i.GetSymbol()}-{_rd_atom_j.GetSymbol()}-{_rd_atom_k.GetSymbol()}'
            ap = self.angles.get((i, j, k)) or self.angles.get((k, j, i))
            if ap is None:
                logger.error(f"In custom finder {self.name}, angle {query} with {name} not found.")
                return
            return OplsBonded(name=name, hash=name, ftype=1, param=f'"[{ap[2]}, {ap[1]}]"', type='angle', idx=None,
                              is_rule=False), query

        else:  # the len(query) has to be 4 now
            ni, i, j, nj = query
            _rd_atom_i = molecule.GetAtomWithIdx(ni)
            _rd_atom_j = molecule.GetAtomWithIdx(i)
            _rd_atom_k = molecule.GetAtomWithIdx(j)
            _rd_atom_l = molecule.GetAtomWithIdx(nj)
            name = f'{_rd_atom_i.GetSymbol()}-{_rd_atom_j.GetSymbol()}-{_rd_atom_k.GetSymbol()}-{_rd_atom_l.GetSymbol()}'
            dp = self.dihedrals.get((ni, i, j, nj)) or self.dihedrals.get((nj, j, i, ni))
            if dp is None:
                logger.error(f"In custom finder {self.name}, dihedral {query} with {name} not found.")
                return

            return OplsBonded(name=name, hash=name, ftype=3,
                              param=f'"[{dp[1]:.3f}, {dp[2]:.3f}, {dp[3]:.3f}, {dp[4]:.3f}, {dp[5]:.3f}, {dp[6]:.3f}]"',
                              type='dihedral', idx=(ni, i, j, nj), is_rule=False), query


from domd_forcefield.oplsaa.ml_functions.models import mlnonbond, mlbond, mlangle, mlcharge, mldihedral,mlimproper

MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral,mlimproper)

if __name__ == '__main__':
    from rdkit import Chem
    MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral,mlimproper)
    mol = Chem.MolFromSmiles('C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F')
    #mol = Chem.MolFromSmiles('[Li+]')
    mol = Chem.AddHs(mol)
    c = []
    n_atom = mol.GetNumAtoms()
    for a in mol.GetAtoms():
        #print(MLModel(mol, a.GetIdx()), a.GetFormalCharge(), a.GetSymbol(), a.GetIdx())
        c.append(MLModel(mol, a.GetIdx()).charge)
    c = np.array(c)
    print(sum(c))
    #print(c - (sum(c)+1)/n_atom)
    #print(MLModel(mol, (2, 3)))
    #MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral)
    #mol = Chem.MolFromSmiles('CCOCOCC(F)(F)F')
    #print(MLModel(mol, (2, 3)))
