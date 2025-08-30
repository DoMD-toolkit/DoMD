from typing import Union

import networkx as nx
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from misc.logger import logger


class Position(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeConf():
    def __init__(self, num_atoms):
        self.x = {}

    def set_pos(self, i, pos):
        self.x[i] = pos

    def GetAtomPosition(self, idx):
        return self.x.get(idx)

    def GetPositions(self):
        ret = np.zeros((len(self.x), 3))
        for k in self.x:
            ret[k, 0] = self.x[k].x
            ret[k, 1] = self.x[k].y
            ret[k, 2] = self.x[k].z
        return ret

    def SetAtomPosition(self, idx, p):
        if isinstance(p, np.ndarray):
            self.x[idx] = Position(*p)
        else:
            self.x[idx] = p


def generate_pos_res(molecule: Union[Chem.Mol, Chem.RWMol], cg_mol: nx.Graph) -> FakeConf | None:
    r"""May break the chirality while dividing into fragments, e.g., B is chiral center of AABA
    but not in ABA.
    """
    logger.warning("Generating positions for molecule via fragments, note that the chirality may be broken.")
    conf = FakeConf(molecule.GetNumAtoms())
    atom_map = {}
    for atom in molecule.GetAtoms():
        res_id = atom.GetIntProp('global_res_id')
        if not atom_map.get(res_id):
            atom_map[res_id] = []
        atom_map[res_id].append(atom.GetIdx())
    adj_dict = dict(cg_mol.adjacency())
    for m_id in tqdm(cg_mol.nodes, total=len(cg_mol.nodes), desc='generating pos fragmently'):
        # generate position of A by its neighbor monomers
        # to obtain better monomer-monomer connections
        n_ids = list(adj_dict[m_id].keys())
        fragment = Chem.RWMol()
        fragment_ids = [m_id] + n_ids
        bonds = set()
        local_map1 = {}
        local_map2 = {}
        count = 0
        broke = set()
        for res_id in fragment_ids:  # Get current monomer
            atoms = atom_map.get(res_id)
            if atoms is not None:
                for atom_id in atoms:
                    local_map1[atom_id] = count
                    local_map2[count] = atom_id  # map monomer to polymer
                    count += 1
        for i in range(count):
            atom_id = local_map2[i]
            atom = molecule.GetAtomWithIdx(atom_id)
            fragment.AddAtom(atom)
            for bond in atom.GetBonds():
                a, b, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
                if molecule.GetAtomWithIdx(a).GetIntProp("global_res_id") not in fragment_ids:
                    # print(f"Atom {a} with residue idx {molecule.GetAtomWithIdx(a).GetIntProp('global_res_id')} not in fragment {fragment_ids}")
                    broke.add(atom_id)
                    continue
                if molecule.GetAtomWithIdx(b).GetIntProp("global_res_id") not in fragment_ids:
                    # print(f"Atom {b} with residue idx {molecule.GetAtomWithIdx(b).GetIntProp('global_res_id')} not in fragment {fragment_ids}")
                    broke.add(atom_id)
                    continue
                if a < b:
                    bonds.add((a, b, t))
                else:
                    bonds.add((b, a, t))
        # logger.warning(f'The broken atoms:{broke}')
        for bond in bonds:
            fragment.AddBond(local_map1[bond[0]], local_map1[bond[1]], bond[2])

        # print(Chem.MolToSmiles(fragment))
        # _rma = []
        # for atom_ in fragment.GetAtoms():
        #    if local_map2.get(atom_.GetIdx()) is None:
        #        continue
        #    m_a_id = local_map2.get(atom_.GetIdx())
        #    atom__ = molecule.GetAtomWithIdx(m_a_id)
        #    if atom__.GetIntProp("global_res_id") != m_id and atom__.GetSymbol() == 'H':
        #        _rma.append(atom_.GetIdx())
        # for idx in sorted(_rma, reverse=True):
        #    fragment.RemoveAtom(idx)
        # print(Chem.MolToSmiles(fragment))

        chiral_tag_center = {}
        for atom in fragment.GetAtoms():  # avoiding breaking aromatic rings
            if atom.GetIntProp("global_res_id") == m_id:
                chiral_tag_center[atom.GetIdx()] = atom.GetChiralTag()  # only care about the target monomer
            if atom.GetIsAromatic():
                if not atom.IsInRing():
                    atom.SetIsAromatic(0)
            if local_map2.get(atom.GetIdx()) in broke:
                # if atom.GetIsAromatic() and atom.IsInRing():
                #    continue
                atom.SetIsAromatic(0)
                for btom in atom.GetNeighbors():
                    # logger.info(f'Neighbors of broken atom {atom.GetIdx()} : {btom.GetIdx()}')
                    if btom.GetIsAromatic() and btom.IsInRing():
                        continue
                    btom.SetIsAromatic(0)
                    _bond = fragment.GetBondBetweenAtoms(atom.GetIdx(), btom.GetIdx())
                    _bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        # print(Chem.MolToSmiles(fragment), '------')
        # fragment = AllChem.AddHs(fragment)
        Chem.SanitizeMol(fragment, Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
        res = Chem.SanitizeMol(fragment, catchErrors=True)

        if not res is Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            logger.warning(f'Sanitize failed on: {Chem.MolToSmiles(fragment)}')
            # fragment = Chem.MolFromSmiles(Chem.MolToSmiles(fragment))
            # fragment = Chem.MolFromPDBBlock(Chem.MolToPDBBlock(fragment, flavor=4))
        # I don't know why yet
        # but not important, for the truncated monomers are not used
        # raise ValueError(f"{res}, {Chem.MolToSmiles(fragment)}")
        _mh = AllChem.AddHs(fragment)
        # print(Chem.MolToSmiles(_mh),'2222222')
        # print(Chem.MolToSmiles(_mh))
        for atom in _mh.GetAtoms():
            if not chiral_tag_center.get(atom.GetIdx()) is None:
                atom.SetChiralTag(chiral_tag_center[atom.GetIdx()])
            else:
                atom.SetChiralTag(rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                # other monomers will be set to unspecified
        conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=1000, useRandomCoords=False)
        if conf_id != -1 and res is not Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            logger.warning(f'Sanitize failed on: {Chem.MolToSmiles(fragment)} with H: {Chem.MolToSmiles(_mh)}')
        # if not res is Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        # Chem.MolToPDBFile(_mh, 'a0.pdb', flavor=4)
        # AllChem.UFFOptimizeMolecule(_mh)
        if conf_id == -1:
            _mh_metadata = {'atoms': {}, 'bonds': set()}
            for bond in _mh.GetBonds():
                a = bond.GetBeginAtom()
                b = bond.GetEndAtom()
                _mh_metadata['atoms'][a.GetIdx()] = a.GetSymbol()
                _mh_metadata['atoms'][a.GetIdx()] = a.GetSymbol()
                _mh_metadata['bonds'].add((a.GetIdx(), b.GetIdx()))
            # pickle.dump(_mh_metadata,open('/home/lmy/HTSP/FPSG/fuckmol_meta.pkl','wb'))
            # pickle.dump((_mh,fragment),open('/home/lmy/HTSP/FPSG/fuckmol.pkl','wb'))
            logger.error(
                f"Residue generation with conversation of chirality failed!\n{Chem.MolToSmiles(_mh, isomericSmiles=True)}")
            logger.error(f"The chirality of target monomer is {chiral_tag_center}, please make sure that the cut"
                         f"off method does not break chirality of molecule!")
            logger.error(
                f"This error would not stop the generation program but use a random initial coordinate for bead."
                f"This may contribute to wrong conformation breaking chirality.")
            conf_id = AllChem.EmbedMolecule(_mh, maxAttempts=1000, useRandomCoords=True)
            if conf_id == -1:
                logger.error(
                    f"Residue generation without conversation of chirality failed! Please check your reaction template, make sure"
                    f"get the right molecular in SMARTS")
                return
            AllChem.UFFOptimizeMolecule(_mh)
        _conf = _mh.GetConformer(conf_id)
        for atom in _mh.GetAtoms():
            if local_map2.get(atom.GetIdx()) is None:
                continue
            m_a_id = local_map2.get(atom.GetIdx())
            p = _conf.GetAtomPosition(atom.GetIdx())
            if molecule.GetAtomWithIdx(m_a_id).GetIntProp("global_res_id") != m_id:
                continue
            logger.debug(f'{atom.GetSymbol()}, {molecule.GetAtomWithIdx(m_a_id).GetSymbol()}')
            conf.set_pos(m_a_id, Position(p.x, p.y, p.z))
    return conf


def embd(molecule: Union[Chem.Mol, Chem.RWMol], cg_molecule: nx.Graph, large: int = 500, custom_conf=None):
    if custom_conf is not None:
        molecule.AddConformer(custom_conf)
        return molecule.GetConformer(0)
    if molecule.GetNumConformers() > 0:
        logger.warning(f"The molecule has {molecule.GetNumConformers()} conformers, return the 0th.")
        return molecule.GetConformer(0)
    if molecule.GetNumAtoms() > large:
        logger.warning(f"Num of atoms {molecule.GetNumAtoms()} is greater than {large}, generating by residue.")
        logger.warning(f"The residue method may break rings so that break the planar structure of molecule."
                       f" Also broken chirality may cause generation failure.")
        conf = generate_pos_res(molecule, cg_molecule)
        #print(conf.GetPositions())
        if (conf is None) or (len(conf.x) != molecule.GetNumAtoms()):
            logger.error(f"Configuration generation error.")
            return
    else:
        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True)
        if conf_id == -1:
            conf = generate_pos_res(molecule, cg_molecule)
            if len(conf.x) != molecule.GetNumAtoms():
                logger.warning(f"Configuration generation error! {len(conf.x)} != {molecule.GetNumAtoms()}")
        else:
            conf = molecule.GetConformer(conf_id)
    #print(conf.GetPositions())
    return conf
