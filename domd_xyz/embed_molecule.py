from typing import Union, Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from domd_xyz.embed import embd, Meta, optimize_res_orientation
from misc.logger import logger


def embed_molecule(molecule: Union[Chem.Mol, Chem.RWMol],
                   cg_molecule: Union[nx.Graph, None] = None,
                   box=None, large=500, chunk_per_d=1) -> Any:
    _no_res_id = False
    _residue_map = {}
    ret = None
    for atom in molecule.GetAtoms():
        if 'res_id' not in atom.GetPropNames():
            _no_res_id = True
            logger.warning(f"No res_id property found in atom {atom.GetIdx()}, {atom.GetSymbol()} "
                           "the whole molecule is considered as one residue.")
    if _no_res_id:
        for atom in molecule.GetAtoms():
            atom.SetIntProp("res_id", 0)

    for atom in molecule.GetAtoms():
        if not _residue_map.get(atom.GetIntProp("res_id")):
            _residue_map[atom.GetIntProp("res_id")] = set()
        _residue_map[atom.GetIntProp("res_id")].add(atom.GetIdx())

    if _no_res_id:
        logger.warning("The entire molecule is considered "
                       "as one residue, for large molecules "
                       "this may cause problems.")
        if len(cg_molecule) > 1:
            logger.error("The ref CG molecule contains more than 1"
                         " residues but the whole aa molecule is considered as one residue")
        return

    if cg_molecule is None:
        logger.warning("There is no ref CG info, so that the whole molecule is generated at once."
                       " This may be slow or fail for large molecule (e.g., >500)")

        conf_id = AllChem.EmbedMolecule(molecule, useRandomCoords=True)
        if conf_id == -1:
            logger.error("Configuration generation failed!")
        ret = molecule.GetConformer(conf_id)
    else:
        if len(_residue_map) != len(cg_molecule):
            logger.error("The number of residue in aa molecule is not equal to cg molecule "
                         f"{len(_residue_map)} != {len(cg_molecule)}")
            return
        logger.info('Embed molecule by using ETKDG ...')
        conf = embd(molecule, cg_molecule, large=large, custom_conf=None)
        #print(conf.GetPositions(),'***** after embd *****')
        logger.info('Embed finished.')
        if conf is None:
            return
        # make residue centered at 0
        for res_id in _residue_map:
            atom_ids = _residue_map[res_id]
            com = np.zeros(3)
            sam = 0
            for atom_id in atom_ids:
                mas = molecule.GetAtomWithIdx(atom_id).GetMass()
                pos = conf.GetAtomPosition(atom_id)
                com += np.array([pos.x, pos.y, pos.z]) * mas
                sam += mas
            for atom_id in atom_ids:
                pos = conf.GetAtomPosition(atom_id)
                conf.SetAtomPosition(atom_id, np.array([pos.x, pos.y, pos.z]) - com / sam)
            # debug
            if logger.level <= 10:
                com = np.zeros(3)
                sam = 0
                for atom_id in atom_ids:
                    mas = molecule.GetAtomWithIdx(atom_id).GetMass()
                    pos = conf.GetAtomPosition(atom_id)
                    com += np.array([pos.x, pos.y, pos.z]) * mas
                    sam += mas
                logger.debug(f"The com of residue {res_id} is {com / sam}.")
        # debug
        if logger.level <= 10:
            for res_id in _residue_map:
                debug_xyz = ""
                debug_xyz += f"{len(_residue_map[res_id])}\n\n"
                for atom_id in _residue_map[res_id]:
                    p = conf.GetAtomPosition(atom_id)
                    atom = molecule.GetAtomWithIdx(atom_id)
                    debug_xyz += f"{atom.GetSymbol()} {p.x} {p.y} {p.z}\n"
                logger.debug(f"structure of res {res_id}:\n{debug_xyz}")

        atom_pos = conf.GetPositions()
        atom_res_id = np.array([a.GetIntProp("res_id") for a in molecule.GetAtoms()])
        n_residue = len(_residue_map)
        bonds = []
        trans = np.zeros((n_residue, 3))
        local_frame_idx = []
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            atom_i = molecule.GetAtomWithIdx(i)
            atom_j = molecule.GetAtomWithIdx(j)
            res_id_i = atom_i.GetIntProp("res_id")
            res_id_j = atom_j.GetIntProp("res_id")
            # print(res_id_i,i,'*',res_id_j,j)
            if res_id_i != res_id_j:
                bonds.append((res_id_i, res_id_j))
                local_frame_idx.append((i, j))
                # print(i,j)
        for res_id in _residue_map:
            for node in cg_molecule.nodes:
                if cg_molecule.nodes[node].get('local_res_id') == res_id:
                    trans[res_id] = cg_molecule.nodes[node].get("x")
        if box is None:
            box = np.ones(3) * abs(conf.GetPositions().max()) * 100.0
            logger.info(f"Box is not given. Set to {box} to eliminate pbc.")
        meta = Meta(np.array(bonds, dtype=np.int64),
                    trans,
                    np.array(local_frame_idx, dtype=np.int64),
                    atom_pos,
                    atom_res_id,
                    box)
        # pickle.dump(meta,open('meta.pkl','wb'))
        logger.info("Optimizing orientations...")
        rot = optimize_res_orientation(n_residue, meta, chunk_per_d=chunk_per_d)
        # rot = np.asarray([np.eye(3),] * n_residue)
        logger.info(f"Optimize finished.")
        # debug
        for ir, r in enumerate(rot):
            logger.debug(f"Rotation for residue {ir} is {r} and r.T.dot(r) is {r.T.dot(r)}")
        for res_id in _residue_map:
            atoms = _residue_map[res_id]
            for atom_id in atoms:
                p = rot[res_id].dot(atom_pos[atom_id])
                # p = conf.GetAtomPosition(atom_id)
                conf.SetAtomPosition(atom_id, p + trans[res_id])
        #print(conf.GetPositions(),'**** after optimization ****')
        ret = Chem.Conformer()
        for atom in molecule.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            p = np.array([pos.x, pos.y, pos.z])
            ret.SetAtomPosition(atom.GetIdx(), p)
        #print(ret.GetPositions(), '**** man make conf ****')
    #print(ret.GetPositions())
    return ret

