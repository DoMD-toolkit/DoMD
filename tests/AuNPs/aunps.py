import sys
from domd_cgbuilder._conf_gen import embed_CG_system
from domd_cgbuilder.cg_mol import CGMol
from misc.cg_system import read_cg_topology
from misc.io.xml_writer import write_xml
from domd_cgbuilder.cg_ff import cg_info, GetBeadinfo, GetBondedtype
import networkx as nx
import numpy as np
import math
import time
from misc.io.xml_reader import XmlParser
import os
from domd_topology.reactor import Reactor
from domd_topology.functions import set_molecule_id_for_h
from rdkit import Chem
from rdkit.Chem import AllChem
from domd_xyz.embed_molecule import embed_molecule
from domd_forcefield.oplsaa.opls import OplsFF
from domd_forcefield.oplsaa.opls_db import opls_db
from domd_forcefield.oplsaa.ml import OplsMlRule
from domd_forcefield.oplsaa.ml_functions.models import mlnonbond, mlbond, mlangle, mlcharge, mldihedral, mlimproper
from misc.io.assemble import assemble_opls
from misc.io.xml_writer import write_xml_opls
from misc.io.gmx_writer import write_gro_opls

start = time.time()
PEO = 'CCOF'
Au = '[Au]'
LYS1 = 'NCCCC[C@@H](N)C(=O)O'
LYS2 = 'NCCCC[C@@H](N)C(=O)O'
LYS3 = 'NCCCC[C@@H](N)C(=O)O'
LYS4 = 'NCCCC[C@@H](N)C(=O)O'
LYS5 = 'NCCCC[C@@H](N)C(=O)O'
LYS6 = 'NCCCC[C@@H](N)C(=O)O'
LYS7 = 'NCCCC[C@@H](N)C(=O)O'
VAL1 = 'CC(C)[C@@H](N)C(=O)O'
VAL2 = 'CC(C)[C@@H](N)C(=O)O'
TRP1 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
TRP2 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
TRP4 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
TRP5 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
TRP6 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
TRP7 = 'c1cccc2c(C[C@@H](N)C(=O)O)cnc2c1'
ALA = 'C[C@@H](N)C(=O)O'
LEU = 'CC(C)C[C@@H](N)C(=O)O'
W = 'O'
NC3 = 'C[N+](C)(C)C'
TAP = 'C[N+](C)C'
PO4 = 'OP(=O)([O-])OC'
GL1 = 'FC[C@H](OC(=O)CC)CO'
GL2 = 'C(=O)CC'
GL0 = 'OCCO'
C1A = 'CCCF'
C2A = 'CCCF'
C3A = 'CCCF'
C4A = 'CCCF'
C5A = 'CCCF'
D2A = 'CCC=CCC'
D2B = 'CCC=CCC'
C1B = 'CCCF'
C2B = 'CCCF'
C3B = 'CCCF'
C4B = 'CCCF'
C5B = 'CCCF'


reaction_template = {
    'a': {
        'cg_reactant_list': [('PEO', 'PEO')],
        'smarts': '[CH3:1].[O:2][F:3]>>[C:1][O:2].[F:3]',
        'prod_idx': [0]
    },
    'b': {
        'cg_reactant_list': [('Au', 'PEO')],
        'smarts': '[Au:1].[O:2][F:3]>>[Au:1][O:2].[F:3]',
        'prod_idx': [0]
    },
    'Au': {
        'cg_reactant_list': [('Au',)],
        'smarts': '[Au:1]>>[Au:1]',
        'prod_idx': [0]
    },
    'NC3-PO4': {
        'cg_reactant_list': [('NC3', 'PO4')],
        'smarts': '[C:1].[C:2]>>[C:1][C:2]',
        'prod_idx': [0]
    },
    'TAP-GL1': {
        'cg_reactant_list': [('TAP', 'GL1')],
        'smarts': '[N:1].[C:2][F:3]>>[N:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'GL0-PO4': {
        'cg_reactant_list': [('GL0', 'PO4')],
        'smarts': '[C:1].[C:2]>>[C:1][C:2]',
        'prod_idx': [0]
    },
    'PO4-GL1': {
        'cg_reactant_list': [('PO4', 'GL1')],
        'smarts': '[O:1].[C:2][F:3]>>[O:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'GL1-GL2': {
        'cg_reactant_list': [('GL1', 'GL2')],
        'smarts': '[OH1:1].[C:2]=[O:3]>>[O:1][C:2]=[O:3]',
        'prod_idx': [0]
    },
    'GL1-C1A': {
        'cg_reactant_list': [('GL1', 'C1A')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'GL2-C1B': {
        'cg_reactant_list': [('GL2', 'C1B')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][C:2].[F:3]',
        'prod_idx': [0]
    },
    'C1A-D2A': {
        'cg_reactant_list': [('C1A', 'D2A')],
        'smarts': '[CH3:1].[CH3:2]>>[CH2:1][CH2:2]',
        'prod_idx': [0]
    },
    'D2A-C3A': {
        'cg_reactant_list': [('D2A', 'C3A')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][CH2:2].[F:3]',
        'prod_idx': [0]
    },
    'C1A-C2A': {
        'cg_reactant_list': [('C1A', 'C2A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C2A-C3A': {
        'cg_reactant_list': [('C2A', 'C3A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C3A-C4A': {
        'cg_reactant_list': [('C3A', 'C4A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C4A-C5A': {
        'cg_reactant_list': [('C4A', 'C5A')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C1B-C2B': {
        'cg_reactant_list': [('C1B', 'C2B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[C:1][C:3].[F:2]',
        'prod_idx': [0]
    },
    'C1B-D2B': {
        'cg_reactant_list': [('C1B', 'D2B')],
        'smarts': '[CH3:1].[CH3:2]>>[CH2:1][CH2:2]',
        'prod_idx': [0]
    },
    'D2B-C3B': {
        'cg_reactant_list': [('D2B', 'C3B')],
        'smarts': '[CH3:1].[C:2][F:3]>>[CH2:1][CH2:2].[F:3]',
        'prod_idx': [0]
    },
    'C2B-C3B': {
        'cg_reactant_list': [('C2B', 'C3B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C3B-C4B': {
        'cg_reactant_list': [('C3B', 'C4B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'C4B-C5B': {
        'cg_reactant_list': [('C4B', 'C5B')],
        'smarts': '[CH3:1].[F:2][C:3]>>[CH2:1][CH2:3].[F:2]',
        'prod_idx': [0]
    },
    'VAL1-TRP1': {
        'cg_reactant_list': [('VAL1', 'TRP1')],
        'smarts': '[NH2:1][C@@H1:2].[C:3](=[O:4])[OH1:5]>>[C:3](=[O:4])[NH1:1][C@@H1:2].[OH2:5]',
        'prod_idx': [0]
    },
    'TRP1-LYS1': {
        'cg_reactant_list': [('TRP1', 'LYS1')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'LYS1-LYS2': {
        'cg_reactant_list': [('LYS1', 'LYS2')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'LYS2-TRP2': {
        'cg_reactant_list': [('LYS2', 'TRP2')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'TRP2-LYS3': {
        'cg_reactant_list': [('TRP2', 'LYS3')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'LYS3-LYS4': {
        'cg_reactant_list': [('LYS3', 'LYS4')],
        'smarts': '[NH2:1][C@@H1:2].[C:3](=[O:4])[OH1:5]>>[C:3](=[O:4])[NH1:1][C@@H1:2].[OH2:5]',
        'prod_idx': [0]
    },
    'LYS4-TRP4': {
        'cg_reactant_list': [('LYS4', 'TRP4')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'TRP4-TRP5': {
        'cg_reactant_list': [('TRP4', 'TRP5')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'TRP5-LYS5': {
        'cg_reactant_list': [('TRP5', 'LYS5')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'LYS5-LYS6': {
        'cg_reactant_list': [('LYS5', 'LYS6')],
        'smarts': '[NH2:1][C@@H1:2].[C:3](=[O:4])[OH1:5]>>[C:3](=[O:4])[NH1:1][C@@H1:2].[OH2:5]',
        'prod_idx': [0]
    },
    'LYS6-TRP6': {
        'cg_reactant_list': [('LYS6', 'TRP6')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'TRP6-TRP7': {
        'cg_reactant_list': [('TRP6', 'TRP7')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'TRP7-LYS7': {
        'cg_reactant_list': [('TRP7', 'LYS7')],
        'smarts': '[NH2:1].[C:2](=[O:3])[OH1:4]>>[C:2](=[O:3])[NH1:1].[OH2:4]',
        'prod_idx': [0]
    },
    'LYS7-VAL2': {
        'cg_reactant_list': [('LYS7', 'VAL2')],
        'smarts': '[NH2:1][C@@H1:2].[C:3](=[O:4])[OH1:5]>>[C:3](=[O:4])[NH1:1][C@@H1:2].[OH2:5]',
        'prod_idx': [0]
    },
    'W': {
        'cg_reactant_list': [('W',)],
        'smarts': '[O:1]>>[O:1]',
        'prod_idx': [0]
    },
}

mols = {
    'PEO': {'smiles': PEO, 'file': None},
    'Au': {'smiles': Au, 'file': None},
    'TAP': {'smiles': TAP, 'file': None},
    'NC3': {'smiles': NC3, 'file': None},
    'PO4': {'smiles': PO4, 'file': None},
    'GL0': {'smiles': GL0, 'file': None},
    'GL1': {'smiles': GL1, 'file': None},
    'GL2': {'smiles': GL2, 'file': None},
    'C1A': {'smiles': C1A, 'file': None},
    'C2A': {'smiles': C2A, 'file': None},
    'C3A': {'smiles': C3A, 'file': None},
    'C4A': {'smiles': C4A, 'file': None},
    'C5A': {'smiles': C5A, 'file': None},
    'C1B': {'smiles': C1B, 'file': None},
    'C2B': {'smiles': C2B, 'file': None},
    'C3B': {'smiles': C3B, 'file': None},
    'C4B': {'smiles': C4B, 'file': None},
    'C5B': {'smiles': C5B, 'file': None},
    'D2A': {'smiles': D2A, 'file': None},
    'D2B': {'smiles': D2B, 'file': None},
    'W': {'smiles': 'O', 'file': None},
    'LYS1': {'smiles': LYS1, 'file': None},
    'LYS2': {'smiles': LYS2, 'file': None},
    'LYS3': {'smiles': LYS3, 'file': None},
    'LYS4': {'smiles': LYS4, 'file': None},
    'LYS5': {'smiles': LYS5, 'file': None},
    'LYS6': {'smiles': LYS6, 'file': None},
    'LYS7': {'smiles': LYS7, 'file': None},
    'TRP1': {'smiles': TRP1, 'file': None},
    'TRP2': {'smiles': TRP2, 'file': None},
    'TRP4': {'smiles': TRP4, 'file': None},
    'TRP5': {'smiles': TRP5, 'file': None},
    'TRP6': {'smiles': TRP6, 'file': None},
    'TRP7': {'smiles': TRP7, 'file': None},
    'VAL1': {'smiles': VAL1, 'file': None},
    'VAL2': {'smiles': VAL2, 'file': None},

}

for key in mols:
    if not mols[key]['file'] is None:
        mol = Chem.RemoveAllHs(Chem.MolFromPDBFile(mols[key]['pdb']))
        mols[key]['smiles'] = Chem.MolToSmiles(mol)

xml = XmlParser(os.path.join('cg','lipidmem_with_AuNP.xml'))
box = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
box = np.array(tuple(map(float, box))[:3]) * 10  # a
cg_sys, cg_mols = read_cg_topology(xml, mols)

reactor = Reactor(mols, reaction_template)

types = xml.data['type']
reactions = []
bonded_au = []
for bond in xml.data['bond']:
    reactions.append((bond[0], bond[1], bond[2]))
    if types[bond[1]] == 'Au':
        bonded_au.append(bond[1])
    elif types[bond[2]] == 'Au':
        bonded_au.append(bond[2])
for i,t in enumerate(xml.data['type']):
    if t == 'W':
        reactions.append(('W',i))
    if t == 'Au' and i not in bonded_au:
        reactions.append((t,i))

aa_mols, meta = reactor.process(cg_mols, reactions)
[Chem.SanitizeMol(_) for _ in aa_mols]
aa_mols_h = [Chem.AddHs(m) for m in aa_mols]
[set_molecule_id_for_h(m) for m in aa_mols_h]

#from multiprocessing import Pool
#def proc(i):
#    global aa_mols_h, cg_mols, box
#    aa_mol = aa_mols_h[i]
#    cg_mol = cg_mols[i]
#    if aa_mol.GetNumAtoms() > 10000:
#        chunk_per_d=4
#    else:
#        chunk_per_d=1
#    conf = embed_molecule(aa_mol, cg_mol, box = box, chunk_per_d=chunk_per_d)
#    return conf
#
#with Pool(processes=4) as pool:
#    # map guarantees that results[i] is worker(i)
#    confs = pool.map(proc, range(len(aa_mols_h)))

confs = []
for i,aa_mol, cg_mol in zip(range(len(aa_mols_h[:])),aa_mols_h[:],cg_mols[:]):
    if aa_mol.GetNumAtoms() > 10000:
        chunk_per_d=4
    else:
        chunk_per_d=1
    conf = embed_molecule(aa_mol, cg_mol, box = box, chunk_per_d=chunk_per_d)
    #Chem.MolToPDBFile(aa_mol, f"out_{i:0>3d}.pdb", flavor=4)
    confs.append(conf)

gmx_rules = opls_db.rules
ffs = []

for aa_mol in aa_mols_h[:]:
    if aa_mol.GetNumAtoms() < 2:
        ff = OplsFF(database=opls_db,gmx_rules=gmx_rules)
        ff.parameterize(aa_mol)
        ffs.append(ff)
    else:
        MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral, mlimproper)
        ff = OplsFF(database=opls_db,gmx_rules=gmx_rules,custom_typing=[MLModel], custom_angles=[MLModel],
                    custom_dihedrals=[MLModel], custom_bonding=[MLModel], custom_impropers=[MLModel])
        ff.parameterize(aa_mol,custom_rules='all')
        ffs.append(ff)

ret = assemble_opls(aa_mols_h,ffs,confs)

aa_system, xyz, all_forcefields, mols_graphs = ret

write_gro_opls(aa_system, xyz, all_forcefields, mols_graphs,box=list(box)+[0,0,0],ext='gro')
print("total time:", round(time.time()-start,3), 'seconds')
## total time: 6322.657 seconds