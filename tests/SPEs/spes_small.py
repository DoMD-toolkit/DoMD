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

# TFSI
MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral,mlimproper)
mol = Chem.MolFromSmiles('C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F')
ff = OplsFF(database=opls_db,custom_typing=[MLModel], custom_angles=[MLModel],custom_bonding=[MLModel],custom_dihedrals=[MLModel],custom_impropers=[MLModel])
ff.parameterize(mol)

#import pickle
#aa_system, xyz, all_forcefields, mols_graphs = pickle.load(open('spe_small_meta.pkl','rb'))
#write_gro_opls(aa_system, xyz, all_forcefields, mols_graphs,box=[20,20,20,0,0,0],ext='pdb', prefix='test')
#raise
#ff.stats()
#for a in ff.atoms.atoms:
#    atom = ff.atoms.atoms[a]
#    print(atom)
#raise
start = time.time()

A0 = 'C1(=O)N(F)C(=O)N(F)C(=O)N1F'
A1 = 'c1coc(COC(=O)N)c1'
A2 = 'NC(=O)O'
A3 = 'C1=CC(=O)NC1(=O)'
A4 = 'CCOC(=O)N'
B0 = 'CCC'
B2 = 'C'
B3 = 'Fc1ccc(Br)cc1'
C = 'OCCO'
Li = '[Li+]'
N = 'O[N-]O'
S = 'FC(F)(F)S(=O)(=O)O'

reaction_template = {
        'A0-B0': {
            'cg_reactant_list': [('A0', 'B0')],
            'smarts': '[#7:1][F:2].CC[C:3]>>[#7:1][C:3]CC.[F:2]',
            'prod_idx': [0]
        },
        'B0-A1': {
            'cg_reactant_list': [('B0', 'A1')],
            'smarts': '[CH3:1].[N:2]>>[C:1][N:2]',
            'prod_idx': [0]
        },
    'B0-A2': {
        'cg_reactant_list': [('B0', 'A2')],
        'smarts': '[CH3:1].[N:2][C:3](=[O:4])>>[C:1][N:2][C:3](=[O:4])',
        'prod_idx': [0]
    },
    'B0-A4': {
        'cg_reactant_list': [('B0', 'A4')],
        'smarts': '[CH3:1].[N:2][C:3](=[O:4])>>[C:1][N:2][C:3](=[O:4])',
        'prod_idx': [0]
    },
    'B0-B0': {
        'cg_reactant_list': [('B0', 'B0')],
        'smarts': '[CH3:1].[CH3:2]>>[C:1][C:2]',
        'prod_idx': [0]
    },
    'A2-C': {
        'cg_reactant_list': [('A2', 'C')],
        'smarts': '[OH1:1].[C:2][O:3]>>[O:1][C:2].[O:3]',
        'prod_idx': [0]
    },
    'C-A4': {
        'cg_reactant_list': [('C', 'A4')],
        'smarts': '[OH1:1].[CH3:2]>>[O:1][C:2]',
        'prod_idx': [0]
    },
    'C-C': {
        'cg_reactant_list': [('C', 'C')],
        'smarts': '[C:1][O:2].[C:3][O:4]>>[C:1][O:2][C:3].[O:4]',
        'prod_idx': [0]
    },
    'A3-B3': {
        'cg_reactant_list': [('A3', 'B3')],
        'smarts': '[N:1].[F:2][c:3]>>[N:1][c:3].[F:2]',
        'prod_idx': [0]
    },
    'B3-B2-B3': {
        'cg_reactant_list': [('B3', 'B2', 'B3')],
        'smarts': '[c:1][Br:2].[C:3].[c:4][Br:5]>>[c:1][C:3][c:4].[Br:2][Br:5]',
        'prod_idx': [0]
    },
    'S-N-S': {
        'cg_reactant_list': [('S', 'N', 'S')],
        'smarts': '[S:1][O:2].[O:3][N-:4][O:5].[S:6][O:7]>>[S:1][N-:4][S:6].[O:2][O:3].[O:5][O:7]',
        'prod_idx': [0]
    },
    'A1-A3': {
        'cg_reactant_list': [('A1', 'A3')],
        'smarts': '[#6:1]1[#6:2][#6:3][#8:4][#6:5]1.[#6:6]=[#6:7]>>[#6:1]1=[#6:2][#6:3]2[#6:6][#6:7][#6:5]1[#8:4]2',
        'prod_idx': [0]
    },
    'Li': {
        'cg_reactant_list': [('Li',)],
        'smarts': '[Li+]>>[Li+]',
        'prod_idx': [0]
    },
}

mols = {
    'A0': {'smiles': A0, 'file': None},
    'A1': {'smiles': A1, 'file': None},
    'A2': {'smiles': A2, 'file': None},
    'A3': {'smiles': A3, 'file': None},
    'A4': {'smiles': A4, 'file': None},
    'B0': {'smiles': B0, 'file': None},
    'B2': {'smiles': B2, 'file': None},
    'B3': {'smiles': B3, 'file': None},
    'C' : {'smiles': C , 'file': None},
    'Li': {'smiles': Li, 'file': None},
    'N':  {'smiles': N , 'file': None},
    'S':  {'smiles': S , 'file': None},
}

for key in mols:
    if not mols[key]['file'] is None:
        mol = Chem.RemoveAllHs(Chem.MolFromPDBFile(mols[key]['pdb']))
        mols[key]['smiles'] = Chem.MolToSmiles(mol)

xml = XmlParser(os.path.join('cg','spes_cross.xml'))
box = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
box = np.array(tuple(map(float, box))[:3]) * 10  # a
cg_sys, cg_mols = read_cg_topology(xml, mols)

reactor = Reactor(mols, reaction_template)

reactions = []
ignore = set(['B3-B2','S-N'])
for bond in xml.data['bond']:
    if bond[0] in ignore:
        continue
    reactions.append((bond[0], bond[1], bond[2]))
needed = set(['B3-B2-B3','S-N-S'])
for a in xml.data['angle']:
    if a[0] in needed:
        reactions.append((a[0],a[1],a[2],a[3]))
for i,t in enumerate(xml.data['type']):
    if t == 'Li':
        reactions.append((t,i))

aa_mols, meta = reactor.process(cg_mols, reactions)
[Chem.SanitizeMol(_) for _ in aa_mols]
aa_mols_h = [Chem.AddHs(m) for m in aa_mols]
[set_molecule_id_for_h(m) for m in aa_mols_h]

gmx_rules = opls_db.rules
ffs = []

for aa_mol in aa_mols_h[:]:
    if aa_mol.GetNumAtoms() < 2:
        ff = OplsFF(database=opls_db,gmx_rules=gmx_rules)
        ff.parameterize(aa_mol)
        ffs.append(ff)
    elif aa_mol.GetNumAtoms() < 20:
        MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral,mlimproper)
        ff = OplsFF(database=opls_db,custom_typing=[MLModel], custom_angles=[MLModel],
                       custom_dihedrals=[MLModel], custom_bonding=[MLModel], custom_impropers=[MLModel])
        ff.parameterize(aa_mol)
        ffs.append(ff)
    else:
        MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral, mlimproper)
        ff = OplsFF(custom_typing=[MLModel], custom_angles=[MLModel],
                       custom_dihedrals=[MLModel], custom_bonding=[MLModel], custom_impropers=[MLModel])
        ff.parameterize(aa_mol,custom_rules='all')
        ffs.append(ff)

confs = []
for i,aa_mol, cg_mol in zip(range(len(aa_mols_h[:])),aa_mols_h[:],cg_mols[:]):
    if aa_mol.GetNumAtoms() > 10000:
        chunk_per_d=4
    else:
        chunk_per_d=1
    conf = embed_molecule(aa_mol, cg_mol, box = box, chunk_per_d=chunk_per_d)
    #Chem.MolToPDBFile(aa_mol, f"out_{i:0>3d}.pdb", flavor=4)
    confs.append(conf)


#print("total time:", round(time.time()-start,3), 'seconds')
#raise
ret = assemble_opls(aa_mols_h,ffs,confs)

aa_system, xyz, all_forcefields, mols_graphs = ret
import pickle
#pickle.dump(ffs, open('spe_small_ffs.pkl','wb'))
pickle.dump((aa_system, xyz, all_forcefields, mols_graphs), open('spe_small_meta.pkl','wb'))
write_gro_opls(aa_system, xyz, all_forcefields, mols_graphs,box=list(box)+[0,0,0],ext='pdb')
print("total time:", round(time.time()-start,3), 'seconds')
## total time: 2094.721 seconds