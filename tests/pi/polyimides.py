import sys
sys.path.append('<path-to-DoMD>')
from domd_cgbuilder._conf_gen import embed_CG_system
from domd_cgbuilder.cg_mol import CGMol
from misc.cg_system import read_cg_topology
from misc.io.xml_writer import write_xml
from domd_cgbuilder.cg_ff import cg_info, GetBeadinfo, GetBondedtype
import networkx as nx
import numpy as np
import math
import time

start = time.time()

smiA = 'Nc1ccc(Oc2ccc(N)cc2)cc1'
smiB = 'O1C(=O)c2cc(Oc3cc4C(=O)OC(=O)c4cc3)ccc2C1=O'

reaction_template = {
        'A-B': {
            'cg_reactant_list': [('A', 'B')],
            'smarts': '[#7H2:1].[#6:3](=[#8:4])[#8:2][#6:5]=[#8:6]>>[#6:3](=[#8:4])[#7:1][#6:5]=[#8:6].[#8:2].[#1][#1]',
            'prod_idx': [0]
        }
    }
mols = {
    'A': {'smiles': smiA, 'file': None},
    'B': {'smiles': smiB, 'file': None},
}

CG_systems = []
types = ['A','B']
cl = 10
num_chains = 20

for ic in range(num_chains):
    cg_mol = CGMol()
    for i in range(cl):
        t = types[i%2]
        cg_mol.add_atom(ic*cl+i,t)
    for i in range(cl-1):
        #bt = f'{cg_mol.nodes[ic*cl+i]['type']}-{cg_mol.nodes[ic*cl+i+1]['type']}'
        cg_mol.add_bond(ic*cl+i,ic*cl+i+1)
    CG_systems.append(cg_mol)


combined_graph = nx.compose_all(CG_systems)
FFPara = GetBeadinfo(mols, reaction_template, combined_graph)
nonBondPara, nonBondPairPara, BondPara, AnglePara, DihPara, HSPPara = FFPara

for cg_mol in CG_systems:
    for i, j in cg_mol.edges():
        ti = cg_mol.nodes[i]['type']
        tj = cg_mol.nodes[j]['type']
        cg_mol.nodes[i]['r'] = nonBondPara[ti]
        cg_mol.nodes[j]['r'] = nonBondPara[tj]
        bt = f'{ti}-{tj}' if f'{ti}-{tj}' in BondPara.keys() else f'{tj}-{ti}'
        cg_mol.edges[(i,j)]['r0'] = BondPara[bt][0]
        cg_mol.edges[(i,j)]['k'] = BondPara[bt][1]
        cg_mol.edges[(i,j)]['bt'] = bt
    bond_top, angle_top, dih_top = GetBondedtype(FFPara,cg_mol)

    for angle in angle_top:
        i,j,k = angle
        at, t0, tk = angle_top[angle]
        cg_mol.add_angle(i,j,k,tk=tk,t0=t0,bt=at)
    for dih in dih_top:
        i,j,k,l = dih
        dt, t0, tk = dih_top[dih]
        cg_mol.add_dihedral(i,j,k,l,tk=tk,t0=t0,bt=dt)
sigmas = np.array([round(nonBondPara[k][0],3) for k in nonBondPara])
max_sigma = sigmas.max()
rc = 1
rcut = round(max_sigma*1.1,3)
rho = 0.85
boxl = math.ceil((len(combined_graph.nodes)*(rcut*2)**3/rho)**(1/3.0))
box = np.array([boxl,boxl,boxl]).astype(int)
embed_CG_system(CG_systems,box,FFPara,rc,rcut)

write_xml(CG_systems,box,program='hoomd')

# run CG simulation for pre-equilibrium relaxation

# 

# load xml CG conformation file

#from misc.cg_system import read_cg_topoloy
from misc.io.xml_reader import XmlParser
import os

xml = XmlParser(os.path.join('out_chemfast.xml'))
box = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
box = np.array(tuple(map(float, box))[:3]) * 10  # a
cg_sys, cg_mols = read_cg_topology(xml, mols)

# build the whole molecule via reaction
from domd_topology.reactor import Reactor
from domd_topology.functions import set_molecule_id_for_h
from rdkit import Chem
from rdkit.Chem import AllChem
reactor = Reactor(mols, reaction_template)

# find tri / di reactions for PFR
reactions = []
for bond in xml.data['bond']:
    reactions.append((bond[0], bond[1], bond[2]))
# end

aa_mols, meta = reactor.process(cg_mols, reactions)
[Chem.SanitizeMol(_) for _ in aa_mols]
aa_mols_h = [Chem.AddHs(m) for m in aa_mols]
[set_molecule_id_for_h(m) for m in aa_mols_h]

# embed molecule
from domd_xyz.embed_molecule import embed_molecule
#aa_mol = aa_mols_h[0]

confs = []
for i,aa_mol, cg_mol in zip(range(len(aa_mols_h[:])),aa_mols_h[:],cg_mols[:]):
    conf = embed_molecule(aa_mol, cg_mol, box = box)
    Chem.MolToPDBFile(aa_mol, f"out_{i:0>3d}.pdb", flavor=4)
    confs.append(conf)
    #break

# force field parameterization

from domd_forcefield.oplsaa.opls import OplsFF
from domd_forcefield.oplsaa.opls_db import opls_db
from domd_forcefield.oplsaa.ml import OplsMlRule
from domd_forcefield.oplsaa.ml_functions.models import mlnonbond, mlbond, mlangle, mlcharge, mldihedral,mlimproper

gmx_rules = opls_db.rules
ffs = []

for aa_mol in aa_mols_h[:]:
    MLModel = OplsMlRule(mlnonbond, mlcharge, mlbond, mlangle, mldihedral,mlimproper)
    ff = OplsFF(database=opls_db,gmx_rules=gmx_rules,custom_typing=[MLModel], custom_angles=[MLModel],
                       custom_dihedrals=[MLModel], custom_bonding=[MLModel], custom_impropers=[MLModel])
    #ff.parameterize(aa_mol)
    #ff.stats()
    #ff = OplsFF(database=opls_db,gmx_rules=gmx_rules)
    ff.parameterize(aa_mol,custom_rules='all')
    ff.stats()
    ffs.append(ff)
    #break
#raise
from misc.io.assemble import assemble_opls

ret = assemble_opls(aa_mols_h,ffs,confs)

aa_system, xyz, all_forcefields, mols_graphs = ret

from misc.io.xml_writer import write_xml_opls
write_xml_opls(aa_system, xyz, all_forcefields, box=list(box)+[0,0,0])

from misc.io.gmx_writer import write_gro_opls
write_gro_opls(aa_system, xyz, all_forcefields, mols_graphs,box=list(box)+[0,0,0],ext='gro')

print("total time:", round(time.time()-start,3), 'seconds')