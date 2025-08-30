import os
import sys
sys.path.append('/home/lmy/HTSP/FPSG/')
import math
#from FF.ml.predict import mlcharge
from sys import argv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import combine
from FF.opls_aa.opls_db import opls_db
from FG.cg_topology import read_cg_topology
from FF.opls_aa.opls import OplsFF, DBCharge, MLCharge, MLBond, MLAngle, MLDihedral, MLAtomType
from FF.ml.charge.predict import mlcharge
from FF.ml.nonbond.predict import mlnonbond
from FF.ml.bond.predict import mlbond
from FF.ml.angle.predict import mlangle
from FF.ml.dihedral.predict import mldihedral
from FF.misc import mol2MLgraph
from utils.io.xml_parser import XmlParser
from FG.reactor import Reactor
from FG.misc import set_molecule_id_for_h, AAMoleculeModel
from CG.GenXYZ import Sysinit
from CG.cg_info import *
import pickle
import multiprocessing
import time

start = time.time()

fn_dir = 'aa'

if not os.path.exists(fn_dir):
    os.mkdir(fn_dir)

reaction_template = {
    'P-P': {
        'cg_reactant_list' : [('P', 'P')],
        'smarts' : '[CH3][C:1].[C:2][CH2]>>[CH3][C:1][C:2][CH2]',
        'prod_idx' : [0, 0]
        }
}

#mols = {
#    'P' : {'smiles' : 'CC(=C)C(=O)OC', 'file' : None},
#    'P' : {'smiles' : 'CC(=C)C(=O)OC', 'file' : None},
#}
mols = {
    'P' : {'smiles' : 'CCc1ccccc1', 'file' : None},
}


for key in mols:
    if not mols[key]['file'] is None:
        mol = Chem.RemoveAllHs(Chem.MolFromPDBFile(mols[key]['pdb']))
        mols[key]['smiles'] = Chem.MolToSmiles(mol)

#xml = XmlParser('out.xml')
#xml = XmlParser(os.path.join('cg','final.xml'))
xml = XmlParser(os.path.join('cg','nvt.0001500000.xml'))
box = (xml.box.lx, xml.box.ly, xml.box.lz, xml.box.xy, xml.box.xz, xml.box.yz)
box = np.array(tuple(map(float, box))[:3]) * 10  # a
cg_sys, cg_mols = read_cg_topology(xml, mols)
reactor = Reactor(mols, reaction_template)

# find tri / di reactions for PFR
reactions = []
for bond in xml.data['bond']:
    reactions.append(('P-P', bond[1], bond[2]))
# end

aa_mols, meta = reactor.process(cg_mols, reactions)
[Chem.SanitizeMol(_) for _ in aa_mols]
aa_mols_h = [Chem.AddHs(m) for m in aa_mols]
[set_molecule_id_for_h(m) for m in aa_mols_h]
#print(aa_mols_h[0].GetNumAtoms())
def proc(i):
    global aa_mols_h
    global cg_mols
    global cms
    aa_mol = aa_mols_h[i]
    cg_mol = cg_mols[i]
    use_ml = False 
    ff = OplsFF(aa_mol, database=opls_db)
    molg, bondg, angleg = mol2MLgraph(aa_mol)
    #cm = DBCharge(aa_mol, database=opls_db, radius=3)
    cm = MLCharge(aa_mol, mlcharge)
    cn = MLAtomType(aa_mol, molg, mlnonbond)
    cb = MLBond(aa_mol, molg, mlbond)
    ca = MLAngle(aa_mol, bondg, mlangle)
    cd = MLDihedral(aa_mol, angleg, mldihedral)
    ff.parameterize(custom_charge=[cm], custom_typing=[cn], custom_bond=[cb], custom_angle=[ca], custom_torsion=[cd], boss_radius=3)
    ff.stats()
    force_field = {}
    for k in ff.ff_params:
        force_field[k] = ff.__dict__[k]
    f1 = open(fn_dir + '/%03d.ff.obj' % i, 'wb')
    pickle.dump(force_field, f1)
    f1.close()
    #aa_model = AAMoleculeModel(aa_mol)
    #s = time.time()
    #aa_model.embed(cg_mol, large=20, box=10 * np.array([xml.box.lx, xml.box.ly, xml.box.lz], dtype=np.float64))
    #print(f"Embed {time.time()-s:.3f}s.")
    #aa_model.save_pdb(fn_dir + '/out_%03d.pdb' % i)
arglist = [(aa_mol, cg_mol, i) for aa_mol, cg_mol, i in zip(aa_mols_h, cg_mols, range(len(cg_mols)))]

#proc(0)
with multiprocessing.Pool(30) as p:
    p.map(proc,range(len(arglist)))
#for i in range(len(arglist)):
#    proc(i)
combine.to_gmx(bl=box[0]*0.1,affine=1.0,fn_dir_=fn_dir)
print(f'Total duration: {time.time()-start:.3f} s ')
