import os

import sys
sys.path.append('/home/lmy/HTSP/FPSG/')

from sys import argv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import combine
from FF.opls_aa.opls_db import opls_db
from FG.cg_topology import read_cg_topology
from FF.opls_aa.opls import OplsFF, DBCharge
from utils.io.xml_parser import XmlParser
from FG.reactor import Reactor
from FG.misc import set_molecule_id_for_h, AAMoleculeModel
from CG.GenXYZ import Sysinit
from CG.cg_info import *
import pickle

if not os.path.exists('cg'):
    os.mkdir('cg')

fn_dir = 'cg'
smiA = 'Nc1ccc(N)cc1'
smiB = 'O1C(=O)c2cc3C(=O)OC(=O)c3cc2C1=O'
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
molA = Chem.MolFromSmiles(mols['A']['smiles'])
molB = Chem.MolFromSmiles(mols['B']['smiles'])
molAh = AllChem.AddHs(molA)
molBh = AllChem.AddHs(molB)
Na = molAh.GetNumAtoms()
Nb = molBh.GetNumAtoms()
rho = 0.85
cl = 20
n_mol = 100
ty = ['A', 'B']
top = nx.Graph()
for j in range(n_mol):
    for i in range(cl):
        if (j*cl+i) % 2 == 0:
            top.add_node(j*cl+i, type=ty[1])
        else:
            top.add_node(j*cl+i, type=ty[0])
    top.add_edges_from([(j*cl+i, j*cl+i + 1) for i in range(cl-1)])
top = cg_info(top)
bondPara = GetBeadinfo(mols, reaction_template, top,epsilons={'A':1.0,'B':1.0})
chain_length = [cl for i in range(n_mol)]
a = Sysinit(bondPara,top,rho)
#res = a.LinearPolymer(chain_length)
res = a.UniformPolymer()
a.Dumpxml(res,filename='cg/out')
a.DumpHoomdscript(filename='cg/Pre_CG')

