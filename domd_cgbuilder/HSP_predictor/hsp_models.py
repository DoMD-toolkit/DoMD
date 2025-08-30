import torch.nn as nn
import torch
from torch.nn import functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import numpy as np
import pickle
import pandas as pd
import tqdm
import time
import os

en = {
    "H": 2.300, "He": 4.160,
"Li": 0.912, "Be": 1.576,    "B": 2.051, "C": 2.544,    "N": 3.066, "O": 3.610,
    "F": 4.193, "Ne": 4.787,    "Na": 0.869, "Mg": 1.293,    "Al": 1.613, "Si": 1.916,
    "P": 2.253, "S": 2.589,    "Cl": 2.869, "Ar": 3.242,    "K": 0.734, "Ca": 1.034,
    "Sc": 1.19, "Ti": 1.38,    "V": 1.53, "Cr": 1.65,    "Mn": 1.75, "Fe": 1.80,
    "Co": 1.84, "Ni": 1.88,    "Cu": 1.85, "Zn": 1.588,    "Ga": 1.756, "Ge": 1.994,
    "As": 2.211, "Se": 2.424,    "Br": 2.685, "Kr": 2.966,    "Rb": 0.706, "Sr": 0.963,
    "Y": 1.12, "Zr": 1.32,    "Nb": 1.41, "Mo": 1.47,    "Tc": 1.51, "Ru": 1.54,
    "Rh": 1.56, "Pd": 1.58,    "Ag": 1.87, "Cd": 1.521,    "In": 1.656, "Sn": 1.824,
    "Sb": 1.984, "Te": 2.158,    "I": 2.359, "Xe": 2.582,    "Cs": 0.659, "Ba": 0.881,
    "Lu": 1.09, "Hf": 1.16,    "Ta": 1.34, "W": 1.47,    "Re": 1.60, "Os": 1.65,
    "Ir": 1.68, "Pt": 1.72,    "Au": 1.92, "Hg": 1.765,    "Tl": 1.789, "Pb": 1.854,
    "Bi": 2.01, "Po": 2.19,    "At": 2.39, "Rn": 2.60,    "Fr": 0.67, "Ra": 0.89
}

def getneimasssum(atom):
    s = 0
    for nei in atom.GetNeighbors():
        s += nei.GetMass()
    return s

def envien(atom,rdmol):
    bs = []
    nei_en = []
    nei_bias = []
    for nei in atom.GetNeighbors():
        nei_en.append(en[nei.GetSymbol()])
        bs.append(rdmol.GetBondBetweenAtoms(atom.GetIdx(),nei.GetIdx()).GetBondTypeAsDouble())
        bias = 0
        for nnei in nei.GetNeighbors():
            if nnei.GetIdx() == atom.GetIdx():
                continue
            bias += 0.1* en[nnei.GetSymbol()]
        nei_en[-1] += bias
        #print(bias)
    bs = np.array(bs)
    nei_en = np.array(nei_en)
    z = bs.sum()
    if z == 0:
        return 0
    bs = bs/z

    return ( bs * nei_en ).sum()

def smiles2graph(smiles,dD=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise
    mol = AllChem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    fpgen = AllChem.GetMorganGenerator(radius=3,fpSize=1024)
    mp = fpgen.GetFingerprintAsNumPy(mol)
    mp = torch.tensor(list(mp), dtype=torch.float).reshape(1,-1)
    ComputeGasteigerCharges(mol, nIter=120)
    atom_features = []
    for atom in mol.GetAtoms():
        gei = float(atom.GetProp('_GasteigerCharge'))
        hi = atom.GetHybridization()
        eni = en[atom.GetSymbol()]
        neni = float(envien(atom,mol))

        atom_features.append([
            gei * 100,
            atom.GetAtomicNum(),
            int(atom.GetIsAromatic()) * 10,
            int(atom.IsInRing()) * 10,
            atom.GetFormalCharge() * 5,
            2 * hi.real + hi.imag,
            atom.GetExplicitValence() * 5,
            getneimasssum(atom),
            eni * 5,
            neni * 5
        ])
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append((i, j))
        edge_indices.append((j, i))
        edge_attrs.append([bond.GetBondTypeAsDouble()])
        edge_attrs.append([bond.GetBondTypeAsDouble()])
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Graph data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(dD, dtype=torch.float),mp=mp)

    return data


#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
this_dir, this_file = os.path.split(__file__)

class dD_predictor(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_heads, dropout=0.02):
        super(dD_predictor, self).__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.shift = 100

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.gat1(x, edge_index, edge_attr)
        x2 = self.gat2(x1, edge_index, edge_attr)
        x3 = self.gat3(x2+x1, edge_index, edge_attr)
        x = global_mean_pool(x3, batch)# Global pooling
        x1 = self.l1(F.relu(x))
        x2 = self.fc1(F.relu(x1))
        x3 = F.relu(x2)
        y = self.fc2(x3)
        return y.ravel()/self.shift

num_features = 10
hidden_dim = 1024
output_dim = 1
num_heads = 1
dD_model = dD_predictor(num_features,hidden_dim,output_dim,num_heads)
#print(os.path.join('models','mindPPredictor.pt'))
model_p = torch.load(os.path.join(this_dir,'models','dDPredictor.pt'))
dD_model.load_state_dict(model_p)
dD_model.eval()

def predict_dD(smiles):
    gdata = smiles2graph(smiles)
    with torch.no_grad():
        dD = dD_model(gdata.x, gdata.edge_index, gdata.edge_attr, gdata.batch)
    return dD.detach().cpu().numpy().ravel()

class dP_predictor(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_heads, dropout=0.02):
        super(dP_predictor, self).__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.45)
        self.shift = 10

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.gat1(x, edge_index, edge_attr)
        x2 = self.gat2(x1, edge_index, edge_attr)
        x3 = self.gat3(x2+x1, edge_index, edge_attr)
        x = global_mean_pool(x3, batch)# Global pooling
        x1 = self.l1(x)
        x2 = self.fc1(F.relu(x1))
        x3 = F.relu(x2)
        x3 = self.dropout(x3)
        y = self.fc2(x3)
        return y.ravel()/self.shift

num_features = 10
hidden_dim = 128
output_dim = 1
num_heads = 1
dP_model = dP_predictor(num_features,hidden_dim,output_dim,num_heads)
model_p = torch.load(os.path.join(this_dir,'models','dPPredictor.pt'))
dP_model.load_state_dict(model_p)
dP_model.eval()

def predict_dP(smiles):
    gdata = smiles2graph(smiles)
    with torch.no_grad():
        dP = dP_model(gdata.x, gdata.edge_index, gdata.edge_attr, gdata.batch)
    return dP.detach().cpu().numpy().ravel()

class dH_predictor(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_heads, dropout=0.02):
        super(dH_predictor, self).__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.shift = 10

    def forward(self, x, edge_index, edge_attr, batch):
        x1 = self.gat1(x, edge_index, edge_attr)
        x2 = self.gat2(x1, edge_index, edge_attr)
        x3 = self.gat3(x2+x1, edge_index, edge_attr)
        x = global_mean_pool(x3, batch)# Global pooling
        x1 = self.l1(x)
        x2 = self.fc1(F.relu(x1))
        x3 = F.relu(x2)
        x3 = self.dropout(x3)
        y = self.fc2(x3)
        return y.ravel()/self.shift
num_features = 10
hidden_dim = 128
output_dim = 1
num_heads = 1
dH_model = dH_predictor(num_features,hidden_dim,output_dim,num_heads)
model_p = torch.load(os.path.join(this_dir,'models','dHPredictor.pt'))
dH_model.load_state_dict(model_p)
dH_model.eval()

def predict_dH(smiles):
    gdata = smiles2graph(smiles)
    with torch.no_grad():
        dH = dH_model(gdata.x, gdata.edge_index, gdata.edge_attr, gdata.batch)
    return dH.detach().cpu().numpy().ravel()


if __name__ == '__main__':
    smiles = 'C1=C(C(=CC(=C1)[N+]([O-])=O)C)N'
    dD = predict_dD(smiles)
    dP = predict_dP(smiles)
    dH = predict_dH(smiles)
    hsp = (dD**2+dP**2+dH**2)**0.5
    print(dD,dP,dH,hsp)
