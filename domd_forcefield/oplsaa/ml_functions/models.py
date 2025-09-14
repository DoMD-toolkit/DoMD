import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

device = torch.device('cpu')
this_dir, this_file = os.path.split(__file__)


## Nonbond
class GATNonBond(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATNonBond, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y3 = F.relu(self.gat3(y1 + y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1 + y3))
        # y4 = self.dp(y4)
        for ly in self.layers:
            y4 = F.relu(ly(y4))
        xr = self.lf(y4)
        # xr = (F.tanh(self.lf(y4)) * 5 + 1)*shift
        return xr, y3


in_features, out_features, hidden_size, heads = 10, 29, 128, 1
flag = 2
NBModel = GATNonBond(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minNonbond.pt'), map_location="cpu", weights_only=True)
NBModel.load_state_dict(model_p)
NBModel.to(device)
NBModel.eval()

idx_nonbond = pickle.load(open(os.path.join(this_dir, 'resources', 'idx_nonbond.pkl'), 'rb'))
nb_an = pickle.load(open(os.path.join(this_dir, 'resources', 'nbtype_an_hash.pkl'), 'rb'))

def mlnonbond(mol_graph):
    data = mol_graph
    with torch.no_grad():
        crossE, _ = NBModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
    an = data.x_f[:, 1].numpy().ravel()
    classB = []#[np.argmax(np.array([t[i] for i in range(len(t)) if nb_an[idx_nonbond[i]] == an[j]])) for j,t in enumerate(crossE.detach().numpy())]
    #print(crossE.shape)
    for j,t in enumerate(crossE.detach().numpy()):
        candidates = []
        for i in range(len(t)):
            if nb_an[idx_nonbond[i]] == an[j]:
                candidates.append((np.exp(t[i])+1))
            else:
                candidates.append(0)
        #print(candidates)
        classB.append(np.argmax(np.array(candidates)))
    #print(classB)
    nonbondpara_ = [idx_nonbond[i] for i in classB]
    nonbondpara = {}
    orgi_idx = data.orig_idx.numpy()
    for i, p in enumerate(nonbondpara_):
        nonbondpara[orgi_idx[i]] = p
    #for i in nonbondpara:
    #    p = nonbondpara[i]
    #    if nb_an[p] != an[i]:
    #        print('Error in nonbond assignment!')
    #        raise
    return nonbondpara

#raise

## End of nonbond

## Charge
class GATCharge(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATCharge, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gcn1 = GCNConv(hidden_size, hidden_size)
        self.gcn2 = GCNConv(hidden_size, hidden_size)
        self.gcn3 = GCNConv(hidden_size, hidden_size)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)  # ,concat=False)
        self.gat2 = GATConv((3 * hidden_size + hidden_size), hidden_size, edge_dim=2, heads=heads,
                            dropout=0.0)  # ,concat=False)
        self.gat2_1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.0)
        self.gat2_2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.0)
        self.gat2_3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.0)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=heads, dropout=0.02)  # ,concat=False)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.lf = nn.Linear(hidden_size, out_features)
        self.gatf = GATConv(hidden_size, out_features, edge_dim=2, heads=1, dropout=0.0)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat2_1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2_2(y0, edge_index, edge_attr=edge_attr))
        y3 = F.relu(self.gat2_3(y1 + y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1 + y3))
        xr = self.lf(y4)
        return xr, y3


in_features, out_features, hidden_size, heads = 10, 1, 512, 1
CHModel = GATCharge(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minCharge.pt'), map_location="cpu", weights_only=True)
CHModel.load_state_dict(model_p)
CHModel.to(device)


def mlcharge(mol_graph):
    data = mol_graph
    shift = 10
    fcharge = (data.x_f[:, 4]/5).ravel().detach().cpu().numpy().sum()
    with torch.no_grad():
        output, fv = CHModel(data.x_f.float(), data.edge_index, data.bo.float(), shift)
        o = (output.reshape(-1, ) / shift).detach().cpu().numpy()
    #o -= (o.sum()-fcharge) / len(o)
    #print(fcharge.sum(), o.sum())
    charge = {}
    for i, c in enumerate(o):
        charge[i] = c
    return charge


## End of charge

## bond
class GATBondk(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATBondk, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.le0 = nn.Linear(2, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(3):
            self.layers1.append(nn.Linear(2 * hidden_size, 2 * hidden_size))
        self.l4 = nn.Linear(2 * hidden_size, 145)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        e_attr = F.relu(self.le0(edge_attr))
        xb = torch.cat((source + target, e_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        return yb


in_features, out_features, hidden_size, heads = 10, 2, 128, 1
flag = 2
BondModel = GATBondk(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minBond.pt'), map_location="cpu", weights_only=True)
BondModel.load_state_dict(model_p)
BondModel.eval()
BondModel.to(device)

idx_bond = pickle.load(open(os.path.join(this_dir, "resources", 'idx_bond.pkl'), 'rb'))
bond_idx = pickle.load(open(os.path.join(this_dir, "resources", 'bond_idx.pkl'), 'rb'))


def mlbond(mol_graph, mol):
    data = mol_graph
    crossE = BondModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
    classB = [np.argmax(t) for t in crossE.detach().numpy()]
    bondpara_ = [idx_bond[i] for i in classB]
    bondidx = data.bidx.numpy()
    bondpara = {}
    for bidx, bp in zip(bondidx, bondpara_):
        bond = mol.GetBondWithIdx(int(bidx))
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bondpara[(i, j)] = (f'Bond_{bond_idx[bp]:0>3d}_ML', bp[0], bp[1])
    return bondpara


## End of bond

## Angle
class GATAngle(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATAngle, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size + 2, hidden_size + 2)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size + 2, hidden_size + 2))
        self.l4 = nn.Linear(hidden_size + 2, 313)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        xb = torch.cat((source + target, edge_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        xr = self.lf(y4)
        return yb


in_features, out_features, hidden_size, heads = 20, 2, 128, 1
flag = 2

AngleModel = GATAngle(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minAngle.pt'), map_location="cpu", weights_only=True)
AngleModel.load_state_dict(model_p)
AngleModel.to(device)
AngleModel.eval()

idx_angle = pickle.load(open(os.path.join(this_dir, 'resources', 'idx_angle.pkl'), 'rb'))
angle_idx = pickle.load(open(os.path.join(this_dir, 'resources', 'angle_idx.pkl'), 'rb'))


def mlangle(bond_graph):
    data = bond_graph
    crossE = AngleModel(data.b_f.float(), data.edge_index, data.ao.float(), 1)
    bead_idx = data.bead_idx.numpy()
    # print(crossE)
    classA = [np.argmax(t) for t in crossE.detach().numpy()]
    anglepara_ = [idx_angle[i] for i in classA]
    anglepara = {}
    for ap, e in zip(anglepara_, data.edge_index.T):
        bi, bj = e
        bi = int(bi)
        bj = int(bj)
        aidxbi = set(bead_idx[bi])
        aidxbj = set(bead_idx[bj])
        # print(aidxbi,aidxbj)
        aidx_in = aidxbi.intersection(aidxbj)
        aidx_ri = aidxbi - aidx_in
        aidx_li = aidxbj - aidx_in
        i, j, k = list(aidx_ri)[0], list(aidx_in)[0], list(aidx_li)[0]
        anglepara[(i, j, k)] = (f'Angle_{angle_idx[ap]:0>3d}_ML', ap[0], ap[1])
    return anglepara


## End of angle

## Dihedral
class GATDik(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATDik, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size, edge_dim=1, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size, edge_dim=1, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size, edge_dim=1, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.le0 = nn.Linear(1, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size * 2, hidden_size * 2))
        self.l4 = nn.Linear(hidden_size * 2, 204)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self, x, edge_index, edge_attr, shift):
        y0 = F.relu(self.gat1(x, edge_index, edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0, edge_index, edge_attr=edge_attr))
        y4 = F.relu(self.l2(y0 + y1))
        source = y4[edge_index[0]]
        target = y4[edge_index[1]]
        e_attr = F.relu(self.le0(edge_attr.reshape(-1, 1)))
        xb = torch.cat((source + target, e_attr), dim=-1)
        yb0 = F.relu(self.l3(xb))
        for ly in self.layers1:
            yb0 = F.relu(ly(yb0))
        yb = self.l4(yb0)
        return yb


in_features, out_features, hidden_size, heads = 30, 2, 100, 1
flag = 2

DihedralModel = GATDik(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minDi.pt'), map_location="cpu", weights_only=True)
DihedralModel.load_state_dict(model_p)
DihedralModel.to(device)
DihedralModel.eval()

idx_di = pickle.load(open(os.path.join(this_dir, 'resources', 'idx_di.pkl'), 'rb'))
di_idx = pickle.load(open(os.path.join(this_dir, 'resources', 'di_idx.pkl'), 'rb'))

## End of dihedral

## Improper
def mldihedral(angle_graph):
    data = angle_graph
    crossE = DihedralModel(data.a_f.float(), data.edge_index, data.do.float(), 1)
    bead_idx = data.bead_idx.numpy()
    classA = [np.argmax(t) for t in crossE.detach().numpy()]
    dipara_ = [idx_di[i] for i in classA]
    dipara = {}
    idx = data.idx.numpy()
    for dp, e, (ni, i, j, nj) in zip(dipara_, data.edge_index.T, idx):
        bi, bj = e
        bi = int(bi)
        bj = int(bj)
        aidxbi = set(bead_idx[bi])
        aidxbj = set(bead_idx[bj])
        # print(aidxbi,aidxbj)
        aidx_in = aidxbi.intersection(aidxbj)
        aidx_ri = aidxbi - aidx_in
        aidx_li = aidxbj - aidx_in
        # ni, i, j, nj = list(aidx_ri)[0], list(aidx_in)[0], list(aidx_in)[1], list(aidx_li)[0]
        ni, i, j, nj = int(ni), int(i), int(j), int(nj)
        # print(ni, i,j,nj)
        dipara[(ni, i, j, nj)] = (f'Dih_{di_idx[dp]:0>3d}_ML', dp[0], dp[1], dp[2], dp[3], dp[4], dp[5])  # dp
        # dipara[(nj,j,i,ni)] = dp
    return dipara

class GATImp(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, heads, flag=0):
        super(GATImp, self).__init__()
        if flag == 0:
            self.acti = nn.LeakyReLU()
        elif flag == 1:
            self.acti = nn.Softmax()
        elif flag == 2:
            self.acti = nn.ReLU()
        self.l1 = nn.Linear(in_features, hidden_size, bias=False)
        self.gat1 = GATConv(in_features, hidden_size , edge_dim=2, heads=1, dropout=0.01)
        self.gat2 = GATConv(hidden_size, hidden_size , edge_dim=2, heads=1, dropout=0.01)
        self.gat3 = GATConv(hidden_size, hidden_size , edge_dim=2, heads=1, dropout=0.01)

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.layers1 = nn.ModuleList()
        for _ in range(2):
            self.layers1.append(nn.Linear(hidden_size, hidden_size))
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.lf = nn.Linear(hidden_size, out_features)
        self.dp = nn.Dropout(p=0.3)

    def forward(self,x,edge_index,edge_attr,shift):
        y0 = F.relu(self.gat1(x,edge_index,edge_attr=edge_attr))
        y1 = F.relu(self.gat2(y0,edge_index, edge_attr=edge_attr))
        y2 = F.relu(self.gat3(y1+y0,edge_index, edge_attr=edge_attr))
        y3 = F.relu(self.l2(y0+y1+y2))
        yb = self.lf(y3)
        return yb


in_features ,out_features, hidden_size, heads = 10, 3, 128, 1
flag = 2

ImproperModel = GATImp(in_features, hidden_size, out_features, heads, flag)
model_p = torch.load(os.path.join(this_dir, 'resources', 'minImp.pt'), map_location="cpu", weights_only=True)
ImproperModel.load_state_dict(model_p)
ImproperModel.to(device)
ImproperModel.eval()

idx_imp = pickle.load(open(os.path.join(this_dir, 'resources', 'idx_imps.pkl'), 'rb'))
imp_idx = pickle.load(open(os.path.join(this_dir, 'resources', 'imps_idx.pkl'), 'rb'))


def mlimproper(mol_graph):
    data = mol_graph
    crossE = ImproperModel(data.x_f.float(), data.edge_index, data.bo.float(), 1)
    classB = [np.argmax(t) for t in crossE.detach().numpy()]
    impropara_ = [idx_imp[i] for i in classB]
    impropara = {}
    for i, p in enumerate(impropara_):
        impropara[i] = (f'Impr_{imp_idx[p]:0>3d}_ML', p[0], p[1], p[2])
    return impropara
## End of improper