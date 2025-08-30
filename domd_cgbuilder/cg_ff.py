from domd_cgbuilder.HSP_predictor.hsp_models import predict_dD, predict_dH, predict_dP
import math
import os
import random
import uuid
import pdbreader
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from misc.logger import logger
from typing import Union, List, Tuple, Dict, Any
import networkx as nx
import numpy as np
import tempfile
#from FG.reactor import Reactor

def replacesmi(smis:Dict) -> Dict:
    r'''
    this function is used to replace the substruct of PI monomer. anhydride -> diamine
    ATTENTION: only used in PI system!!
    :param smis: mole_template, same as \molecules\v in fg.py
    example:
    molecules = {
    'A': {'smiles': 'c1cc(N)ccc1Oc1ccc(N)cc1', 'file': None},

    'B': {'smiles': 'O=c3oc(=O)c4cc(Oc2ccc1c(=O)oc(=O)c1c2)ccc34', 'file': None},
    }
    :return: smis_: new mole_template with the monomer of diamine instead of anhydride
    '''
    patt = Chem.MolFromSmiles('N=C=O')
    rmol = Chem.MolFromSmiles('N')
    smis_ = {}
    for k in smis:
        file = smis[k]['file']
        smis_[k] = {'smiles':None,'file':file}
        smiles = smis[k]['smiles']
        mol = Chem.MolFromSmiles(smiles)
        mat = mol.GetSubstructMatches(patt)
        flag = 0
        if len(mat) >= 2:
            m = AllChem.ReplaceSubstructs(mol,patt,rmol,replaceAll=True)[0]
            flag += 1
        if flag == 0:
            smis_[k]['smiles'] = smiles
            continue
        smis_[k]['smiles'] = Chem.MolToSmiles(m)
    return smis_

def findorder(cg_reactants_list,cg_reactants):
    ti, tj = cg_reactants
    #print(cg_reactants)
    if f'{ti}-{tj}' in cg_reactants_list:
        return (ti,tj)
    elif f'{tj}-{ti}' in cg_reactants_list:
        return (tj,ti)
    else:
        logger.error("conect %s-%s is not deffined in the topology" % (ti,tj))
        raise 'fuck'
def GetTypeinfo(mols:dict, react:dict, top:Union[nx.Graph,list,list]=None) ->  Union[dict,dict,dict]:
    r'''
    find all monomer, dimer, trimer constructure, (what if no angle? )
    :param mols:  mols = {
                        'A': {'smiles': 'c1cc(N=C=O)ccc1Oc1ccc(N=C=O)cc1', 'file': None},
                        'B': {'smiles': 'O=c3oc(=O)c4cc(Oc2ccc1c(=O)oc(=O)c1c2)ccc34', 'file': None},
                         }
    :param react:  reaction_template = {
                 'PI0': {
                         'cg_reactant_list': [('A', 'B'), ],
                         'smarts': '[#7:1].[#6:3](=[#8:4])[#8:2][#6:5]=[#8:6]>>[#6:3](=[#8:4])[#7:1][#6:5]=[#8:6].[#8:2]',
                         'prod_idx': [0,0]
                         }
                                        }
    :return: Union[type:Dict{typeA:mol A}, bondtype:{typeA-typeB: dimer AB}, angletype:{typeA-typeB-typeC: trimer ABC}]
    '''
    type = {}
    for k in mols:
        type[k] = Chem.MolFromSmiles(mols[k]['smiles'])
    react_names = list(react.keys())
    bondtype = {}
    bt_name = {}
    for name in react_names:
        reaction = AllChem.ReactionFromSmarts(react[name]['smarts'])
        prod_idx = react[name]['prod_idx']
        for cg_reactant in react[name]['cg_reactant_list']:
            bt = cg_reactant[0] + '-' + cg_reactant[1]
            p = reaction.RunReactants((type[cg_reactant[0]],type[cg_reactant[1]]))
            if len(p) == 0:
                logger.error(f'There is no reaction deffined between ({cg_reactant[0]},{cg_reactant[1]})!')
                raise
            p = p[prod_idx[0]][prod_idx[0]]
            bondtype[bt] = p
            bt_name[bt] = name
    angletype = {}
    if not top is None:
        bond, angle, dihedral = top
        at = []
        all_bt = list(bt_name.keys())
        #print(all_bt)
        for _ in angle:
            i, j, k = _
            ti ,tj ,tk = (bond.nodes[i]['type'],bond.nodes[j]['type'],bond.nodes[k]['type'])
            at_ = (f'{ti}-{tj}-{tk}',f'{tk}-{tj}-{ti}')
            if at_[0] not in at and at_[1] not in at:
                at.append(at_[0])
                r1 = (ti,tj)
                r2 = (tj,tk)
                try:
                    r1_ = findorder(all_bt,r1)
                except:
                    #print(r1)
                    raise
                p1 = bondtype[f'{r1_[0]}-{r1_[1]}']
                smip1 = Chem.MolToSmiles(p1)
                smitk = Chem.MolToSmiles(type[tk])
                molp1 = Chem.MolFromSmiles(smip1)
                moltk = Chem.MolFromSmiles(smitk)
                r2_ = findorder(all_bt,r2)
                reaction = AllChem.ReactionFromSmarts(react[bt_name[f'{r2_[0]}-{r2_[1]}']]['smarts'])
                prod_idx = react[bt_name[f'{r2_[0]}-{r2_[1]}']]['prod_idx']
                if r2 != r2_:
                    p2 = reaction.RunReactants((moltk,molp1))[prod_idx[0]][prod_idx[0]]
                else:
                    #print(Chem.MolToSmiles(p1),Chem.MolToSmiles(type[tk]))
                    p2 = reaction.RunReactants((molp1,moltk))
                    print(Chem.MolToSmiles(molp1),Chem.MolToSmiles(moltk),react[bt_name[f'{r2_[0]}-{r2_[1]}']]['smarts'],bt_name[f'{r2_[0]}-{r2_[1]}'],at_[0],f'{r2_[0]}-{r2_[1]}')
                    p2 = p2[prod_idx[0]][prod_idx[0]]
                angletype[at_[0]] = p2
    return (type, bondtype, angletype)

def vdwRadiu(pdb):
    p = pdbreader.read_pdb(pdb)
    ks = list(p.keys())
    pos = np.array([p[ks[0]]['x'].array, p[ks[0]]['y'].array, p[ks[0]]['z'].array])
    pos = pos.T
    #print(pos)
    cm = pos.mean(axis=0)
    cm = np.nan_to_num(cm,copy=True)
    #print(pos,cm)
    r = np.max(np.sum(pos**2,axis=-1))**0.5
    r_ = 0
    for _ in np.sum(pos**2,axis=-1)**0.5 :
        if _ > r_:
            r_ = _
    #print(pos**2)
    #print(r,r_)
    return 0.1*r_

def bondLength(pdb: str,dimer : Chem.rdchem.Mol , mol_A: Chem.rdchem.Mol, mol_B : Chem.rdchem.Mol) -> float:
    p = pdbreader.read_pdb(pdb)
    ks = list(p.keys())
    pos = np.array([p[ks[0]]['x'].array,p[ks[0]]['y'].array,p[ks[0]]['z'].array])
    n_atom = dimer.GetNumAtoms()
    dimer = Chem.MolFromPDBFile(pdb)
    mat = dimer.GetSubstructMatches(mol_A)
    if len(mat) == 0:
        #print('fuck')
        mat = dimer.GetSubstructMatches(mol_B)
    if len(mat) == 0:
        logger.warning(f'The diameter {Chem.MolToSmiles(dimer)} cannot match any substructure of monomer {Chem.MolToSmiles(mol_A)} and {Chem.MolToSmiles(mol_B)}.')
        AllChem.EmbedMolecule(mol_A)
        AllChem.UFFOptimizeMolecule(mol_A)
        AllChem.EmbedMolecule(mol_B)
        AllChem.UFFOptimizeMolecule(mol_B)
        ca = mol_A.GetConformer()
        pa = ca.GetPositions()
        ra = np.max(np.sum(pa**2,axis=-1)**0.5)
        cb = mol_B.GetConformer()
        pb = cb.GetPositions()
        rb = np.max(np.sum(pb**2,axis=-1)**0.5)
        return 0.45*(ra+rb)*0.1
        #raise
    mat = list(mat[0])
    mat.sort()
    #print(mat)
    pos = (pos.T)[:n_atom]
    bl = []
    for i in range(len(pos)):
        if i in mat:
            bl.append(1)
        else:
            bl.append(0)
    bl = np.array(bl)
    cm_A = pos[bl == 1].mean(axis=0)
    cm_B = pos[bl != 1].mean(axis=0)
    #print(cm_A,cm_B,pos)
    r0 = np.sum((cm_A - cm_B)**2,axis=-1)**0.5
    return round(0.1*(r0 + 0.1*r0),3)
def angleDegree(pdb: str, trimer: Chem.rdchem.Mol, mol_A: Chem.rdchem.Mol , mol_B: Chem.rdchem.Mol , mol_C: Chem.rdchem.Mol ) -> float:
    p = pdbreader.read_pdb(pdb)
    ks = list(p.keys())
    pos = np.array([p[ks[0]]['x'].array, p[ks[0]]['y'].array, p[ks[0]]['z'].array])
    trimer = Chem.MolFromPDBFile(pdb)
    n_atom = trimer.GetNumAtoms()
    pos = (pos.T)[:n_atom]
    mat = trimer.GetSubstructMatches(mol_A)
    if len(mat) == 0:
        mat = trimer.GetSubstructMatches(mol_B)
        if len(mat) == 0:
            mat = trimer.GetSubstructMatches(mol_C)
    mat_1 = sorted(list(mat[0]))
    mat_2 = sorted(list(mat[1]))
    bl1 = []
    bl2 = []
    for i in range(len(pos)):
        if i in mat_1:
            bl1.append(1)
        else:
            bl1.append(0)
        if i in mat_2:
            bl2.append(1)
        else:
            bl2.append(0)
    bl1 = np.array(bl1)
    bl2 = np.array(bl2)
    bl = bl1 + bl2
    cm_1 = pos[bl1 == 1].mean(axis=0)
    cm_2 = pos[bl2 == 1].mean(axis=0)
    cm_3 = pos[bl != 1 ].mean(axis=0)
    ri = cm_1 - cm_3
    rj = cm_2 - cm_3
    degree = np.arccos((ri*rj).sum(axis=0)/((rj**2).sum()**0.5)/((ri**2).sum()**0.5))
    return round(degree*180/np.pi,1)
def soluPara(monomers: dict, rho=1) -> dict:
    hsp_ = {}
    sulu = {}
    g = []
    for mono in monomers:
        dD = float(predict_dD(Chem.MolToSmiles(monomers[mono])))
        dP = float(predict_dP(Chem.MolToSmiles(monomers[mono])))
        dH = float(predict_dH(Chem.MolToSmiles(monomers[mono])))
        sulu[mono] = (dD**2+dP**2+dH**2)**0.5 * rho
        hsp_[mono] = {'dD':dD, 'dP': dP, 'dH':dH}
        g.append(sulu[mono])
    g = np.array(g)
    maxium = np.max(g)
    minium = np.min(g)
    delta = maxium - minium
    if delta == 0:
        epsilon = {}
        for mono in monomers:
            epsilon[mono] = 1.0
        return epsilon
    epsilonbin = (0.8,1.2)
    delta_e = epsilonbin[1] - epsilonbin[0]
    epsilon = {}
    for mono in monomers:
        epsilon[mono] = sulu[mono]/maxium #round(epsilonbin[0] + (sulu[mono] - minium)/delta * delta_e,3)
    return epsilon, hsp_

def NonBondCombined(nonbond: dict) -> dict:
    nonbond_ = {}
    ks = list(nonbond.keys())
    for i,ki in enumerate(ks):
        for j,kj in enumerate(ks):
            if j >= i :
                nonbond_[(ki,kj)] = (round(0.5*(nonbond[ki][0]+nonbond[kj][0]),3),round((nonbond[ki][1]*nonbond[kj][1])**0.5,3))
    return nonbond_

molecules = {
        'A': Chem.MolFromSmiles('c1cc(N=C=O)ccc1Oc1ccc(N=C=O)cc1')#(1.0,0.2)#,
        ,'B': Chem.MolFromSmiles('O=c3oc(=O)c4cc(Oc2ccc1c(=O)oc(=O)c1c2)ccc34')#(1.2,1.9)#,
        ,'C': Chem.MolFromSmiles('c1ccccc1O')#(1.5,0.9)#,
        ,'D': Chem.MolFromSmiles('c1ccccc1N')#(2.0,0.4)#
        ,'E': Chem.MolFromSmiles('CCO')
    }
#epsilon = soluPara(molecules)

def cg_info(top: nx.Graph) -> Union[nx.Graph,list,list]: ## Union[bond:nx.Graph, angle: Dict, dihedral: Dict]
    bond = top
    angle = []
    for i in bond.nodes():
        nei = list(bond.neighbors(i))
        if len(nei) >= 2:
            for ci,_i in enumerate(nei):
                for _j in nei[ci+1:]:
                    angle.append((_i,i,_j))
    dihedral = []
    for edge in bond.edges():
        degree = dict(bond.degree(edge))
        keys = list(degree.keys())
        i = edge[0]
        j = edge[1]
        nei_i = list(bond.neighbors(edge[0]))
        nei_j = list(bond.neighbors(edge[1]))
        nei_i.remove(j)
        nei_j.remove(i)
        if bond.degree(keys[0]) >= 2 and bond.degree(keys[1]) >= 2 :
            for ni in nei_i:
                for nj in nei_j:
                    dihedral.append((ni,i,j,nj))
    dihedral_flag = 0
    if dihedral_flag == 0:
        dihedral = []
    return (bond,angle,dihedral)

def GetBondedtype(FFPara,top) -> Union[dict,dict,dict]:
    r'''
    get the specific bonded map to the system
    :return: (bondDict{(i,j):(bondtype,(r0,k0)), ... },
              angleDict{(i,j,k):(angletype, (th0,k0)), ...})
    '''
    bond = {}
    angle = {}
    dihedral = {}
    bond_Para = FFPara[2]
    angle_Para = FFPara[3]
    dihedral_Para = FFPara[4]
    top, angle_top, dih_top = cg_info(top)
    bk = list(bond_Para.keys())
    ak = list(angle_Para.keys())
    dk = list(dihedral_Para.keys())
    for bond_ in top.edges():
        bti = top.nodes[bond_[0]]['type']
        btj = top.nodes[bond_[1]]['type']
        if f'{bti}-{btj}' in bk:
            bt = f'{bti}-{btj}'
        elif f'{btj}-{bti}' in bk:
            bt = f'{btj}-{bti}'
        else :
            raise f"Can't not find bondtype {btj}-{bti} in the bondPara."
        bond_hash = (bond_[0],bond_[1])
        bond[bond_hash] = (bt,bond_Para[bt][0],bond_Para[bt][1])
    for angle_ in angle_top:
        i ,j ,k = angle_
        bti = top.nodes[i]['type']
        btj = top.nodes[j]['type']
        btk = top.nodes[k]['type']
        if f'{bti}-{btj}-{btk}' in ak:
            bt = f'{bti}-{btj}-{btk}'
        elif f'{btk}-{btj}-{bti}' in ak:
            bt = f'{btk}-{btj}-{bti}'
        else :
            raise f"Can't not find angletype {bti}-{btj}-{btk} in the bondPara."
        angle_hash = (i,j,k)
        angle[angle_hash] = (bt,angle_Para[bt][0],angle_Para[bt][1])
    for dih_ in dih_top:
        i, j, k, l = dih_
        bti = top.nodes[i]['type']
        btj = top.nodes[j]['type']
        btk = top.nodes[k]['type']
        btl = top.nodes[l]['type']
        if f'{bti}-{btj}-{btk}-{btl}' in ak:
            bt = f'{bti}-{btj}-{btk}-{btl}'
        elif f'{btl}-{btk}-{btj}-{bti}' in ak:
            bt = f'{btl}-{btk}-{btj}-{bti}'
        else :
            raise f"Can't not find dihedraltype {bti}-{btj}-{btk}-{btl} in the bondPara."
        dih_hash = (i,j,k,l)
        dihedral[dih_hash] = (bt, dihedral_Para[bt][0], dihedral_Para[bt][1])
    return (bond,angle,dihedral)

#cg_top_info: Union[nx.Graph,list,list]
def GetBeadinfo(mols:dict, react:dict, system_graph,epsilons=None) -> Union[dict,dict,Any]:
    r'''
    :param mols: molecules template like init define
    :param react: Reactor of monomer
    :system_graph: the total graph representation of system
    :return: Union[nobond:Dict{(typeA,typeA):(sigma,epsilon), ... , (typeA, typeB): (sigma, epsilon)#from combined rule},
                   bond:Dict{bondtype: (k0,r0), ... },
                   angle:Dict{angletype: (k0,th0), ... }]
    '''
    nonbond = {}
    bond = {}
    angle = {}
    tem_path = tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__())
    ## 1. GET ALL CG TYPE, BOND TYPE, ANGLE TYPE INFOMATION: Reactor? Molecule graph(specific nodes with cg type)?
    ## 2. ITERATE ALL MONOMER, DIMER, TRIMER AND GENERATE PBD TO TEM_PATH
    ## 3. CALCULATE SIGMA, BOND LENGTH, ANGLE DEGREE
    cg_top_info = cg_info(system_graph)
    monomers, dimers, trimers = GetTypeinfo(mols,react,cg_top_info) ##
    if epsilons is None:
        epsilons, hsp_ = soluPara(monomers)
    for i,mono in enumerate(monomers):
        tem_file = os.path.join(tem_path.name, f'monomer_{i}.pdb')
        smiles_ = Chem.MolToSmiles(monomers[mono])
        mol = Chem.MolFromSmiles(smiles_)
        molh = AllChem.AddHs(mol)

        try:
            #print(Chem.MolToSmiles(monomers[mono]))
            AllChem.EmbedMolecule(molh,useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(molh)
            Chem.MolToPDBFile(molh,tem_file,flavor=4)
            #print(f'good creation of {mono}!')
        except:
            Chem.MolToPDBFile(molh, tem_file, flavor=4)
        nonbond[mono] = (vdwRadiu(tem_file),epsilons[mono])
        logger.info(f"Successfully set atomtype {mono} parameter: sigma={nonbond[mono][0]},epsilon={nonbond[mono][1]}")
    nonbond_ = nonbond
    nonbond = NonBondCombined(nonbond)
    for i, dimer in enumerate(dimers):
        tem_file = os.path.join(tem_path.name, f'dimer_{i}.pdb')
        A, B = dimer.split('-')
        fuck = Chem.MolToSmiles(dimers[dimer])
        molh = AllChem.AddHs(Chem.MolFromSmiles(fuck))
        #raise
        #molh = AllChem.AddHs(dimers[dimer])
        try :
            AllChem.EmbedMolecule(molh,useRandomCoords=True)
            AllChem.UFFOptimizeMolecule(molh)
            Chem.MolToPDBFile(molh,tem_file,flavor=4)
            #Chem.MolToPDBFile(molh, f'dimer_{i}.pdb', flavor=4)
        except:
            logger.warning(f"Can't optimize dimer :{fuck}")
            Chem.MolToPDBFile(molh, tem_file, flavor=4)
            #Chem.MolToPDBFile(molh, f'dimer_{i}.pdb', flavor=4)
        #print(dimer)
        bond[dimer] = (bondLength(tem_file, dimers[dimer], monomers[A], monomers[B]),1111.0)
        #if math.isnan(bond[dimer][0]):
        if 1:
            enum_pair = ((A,B),(B,A))
            for pair in enum_pair:
                if pair in nonbond.keys():
                    #bond_p = (round(nonbond[pair][0]*2,2),11111.0)
                    bond[dimer] = (round(nonbond[pair][0]*2,2),1111.0)

        logger.info(f"Successfully set bondtype {dimer} parameter: r0={bond[dimer][0]},k0=1111.0")
    if (trimers.values()) == 0:
        tem_path.cleanup()
        return nonbond_, nonbond, bond, angle, hsp_
    else:
        for i, trimer in enumerate(trimers):
            tem_file = os.path.join(tem_path.name, f'trimer_{i}.pdb')
            A, B, C = trimer.split('-')
            try :
                fuck = Chem.MolToSmiles(trimers[trimer])
                molh = AllChem.AddHs(Chem.MolFromSmiles(fuck))
                AllChem.EmbedMolecule(molh,useRandomCoords=True)
                AllChem.UFFOptimizeMolecule(molh)
                Chem.MolToPDBFile(molh, tem_file, flavor=4)
                ag = angleDegree(tem_file, trimers[trimer], monomers[A], monomers[B], monomers[C])
                if ag < 160 or ag == nan:
                    ag = 180
                angle[trimer] = (ag, 11.0)
            except :
                fuck = Chem.MolToSmiles(trimers[trimer])
                logger.warning(f"The trimer Can't to be added Hs or dump pdb file. Set the angle parameter to (th0=180,k0=11.0).  SMILES: {fuck} ")
                Chem.MolToPDBFile(trimers[trimer], tem_file, flavor=4)
                angle[trimer] = (180, 11.0)
    tem_path.cleanup()
    #print((nonbond, bond, angle))
    return nonbond_, nonbond, bond, angle,{}, hsp_


if __name__ == '__main__':
    r'''
    ## TODO
    #  1. train data set of solubility parameter, and the epoch steps
    #  2. way to set eplison value
    #  3. find dihedral type
    #  4. prod_idx
    ## example code
    '''
    smiA = 'Nc1cc(O)ccc1'
    smiB = 'Oc1cc(O)ccc1'
    smiX0 = 'C1OC1C'
    smiX1 = 'ClCC(O)C'
    smiY = 'N(F)(F)c1ccc(C)cc1'
    reaction_template = {
        'X1-Y': {
            'cg_reactant_list': [('X1', 'Y'),],
            'smarts': '[O:1][C:2][C:3].[F:4][N:5]>>[O:1][C:2][C:3][N:5].[F:4]',
            'prod_idx': [0]
        },
        'X0-Y': {
            'cg_reactant_list': [('X0', 'Y'),],
            'smarts': '[O:1][C:2][C:3].[F:4][N:5]>>[O:1][C:2][C:3][N:5].[F:4]',
            'prod_idx': [0]
        },
        'Y-Y': {
            'cg_reactant_list': [('Y', 'Y')],
            'smarts': '[C:1][c:2].[C:3][c:4]>>[c:2][C:1][c:4].[C:3]',
            'prod_idx': [0]

        },
        'A-B': {
            'cg_reactant_list': [('A', 'B')],
            'smarts': '[c:1][O:2].[c:3][O:4]>>[c:1][O:2][c:3].[O:4]',
            'prod_idx': [0]

        },
        'A-X1': {
            'cg_reactant_list': [('A', 'X1')],
            'smarts': '[N:1].[C:2][Cl:3]>>[N:1][C:2].[Cl:3]',
            'prod_idx': [0]

        }
    }
    mols = {
        'A': {'smiles': smiA, 'file': None},
        'B': {'smiles': smiB, 'file': None},
        'X0': {'smiles': smiX0, 'file': None},
        'X1': {'smiles': smiX1, 'file': None},
        'Y': {'smiles': smiY, 'file': None},
    }

    ty = ['A', 'B']
    N = 100
    ty = ['A', 'B']
    top = nx.Graph()
    for i in range(N):
        top.add_node(15 * i, type='A')
        top.add_node(15 * i + 1, type='B')
        top.add_node(15 * i + 2, type='A')
        top.add_node(15 * i + 3, type='X1')
        top.add_node(15 * i + 4, type='Y')
        top.add_node(15 * i + 5, type='Y')
        top.add_node(15 * i + 6, type='X0')
        top.add_node(15 * i + 7, type='X0')
        top.add_node(15 * i + 8, type='X0')
        top.add_node(15 * i + 9, type='X1')
        top.add_node(15 * i + 10, type='X0')
        top.add_node(15 * i + 11, type='Y')
        top.add_node(15 * i + 12, type='Y')
        top.add_node(15 * i + 13, type='X0')
        top.add_node(15 * i + 14, type='X0')
        top.add_edge(15 * i, 15 * i + 1)
        top.add_edge(15 * i + 1, 15 * i + 2)
        top.add_edge(15 * i + 2, 15 * i + 3)
        top.add_edge(15 * i + 2, 15 * i + 9)
        top.add_edge(15 * i + 3, 15 * i + 4)
        top.add_edge(15 * i + 4, 15 * i + 5)
        top.add_edge(15 * i + 5, 15 * i + 6)
        top.add_edge(15 * i + 9, 15 * i + 11)
        top.add_edge(15 * i + 11, 15 * i + 12)
        top.add_edge(15 * i + 12, 15 * i + 13)
        top.add_edge(15 * i + 10, 15 * i + 11)
        top.add_edge(15 * i + 12, 15 * i + 14)

    #mols = replacesmi(mols)
    epsilons={'A':1.0,'B':1.0,'X1':1.0,'X0':1.0,'Y':1.0}
    bondPara = GetBeadinfo(mols,reaction_template,cg_info(top))
    print(bondPara)
