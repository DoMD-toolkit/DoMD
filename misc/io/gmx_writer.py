import numpy as np
import networkx as nx
from misc.logger import logger



def write_gro_opls(system: nx.Graph, xyz: np.ndarray, forcefields, mols_graphs, box, postfix='chemfast', ext='gro'):
    if ext not in ['gro', 'pdb']:
        logger.warning(f'The program only supports gro and pdb formats for GROMACS coordinates file, instead got {ext}, turning it into gro')
        ext = 'gro'
    if ext == 'gro':
        context = ''
        context += 'System\n'
        n_atoms = len(system.nodes)
        context += f'  {n_atoms:d}\n'
        for n in system.nodes:
            pos = xyz[n] / 10
            vel = np.zeros_like(pos)
            res_id = (system.nodes[n]['res_id'] + 1) % 100000
            atomname = system.nodes[n]['symbol']
            res_name = system.nodes[n]['res_name']
            atom_id = (n+1) % 100000
            context += f'{res_id:>5d}{res_name:<5s}{atomname:>5s}{atom_id:>5d}{pos[0]:>8.4f}{pos[1]:>8.4f}{pos[2]:>8.4f}{vel[0]:>8.4f}{vel[1]:>8.4f}{vel[2]:>8.4f}\n'
        context += f'{box[0]/10:>10.3f}{box[1]/10:>10.3f}{box[2]/10:>10.3f}'
        grofile = open(postfix + '.' + ext, 'w')
        grofile.write(context)
        grofile.close()
    elif ext == 'pdb':
        context = ''
        n_atoms = len(system.nodes)
        for n in system.nodes:
            pos = xyz[n]
            res_id = (system.nodes[n]['res_id'] + 1) % 10000
            atomname = system.nodes[n]['symbol']
            res_name = system.nodes[n]['res_name']
            atom_id = (n+1) % 100000
            context += f'{"ATOM":<6s}{atom_id:>5d} {atomname:<4s} {res_name:>3s}  {res_id:>4d}    {pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}                {atomname:>2s}  \n'
        context += 'END'
        grofile = open(postfix + '.' + ext, 'w')
        grofile.write(context)
    context = ''
    context += '[ defaults ]\n'
    context += '1 3 yes 0.5 0.5\n'
    context += '[ atomtypes ]\n'
    idx_type = {}
    MLparams = set()
    for n in system.nodes:
        opls_type = forcefields['atoms'][n].name
        if 'ML' in opls_type:
            params = (round(forcefields['atoms'][n].sigma,6), round(forcefields['atoms'][n].epsilon,6))
            MLparams.add(params)
    MLtypes = {p:f'dmd_{i:0>2d}' for i,p in enumerate(MLparams)}
    atomtypes = {}
    for n in system.nodes:
        opls_type = forcefields['atoms'][n].name
        if 'ML' in opls_type:
            params = (round(forcefields['atoms'][n].sigma,6), round(forcefields['atoms'][n].epsilon,6))
            forcefields['atoms'][n].name = MLtypes[params]
        opls_type = forcefields['atoms'][n].name
        params = (round(forcefields['atoms'][n].sigma,6), round(forcefields['atoms'][n].epsilon,6))
        atomtypes[opls_type] = {'epsilon':params[1], 'sigma':params[0], 'bondtype':forcefields['atoms'][n].bond_type,
                                'an':forcefields['atoms'][n].atomic_num, 'mass':forcefields['atoms'][n].mass,
                                'charge':round(forcefields['atoms'][n].charge,8),'ptype':forcefields['atoms'][n].ptype}
        idx_type[n] = opls_type
    for opls_type in atomtypes:
        context += f"{opls_type:>15s} {atomtypes[opls_type]['bondtype']:>12s} {atomtypes[opls_type]['an']:>6d} {atomtypes[opls_type]['mass']:>12.6f} {atomtypes[opls_type]['charge']:>12.8f} {atomtypes[opls_type]['ptype']:>6s} {atomtypes[opls_type]['sigma']:>12f} {atomtypes[opls_type]['epsilon']:>12f}\n"
    context += '[ moleculetype ]\n'
    context += 'assembles  3\n'
    context += '[ atoms ]\n'
    for i in forcefields['atoms']:
        i = i+1
        context += f"{i:>8d} {idx_type[i-1]} {system.nodes[i-1]['res_id']} {system.nodes[i-1]['res_name']:>5s} {forcefields['atoms'][i-1].element:>5s}     1 {atomtypes[idx_type[i-1]]['charge']:>12.8f} {atomtypes[idx_type[i-1]]['mass']:>12.6f}\n"

    context += '[ bonds ]\n'
    for i,j in forcefields['bonds']:
        bond = forcefields['bonds'][i,j]
        ftype = bond.ftype
        params = bond.param[2:-2].split(',')
        context += f"{i+1:>8d} {j+1:>8d}    {ftype:>2d}    {params[0]:>12s} {params[1]:>12s}\n"

    context += '[ angles ]\n'
    for i,j,k in forcefields['angles']:
        angle = forcefields['angles'][i,j,k]
        ftype = angle.ftype
        params = angle.param[2:-2].split(',')
        context += f"{i+1:>8d} {j+1:>8d} {k+1:>8d}   {ftype:>2d}    {params[0]:>12s} {params[1]:>12s}\n"

    context += '[ dihedrals ]\n'
    context += ';torsion\n'
    for i,j,k,l in forcefields['dihedrals']:
        dih = forcefields['dihedrals'][i,j,k,l]
        ftype = dih.ftype
        params = dih.param[2:-2].split(',')
        context += f"{i+1:>8d} {j+1:>8d} {k+1:>8d} {l+1:>8d}  {ftype:>2d}    {params[0]:>12s} {params[1]:>12s} {params[2]:>12s} {params[3]:>12s} {params[4]:>12s} {params[5]:>12s}\n"
    context += ';improper\n'
    for i,j,k,l in forcefields['impropers']:
        imp = forcefields['impropers'][i,j,k,l]
        ftype = imp.ftype
        params = imp.param[1:-1].split(',')
        context += f"{i + 1:>8d} {j + 1:>8d} {k + 1:>8d} {l + 1:>8d}  {ftype:>2d}    {params[0]:>12s} {params[1]:>12s} {params[2]:>12s}\n"

    context += '[ pairs ]\n'
    for i,j,k,l in forcefields['dihedrals']:
        context += f"{i + 1:>8d} {l + 1:>8d}   1\n"

    context += '[ system ]\nsystem\n'
    context += '[ molecules ]\n'
    context += ';mol_name    num\n'
    context += 'assembles   1\n'
    topfile = open(postfix + '.' + 'top', 'w')
    topfile.write(context)
    topfile.close()

    return
