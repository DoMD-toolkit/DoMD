import warnings

import networkx as nx
import numpy as np
from rdkit import Chem
from misc.logger import logger


def _warning(message, category=None, filename=None, lineno=None, file=None, line=None):
    print("WARNING: ", message)


warnings.showwarning = _warning

template = '''<?xml version ="1.0" encoding ="UTF-8" ?>
<{program}_xml version="{version}">
<configuration time_step="0" dimensions="3" natoms="{n_atoms:d}" >
<box lx="{lx:.8f}" ly="{ly:.8f}" lz="{lz:.8f}" xy="{xy:8f}" xz="{xz:8f}" yz="{yz:8f}"/>
<position num="{n_atoms:d}">
{positions}</position>
<type num="{n_atoms:d}">
{types}</type>
<opls_type num="{n_atoms:d}">
{opls_type}</opls_type>
<monomer_id num="{n_atoms:d}">
{monomer_id}</monomer_id>
<charge num="{n_atoms:d}">
{charge}</charge>
<mass num="{n_atoms:d}">
{mass}</mass>
<bond num="{n_bonds:d}">
{bond}
</bond>
<angle num="{n_angles:d}">
{angle}
</angle>
<dihedral num="{n_dihedrals:d}">
{dihedral}
</dihedral>
<improper num="{n_impropers:d}">
{improper}
</improper>
</configuration>
</{program}_xml>'''


def write_xml_opls(system: nx.Graph, xyz: np.ndarray, forcefields, box, postfix='chemfast', program='galamost', version='1.3'):
    n_atoms = len(system.nodes)
    n_bonds = len(system.edges)
    mass = types = opls_type = positions = charge = monomer_id = ''
    for atom_graph in system.nodes:
        atom = Chem.Atom(atom_graph['symbol'])
        mass += '%.6f\n' % atom.GetMass()
        types += '%s\n' % atom_graph['bond_type']
        opls_type += '%s\n' % atom_graph['bond_type']
        pos = xyz[atom_graph]
        positions += '%.6f %.6f %.6f\n' % (pos.x / 10, pos.y / 10, pos.z / 10)  # in nm
        charge += '%.6f\n' % forcefields[atom_graph].charge
        monomer_id += '%d\n' % atom_graph['global_res_id']
    n_angles = len(forcefields['angles'])
    n_dihedrals = len(forcefields['dihedrals'])
    n_impropers = len(forcefields['impropers'])
    angle = ''
    for k in forcefields['angles']:
        angle += f"{forcefields['angles'][k].name} {k[0]} {k[1]} {k[2]} # {forcefields['angles'][k].param}\n"
    dihedral = ''
    for k in forcefields['dihedrals']:
        dihedral += (f"{forcefields['dihedrals'][k].name} "
                     f"{k[0]} {k[1]} {k[2]} {k[3]} # {forcefields['dihedrals'][k].param}\n")
    bond = ''
    for k in forcefields['bonds']:
        bond += f"{forcefields['bonds'][k].name} {k[0]} {k[1]} # {forcefields['bonds'][k].param}\n "
    improper = ''
    for k in forcefields['impropers']:
        improper += (f"{forcefields['impropers'][k].name} "
                     f"{k[0]} {k[1]} {k[2]} {k[3]} # {forcefields['impropers'][k].param}\n")
    lx, ly, lz, xy, xz, yz = box
    o = open('out_%s.xml' % postfix, 'w')
    o.write(
        template.format(
            n_atoms=n_atoms, n_bonds=n_bonds, mass=mass, types=types, opls_type=opls_type, positions=positions,
            bond=bond, charge=charge, angle=angle, dihedral=dihedral, n_angles=n_angles, n_dihedrals=n_dihedrals,
            monomer_id=monomer_id, program=program, version=version, lx=lx, ly=ly, lz=lz, xy=xy, xz=xz, yz=yz,
            n_impropers=n_impropers, improper=improper
        ))
    o.close()
    return

import warnings

import networkx as nx
import numpy as np
from rdkit import Chem


def _warning(message, category=None, filename=None, lineno=None, file=None, line=None):
    print("WARNING: ", message)


warnings.showwarning = _warning

template = '''<?xml version ="1.0" encoding ="UTF-8" ?>
<{program}_xml version="{version}">
<configuration time_step="0" dimensions="3" natoms="{n_atoms:d}" >
<box lx="{lx:.8f}" ly="{ly:.8f}" lz="{lz:.8f}" xy="{xy:8f}" xz="{xz:8f}" yz="{yz:8f}"/>
<position num="{n_atoms:d}">
{positions}</position>
<type num="{n_atoms:d}">
{types}</type>
<opls_type num="{n_atoms:d}">
{opls_type}</opls_type>
<monomer_id num="{n_atoms:d}">
{monomer_id}</monomer_id>
<charge num="{n_atoms:d}">
{charge}</charge>
<mass num="{n_atoms:d}">
{mass}</mass>
<bond num="{n_bonds:d}">
{bond}
</bond>
<angle num="{n_angles:d}">
{angle}
</angle>
<dihedral num="{n_dihedrals:d}">
{dihedral}
</dihedral>
<improper num="{n_impropers:d}">
{improper}
</improper>
</configuration>
</{program}_xml>'''


def write_xml_opls(system: nx.Graph, xyz: np.ndarray, forcefields, box, postfix='chemfast', program='galamost', version='1.3'):
    n_atoms = len(system.nodes)
    n_bonds = len(system.edges)
    mass = types = opls_type = positions = charge = monomer_id = ''
    for n in system.nodes:
        atom_graph = system.nodes[n]
        atom = Chem.Atom(atom_graph['symbol'])
        mass += '%.6f\n' % atom.GetMass()
        types += '%s\n' % atom_graph['bond_type']
        opls_type += '%s\n' % atom_graph['bond_type']
        pos = xyz[n]
        positions += '%.6f %.6f %.6f\n' % (pos[0] / 10, pos[1] / 10, pos[2] / 10)  # in nm
        charge += '%.6f\n' % forcefields['atoms'][n].charge
        monomer_id += '%d\n' % atom_graph['res_id']
    n_angles = len(forcefields['angles'])
    n_dihedrals = len(forcefields['dihedrals'])
    n_impropers = len(forcefields['impropers'])
    angle = ''
    for k in forcefields['angles']:
        angle += f"{forcefields['angles'][k].name} {k[0]} {k[1]} {k[2]}\n"
    dihedral = ''
    for k in forcefields['dihedrals']:
        dihedral += (f"{forcefields['dihedrals'][k].name} "
                     f"{k[0]} {k[1]} {k[2]} {k[3]}\n")
    bond = ''
    for k in forcefields['bonds']:
        bond += f"{forcefields['bonds'][k].name} {k[0]} {k[1]}\n"
    improper = ''
    for k in forcefields['impropers']:
        improper += (f"{forcefields['impropers'][k].name} "
                     f"{k[0]} {k[1]} {k[2]} {k[3]}\n")
    lx, ly, lz, xy, xz, yz = box
    lx /= 10; ly/=10; lz/=10; xy/=10; xz/=10; yz/=10
    o = open('out_%s.xml' % postfix, 'w')
    o.write(
        template.format(
            n_atoms=n_atoms, n_bonds=n_bonds, mass=mass, types=types, opls_type=opls_type, positions=positions,
            bond=bond, charge=charge, angle=angle, dihedral=dihedral, n_angles=n_angles, n_dihedrals=n_dihedrals,
            monomer_id=monomer_id, program=program, version=version, lx=lx, ly=ly, lz=lz, xy=xy, xz=xz, yz=yz,
            n_impropers=n_impropers, improper=improper
        ))
    o.close()
    return

def write_xml(CG_systems, box, postfix='chemfast', program='galamost', version='1.3'):
    CG_system_graph = nx.compose_all(CG_systems)
    n_atoms = len(CG_system_graph.nodes)
    n_bonds = len(CG_system_graph.edges)
    mass = types = opls_type = positions = charge = monomer_id = ''
    bond = angle = dihedral = improper  = ''
    for system in CG_systems:
        for atom_graph in system.nodes:
            mass += '1.0\n'
            charge += '0\n'
            monomer_id += f'{atom_graph}\n'
            types += '%s\n' % system.nodes[atom_graph]['type']
            pos = system.nodes[atom_graph]['x']
            positions += '%.6f %.6f %.6f\n' % (pos[0], pos[1], pos[2])  # in nm
        if system._hyperedges.get(3) is None:
            n_angles = 0
            angles = {}
        else:
            n_angles = len(system._hyperedges[3].keys())
            angles = system._hyperedges[3]
        if system._hyperedges.get(4) is None:
            n_dihedrals = 0
            dihedrals = {}
        else:
            n_dihedrals = len(system._hyperedges[4].keys())
            dihedrals = system._hyperedges[4]
        for k in angles:
            angle += f"{angles[k]['bt']} {k[0]} {k[1]} {k[2]}\n"
        dihedral = ''
        for k in dihedrals:
            dihedral += (f"{dihedrals[k]['bt']} "
                         f"{k[0]} {k[1]} {k[2]} {k[3]}\n")
        for k in system.edges:
            bond += f"{system.edges[k]['bt']} {k[0]} {k[1]}\n"
    lx, ly, lz = box
    xy, xz, yz = [0, 0, 0]
    o = open('out_%s.xml' % postfix, 'w')
    o.write(
        template.format(
            n_atoms=n_atoms, n_bonds=n_bonds, mass=mass, types=types, opls_type=opls_type, positions=positions,
            bond=bond, charge=charge, angle=angle, dihedral=dihedral, n_angles=n_angles, n_dihedrals=n_dihedrals,
            monomer_id=monomer_id, program=program, version=version, lx=lx, ly=ly, lz=lz, xy=xy, xz=xz, yz=yz,
            n_impropers=0, improper=improper
        )
    )
    o.close()
    logger.info(f'Successfully generate CG simulation box [{lx},{ly},{lz}] with {n_atoms} particles.')
    return

