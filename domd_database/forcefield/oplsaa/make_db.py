import sys

from misc.logger import logger

import json
import os
import pickle
import re

from tqdm import tqdm

from domd_forcefield.functions import atom_stats, bonded_hash, improper_hash
from domd_forcefield.oplsaa.database import OplsDB

this_dir, this_file = os.path.split(__file__)

this_dir, this_filename = os.path.split(__file__)

proxies = {
    'https': 'socks5://localhost:7890',
    'http': 'socks5://localhost:7890',
}

bond_hashes = set()
angle_hashes = set()
dih_hashes = set()
imp_hashes = set()
atom_hashes = set()
def generate_feature_defn(fpath=os.path.join(this_dir, "data", "STaGE_opls_tomoltemplate_opls.txt")):
    info = {}
    feature_defn = ''
    with open(fpath, 'r') as infile:
        feat_index = 0
        for line in [line.strip() for line in infile if line.strip()]:
            if line[0] != '*':
                el, atomname, typename, patt, lttype, chg, desc = [el.strip() for el in line.split("|")]
                info[typename] = (el, patt)
                if el == 'nom':
                    continue
                feature_defn += \
                    """
                    DefineFeature {0} {1}
                    Family {2}{3}
                    EndFeature""".format(typename, patt, feat_index, atomname)
                # add new charge dictionary entry
                # use gromacs type as the key
                feat_index += 1
    return info, feature_defn


def update_nonbonded(db, url=None):
    # if not url:
    #    url = "https://raw.githubusercontent.com/gromacs/gromacs/bd791c7cba317d8d59c86c2edcc923d6263f8d0a" \
    #          "/share/top/oplsaa.ff/ffnonbonded.itp"

    # response = requests.get(url, proxies=proxies)
    # data = re.split('\n', response.text)
    fn = os.path.join(this_dir, "data", "ffnonbonded.itp")
    data = re.split('\n', open(fn, 'r').read())
    info, _ = generate_feature_defn()
    ret = {}
    for line in data:
        if not line:
            continue
        if line[0] in ['[', '#', ';']:
            continue
        line = re.split(r'\s+', line)
        while '' in line:
            line.remove('')
        name = line[0]
        if 'opls' not in name:
            continue
        ret[name] = line
    for k in info:
        line = ret.get(k)
        if line:
            el, patt = info.get(k)
            db.add("atom_gmx", name=f"gmx_{k}", bond_type=line[1].strip(), atomic_num=int(line[2]),
                   mass=float(line[3]), charge=float(line[4]), ptype=line[5], sigma=float(line[6]),
                   epsilon=float(line[7]),
                   hash='nil', smarts='nil', element=el, desc=patt)
        "name bond_type atomic_num mass charge ptype sigma epsilon"


def update_bonded(db, url=None):
    # if not url:
    #    url = "https://raw.githubusercontent.com/gromacs/gromacs/" \
    #          "bd791c7cba317d8d59c86c2edcc923d6263f8d0a/share/top/oplsaa.ff/ffbonded.itp"
    # response = requests.get(url)
    # data = re.split('\n', response.text)
    fn = os.path.join(this_dir, "data", "ffbonded.itp")
    data = re.split('\n', open(fn, 'r').read())
    bond_reg = r'^\s+[^\s]*?\s+[^\s]*?\s+[1-9]+\s+.*$'
    angle_reg = r'^\s+[^\s]*?\s+[^\s]*?\s+[^\s]*?\s+[1-9]+\s+.*$'
    dihedral_reg = r'^\s+[^\s]*?\s+[^\s]*?\s+[^\s]*?\s+[^\s]*?\s+[1-9]+\s+.*$'
    ret = {}
    for line in data:
        line_l = re.split(r'\s', line)
        while '' in line_l:
            line_l.remove('')
        if re.search(bond_reg, line):
            if int(line_l[2]) == 2:
                line_l.append("0")
            # add constraints to bond, last parameter is 0.
            # classified by func type, 2.
            name = f"{line_l[0]}-{line_l[1]}"
            ftype = int(line_l[2])
            param = json.dumps([float(line_l[3]), float(line_l[4])])
            bond_hashes.add(name)
            db.add('bonded_gmx', name=name, ftype=ftype, param=param, type='bond')
        elif re.search(angle_reg, line):
            name = f"{line_l[0]}-{line_l[1]}-{line_l[2]}"
            ftype = int(line_l[3])
            param = json.dumps([float(line_l[4]), float(line_l[5])])
            angle_hashes.add(name)
            db.add('bonded_gmx', name=name, ftype=ftype, param=param, type='angle')
        elif re.search(dihedral_reg, line):
            # ret[f"{line_l[0]}-{line_l[1]}-{line_l[2]}-{line_l[3]}"] = Opls_dihedral(
            #    *line_l[:4], int(line_l[4]), np.array([float(_) for _ in line_l[5:11]])
            # )
            name = f"{line_l[0]}-{line_l[1]}-{line_l[2]}-{line_l[3]}"
            ftype = int(line_l[4])
            param = json.dumps([float(_) for _ in line_l[5:11]])
            dih_hashes.add(name)
            db.add('bonded_gmx', name=name, ftype=ftype, param=param, type='dihedral')

def bonded_boss(db):
    fn = os.path.join(this_dir, "data", "boss_bonded.sb")
    for l in tqdm(open(fn, 'r')):
        line = re.split(r'\s+', l)
        while '' in line:
            line.remove('')
        name = line[0]
        name_l = re.split('-', name)
        while '' in name_l:
            name_l.remove('')
        if '?' in name:
            is_rule = True
        else:
            is_rule = False
        if len(name_l) == 2:
            ftype, r0, k = int(line[1]), float(line[2]), float(line[3])
            param = json.dumps([r0, k])
            bond_hashes.add(name)
            db.add('bonded_boss', name=name, ftype=ftype, param=param, type='bond', is_rule=is_rule)
        if len(name_l) == 3:
            ftype, th0, k = int(line[1]), float(line[2]), float(line[3])
            param = json.dumps([th0, k])
            angle_hashes.add(name)
            db.add('bonded_boss', name=name, ftype=ftype, param=param, type='angle', is_rule=is_rule)
        if len(name_l) == 4:
            ftype = int(line[1])
            if ftype != 4:
                c0, c1, c2, c3, c4, c5 = float(line[2]), float(line[3]), float(line[4]),\
                    float(line[5]), float(line[6]), float(line[7])
                param = json.dumps([c0, c1, c2, c3, c4, c5])
                dih_hashes.add(name)
                db.add('bonded_boss', name=name, ftype=ftype, param=param, type='dihedral', is_rule=is_rule)
            elif ftype == 4:
                c0, c1, c2 = float(line[2]), float(line[3]), float(line[4])
                param = json.dumps([c0, c1, c2])
                imp_hashes.add(name)
                db.add('bonded_boss', name=name, ftype=ftype, param=param, type='improper', is_rule=is_rule)

def lgp_data(db):
    fn = os.path.join(this_dir, "data", "ligpargen", "AllData.pkl")
    itp_files = pickle.load(open(fn, 'rb'))
    c = 0
    radii = [2, 3]

    for itp in tqdm(itp_files):
        mol = itp[0]
        itp_data = itp[1]
        kl = list(itp_data.keys())
        if mol.GetNumAtoms() != len([k for k in kl if isinstance(k, int)]):
            continue
        c += 1

        for radius in radii:
            type_info = atom_stats(mol, radius=radius)
            # chg_meta = atom_stats(mol, radius=radius)
            bond_types = {}
            for aid in type_info:
                atom = mol.GetAtomWithIdx(aid)
                element = type_info[aid][1]
                assert element == atom.GetSymbol()
                type_hash = type_info[aid][3]
                type_smx = type_info[aid][2]
                itp_atom = itp_data[aid]
                bond_type = itp_atom[0]
                charge = itp_atom[1]
                sigma = itp_atom[2]
                epsilon = itp_atom[3]
                ptype = itp_atom[4]
                bond_types[aid] = bond_type
                if type_hash not in atom_hashes:
                    atom_hashes.add(type_hash)

                    if len(atom.GetSymbol()) == 1:
                        assert atom.GetSymbol() == bond_type[0]
                    if radius <= 3:
                        db.add('atom_boss', uniq=True, name='boss_opls_xxx', ptype=ptype, epsilon=epsilon,
                               charge=charge, sigma=sigma, bond_type=bond_type, hash=type_hash,
                               element=element, mass=atom.GetMass(), atomic_num=atom.GetAtomicNum(), smarts=type_smx)
                    if radius >= 3:
                        db.add('charge_boss', uniq=True, charge=charge, hash=type_hash,
                               element=element, smarts=type_smx)
            if radius not in [3]:
                continue

            # The fucking bonded params are highly repeated!
            # Only use hash, no name as hash shit
            for k in itp_data:
                if isinstance(k, int):
                    continue
                    ######################################
                if len(k) == 2:
                    bond = itp_data[k]
                    bond_name = f"{bond_types[k[0]]}-{bond_types[k[1]]}"
                    ftype = bond[0]
                    r0 = bond[1]
                    k0 = bond[2]
                    hash = bonded_hash([type_info[k[0]][3], type_info[k[1]][3]])
                    #hash = bond_name
                    if hash in bond_hashes:
                        continue
                    bond_hashes.add(hash)
                    param = json.dumps([float(_) for _ in [r0, k0]])
                    db.add("bonded_boss", uniq=True, name=bond_name, ftype=ftype, param=param, type='bond',
                           hash=hash, is_rule=False)
                if len(k) == 3:
                    angle = itp_data[k]
                    angle_name = f"{bond_types[k[0]]}-{bond_types[k[1]]}-{bond_types[k[2]]}"
                    ftype = angle[0]
                    t0 = angle[1]
                    k0 = angle[2]
                    hash = bonded_hash([type_info[k[0]][3], type_info[k[1]][3], type_info[k[2]][3]])
                    #hash = angle_name
                    if hash in angle_hashes:
                        continue
                    angle_hashes.add(hash)
                    param = json.dumps([float(_) for _ in [t0, k0]])
                    db.add("bonded_boss", uniq=True, name=angle_name, ftype=ftype, param=param, type='angle',
                           hash=hash, is_rule=False)

                if len(k) == 4:
                    dih = itp_data[k]
                    dih_name = f"{bond_types[k[0]]}-{bond_types[k[1]]}-{bond_types[k[2]]}-{bond_types[k[3]]}"
                    ftype = dih[0]
                    if ftype != 4:# dih[-1] == 'dihedral':
                        hash = bonded_hash([type_info[k[0]][3], type_info[k[1]][3],
                                            type_info[k[2]][3], type_info[k[3]][3]])
                        #hash = dih_name
                        if hash in dih_hashes:
                            continue
                        dih_hashes.add(hash)
                        param = json.dumps([float(_) for _ in dih[1:7]])
                        db.add("bonded_boss", uniq=False, name=dih_name, ftype=ftype, param=param,
                               type='dihedral', hash=hash, is_rule=False)

                    if ftype == 4:
                        hash = improper_hash(type_info[k[1]][3])
                        #hash = dih_name
                        if hash in imp_hashes:
                            continue
                        imp_hashes.add(hash)
                        param = json.dumps([float(_) for _ in dih[1:3]])
                        db.add("improper_boss", uniq=False, name=dih_name, ftype=ftype, param=param,
                               type='improper', hash=hash, is_rule=False)



if __name__ == "__main__":
    logger.setLevel(20)
    db = OplsDB(os.path.join(this_dir, 'data', 'opls.db'), create=True)
    update_nonbonded(db)
    update_bonded(db)
    bonded_boss(db)
    lgp_data(db)

    # print(db.stats())
    # fn = os.path.join(this_dir, "files", "ligpargen_db", "itps.pkl")
    # types = pickle.load(open(fn, 'rb'))
    # itps = pickle.load(open(fn, 'rb'))
    # print(itps[0]['dihedrals'])
