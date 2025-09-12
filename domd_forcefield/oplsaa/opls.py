import collections
import re
import shutil
import tempfile
import uuid
from itertools import permutations
from openbabel import openbabel as ob
from random import random
from typing import Any, Union, List

from rdkit import Chem
import tqdm

from domd_forcefield.forcefield import CustomRule, ForceField
from domd_forcefield.functions import improper_hash, bonded_hash, atom_stats
from domd_forcefield.oplsaa.database import OplsBonded, OplsAtom, Rule, OplsDB
from domd_forcefield.oplsaa.ml import MLModel
from domd_forcefield.oplsaa.opls_db import opls_db
from domd_forcefield.oplsaa.opls_types import OplsAtomTypes, OplsBondedTypes
from misc.draw import print_mol_ascii
from misc.logger import logger


def gmx_typing(gmx_rules: list[Rule], ob_atom: ob.OBAtom, atom_info: str) -> Union[OplsAtom, None]:
    _opls_atom_gmx = None
    for rule in gmx_rules:
        if ob_atom.MatchesSMARTS(rule.desc):
            _opls_atom_gmx = rule.opls_atom
            logger.debug(f"This is OPLS-aa type finder, atom "
                         f"{atom_info} get {_opls_atom_gmx.name}, "
                         f"{_opls_atom_gmx.bond_type} from rule {rule.desc}.")
            break
    return _opls_atom_gmx


def custom_typing(custom_rules: list[CustomRule],
                  molecule: Union[Chem.Mol, Chem.RWMol], atom: int,
                  atom_info: str) -> Union[OplsAtom, None]:
    _opls_atom_cus = None
    for _i, cr in enumerate(custom_rules):
        # print(Chem.MolToSmiles(molecule))
        _opls_atom_cus = cr(molecule, atom)
        if _opls_atom_cus.name != 'Opls_nfd':
            logger.debug(f"This is OPLS-aa type finder, atom "
                         f"{atom_info} get {_opls_atom_cus.name} from {cr.name} custom rule")
            break  # the first custom_rule
    return _opls_atom_cus


def custom_bonding(custom_rules: list[CustomRule], molecule: Union[Chem.Mol, Chem.RWMol],
                   query: tuple[int, int]) -> Union[OplsBonded, None]:
    _bond = None
    for _i, rule in enumerate(custom_rules):
        _bond: Union[OplsBonded, None] = rule(molecule, query)
        if _bond:  # The 1st custom rule for bonds
            logger.debug(f"This is OPLS-aa bonded finder, found bond "
                         f"{query} from {_i}th custom rule {rule.name}")
            break
    return _bond


def custom_torsion(custom_rules: list[CustomRule], molecule: Union[Chem.Mol, Chem.RWMol],
                   query: tuple[int, int, int, int]) -> tuple[Any, Any]:
    torsion, perm = None, None
    for _i, rule in enumerate(custom_rules):
        # print(nbr_i.GetIdx(),atom_i.GetIdx(),atom_j.GetIdx(),nbr_j.GetIdx())
        dih, _perm = rule(molecule, query)
        if dih:  # The 1st custom rule for bonds
            logger.debug(f"This is OPLS-aa bonded finder, found dihedral for "
                         f", {_perm} from {rule.name} custom rule.")
            torsion = dih
            perm = _perm
            dih.idx = perm
            break
    return torsion, perm


def custom_angle(custom_rules: list[CustomRule], molecule: Union[Chem.Mol, Chem.RWMol],
                 query: tuple[int, int, int]) -> tuple[Any, Any]:
    ni, i, nj = query
    ang, _perm = None, None
    for _i, rule in enumerate(custom_rules):
        ang, _perm = rule(molecule, query)

        if ang:  # The 1st custom rule for bonds
            ang.idx = _perm
            logger.debug(f"This is OPLS-aa bonded finder, found for "
                         f" {ni}-{i}-{nj} from {_i}th custom rule.")
            break
    return ang, _perm


def custom_improper(custom_rules: list[CustomRule], molecule: Union[Chem.Mol, Chem.RWMol],
                    query: int, perm: tuple[int,int,int]) -> tuple[Any, Any]:
    _opls_improper, _perm = None, None
    j = query
    for _i, rule in enumerate(custom_rules):
        #_opls_improper, _perm = rule(molecule, query)
        _opls_improper = rule(molecule, (j,'imp'))
        i, k, l = perm
        #if j != _perm[1]:
        #    logger.error(f"The custom improper rule {rule.name} "
        #                 f"is possibly incorrect, the center atom {j} should always "
        #                 f"be second in permutation {_perm}!")
        _perm = (perm[0], j, perm[1], perm[2])
        if _opls_improper:
            _opls_improper.idx = _perm
            logger.debug(f"This is custom improper finder {rule.name}, "
                         f"get improper for {i}-{j}-{k}-{l}: {_opls_improper}")
            break
    return _opls_improper, _perm


class OplsFF(ForceField):
    def __init__(self, name='oplsaa', gmx_rules: Union[List[Rule], None] = None,
                 database: Union[OplsDB, None] = None, custom_typing: Union[List[CustomRule], None] = None,
                 custom_bonding: Union[List[CustomRule], None] = None,
                 custom_angles: Union[List[CustomRule], None] = None,
                 custom_dihedrals: Union[List[CustomRule], None] = None,
                 custom_impropers: Union[List[CustomRule], None] = None):
        super().__init__(name)
        self.db = database
        self.gmx_rules = gmx_rules
        self.cache = collections.defaultdict(dict)
        self.atoms = OplsAtomTypes()
        self.bond = OplsBondedTypes(name='bond')
        self.angle = OplsBondedTypes(name='angle')
        self.torsion = OplsBondedTypes(name='torsion')
        self.improper = OplsBondedTypes(name='improper')
        self.bond_rules = None
        self.angle_rules = None
        self.dihedral_rules = None
        self.ff_params = ['atoms', 'bond', 'angle', 'torsion', 'improper']
        self.custom_typing = custom_typing
        self.custom_bonding = custom_bonding
        self.custom_angles = custom_angles
        self.custom_dihedral = custom_dihedrals
        self.custom_improper = custom_impropers
        self.molecule = None
        self.molecule_charge = 0.
        self.all_params = {'atoms': {}, 'bonds': {}, 'angles': {}, 'dihedrals': {}, 'impropers': {}}
        # if self.db:
        # self.bond_rules = opls_db.bonded_rules('bond')
        # self.angle_rules = opls_db.bonded_rules('angle')
        # self.dihedral_rules = opls_db.bonded_rules('dihedral')

    def collect(self):
        logger.info("Collecting OPLS data")
        s = len(self.atoms) + len(self.bond) + len(self.angle) + len(self.torsion) + len(self.improper)
        for key in self.atoms.keys():
            self.all_params['atoms'][key] = self.atoms[key]
        for key in self.bond.keys():
            self.all_params['bonds'][key] = self.bond[key]
        for key in self.angle.keys():
            self.all_params['angles'][key] = self.angle[key]
        for key in self.torsion.keys():
            self.all_params['dihedrals'][key] = self.torsion[key]
        for key in self.improper.keys():
            self.all_params['impropers'][key] = self.improper[key]
        num_all = sum([len(self.all_params[sect]) for sect in self.all_params.keys()])
        assert num_all == s, "The intersections should be 0."

    def charge_calibration(self, charge_method: CustomRule):
        r"""Please do this after parameterize.
        :param charge_method:
        :return:
        """
        if self.molecule is None:
            logger.error(f"Molecule is not defined!")
            return
        for atom in self.molecule.GetAtoms():
            charge = charge_method(self.molecule, atom.GetIdx())
            self.atoms[atom.GetIdx()].charge = charge
        logger.info("Charge calibration is done:")
        self.stats()

    def parameterize(self, molecule: Union[Chem.Mol, Chem.RWMol],
                     use_cache: bool = True, boss_radius: Union[None, int] = 3,
                     use_cache_in_custom: bool = False, custom_rules: str = 'bonded'):
        r"""
        The prediction sequence is custom (ML) -> gmx (overwrite if use) -> boss (overwrite if use).
        Or Only ML can be applied.
        """
        self.molecule = molecule
        self.molecule_charge = 0.
        # Typing, preparing work:
        ob_mol: Union[None, ob.OBMol] = None
        env_hashes: dict[int, str] = {}
        atom_infos = {}

        _skip_gmx_atom = False
        _skip_boss_atom = False
        _skip_gmx_bonded = False
        _skip_boss_bonded = False

        if custom_rules == 'bonded':
            _skip_boss_bonded = True
            _skip_gmx_bonded = True

        if custom_rules == 'all':  # use this to skip all database/rule searches
            _skip_gmx_atom = True
            _skip_boss_atom = True
            _skip_gmx_bonded = True
            _skip_boss_bonded = True

        # build env hash
        logger.debug(f"Building atom environment hash.")
        if molecule.GetNumAtoms() != 1:
            atom_meta: dict[int, list] = atom_stats(molecule, boss_radius)
            for k in atom_meta:
                env_hashes[k] = atom_meta[k][-1]
        else:  # single atom molecule
            atom = molecule.GetAtomWithIdx(0)
            env_hashes[0] = f"{atom.GetSymbol()}"
            atom_meta = {}
            atom_meta[atom.GetIdx()] = [atom.GetIdx(), atom.GetSymbol(), Chem.MolToSmiles(molecule), env_hashes[0]]

        # turn off gmx rules and boss if database not provided
        use_gmx = use_boss = True
        if self.db is None:
            logger.warning("Database is not provided, boss rules are turned off.")
            use_boss = False

        if self.gmx_rules is None:
            logger.warning("gmx_rules is not provided, gmx rules are turned off.")
            use_gmx = False

        if use_gmx:
            logger.info("GMX RULES are applied, transform rdmol to obmol.")
            s = Chem.MolToSmiles(molecule)
            tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                   suffix=uuid.uuid4().__str__() + f'{random()}'
                                                          + uuid.uuid4().__str__())
            tmp_file.close()
            ob_conv = ob.OBConversion()
            ob_conv.SetInFormat("pdb")
            ob_mol = ob.OBMol()
            Chem.MolToPDBFile(molecule, filename=tmp_file.name, flavor=4)
            ob_conv.ReadFile(ob_mol, tmp_file.name)
            # os.system('sleep 5')
            # os.remove(tmp_file.name)
            try:
                shutil.rmtree(tmp_file.name)
            except:
                # os.remove(tmp_file.name)
                pass
            # gmx_rules: list[Rule] = self.db.rules

        atom: Chem.Atom
        missing_atom_idx = []
        for atom in tqdm.tqdm(molecule.GetAtoms(), total=molecule.GetNumAtoms(), desc='searching atom type'):
            not_found_atom = OplsAtom(name="Opls_nfd", bond_type=f"{atom.GetSymbol()}_Opls_nfd",
                                      smarts=f"{atom.GetSmarts()}", atomic_num=atom.GetAtomicNum(),
                                      hash='nil', mass=atom.GetMass(), ptype='2', sigma=0, epsilon=0,
                                      charge=0, element=atom.GetSymbol())
            atom_info = atom_meta[atom.GetIdx()][:-1]
            atom_info_str = ", ".join([str(_) for _ in atom_meta[atom.GetIdx()][:-1]])
            atom_infos[atom.GetIdx()] = atom_info_str
            env_hash = env_hashes.get(atom.GetIdx())
            found_p = False

            if use_cache_in_custom:
                _atom_type_cus = self.cache['atom_type_cus'].get(env_hash)
                if _atom_type_cus is not None:
                    self.atoms[atom.GetIdx()] = _atom_type_cus
                    logger.warning(f"This is OPLS-aa type finder, atom "
                                   f"{atom_info} get {_atom_type_cus.name} from custom rule"
                                   f" in cache!")
                    continue  # Found in cache, go

            if use_cache:
                opls_atom_gmx_ = self.cache['atoms_gmx'].get(env_hash)
                opls_atom_boss_ = self.cache['atoms_boss'].get(env_hash)

                opls_atom = opls_atom_gmx_ or opls_atom_boss_

                if opls_atom is not None:
                    logger.debug(f"This is cache, found atom {atom.GetIdx()}, {atom.GetSymbol()} {opls_atom}")
                    self.atoms[atom.GetIdx()] = opls_atom
                    continue  # Found in cache, go

            # not found in cache, try custom -> gmx (if used, overwrite) -> boss (if used, overwrite)
            if self.custom_typing is not None:  # Use custom
                _atom_type_cus = custom_typing(self.custom_typing, molecule, atom.GetIdx(), atom_info)
                self.atoms.charge[atom.GetIdx()] = _atom_type_cus.charge
                if _atom_type_cus is not None:
                    self.atoms[atom.GetIdx()] = _atom_type_cus
                    # custom rules creates cache
                    self.cache['atoms_cus'][env_hash] = _atom_type_cus
                    found_p = True

            if use_gmx and (not _skip_gmx_atom):
                ob_atom: ob.OBAtom = ob_mol.GetAtomById(atom.GetIdx())
                _atom_type_gmx = gmx_typing(self.gmx_rules, ob_atom=ob_atom, atom_info=atom_info_str)
                if _atom_type_gmx is not None:
                    logger.debug(f"This is OPLS-aa type finder, gmx rule finds atom {atom.GetIdx()}, {atom.GetSymbol()}"
                                 f" {_atom_type_gmx} from gmx rule")
                    self.atoms[atom.GetIdx()] = _atom_type_gmx
                    self.cache['atoms_gmx'][env_hash] = _atom_type_gmx
                    found_p = True

            if use_boss and (not _skip_boss_atom):  # not found by gmx
                _opls_atom_boss = self.db.search(target='atom_boss', hash=env_hash)
                if _opls_atom_boss:
                    logger.debug(f"This is OPLS-aa type finder, atom "
                                 f"{atom_info} get {_opls_atom_boss.name}, "
                                 f"{_opls_atom_boss} from database.")
                    if _opls_atom_boss.element != atom.GetSymbol():
                        logger.error(Exception(f"This is OPLS-aa type finder, atom {atom_info} is {atom.GetSymbol()} "
                                               f"but the hash considers with {_opls_atom_boss} in database "
                                               f"which is {_opls_atom_boss.element}"))
                        raise ValueError("Element not match error!")
                    self.atoms[atom.GetIdx()] = _opls_atom_boss
                    found_p = True

            # not found after all searching methods, mark as missing.
            if not found_p:
                self.atoms[atom.GetIdx()] = not_found_atom
                self.atoms.missing_atoms[atom.GetIdx()] = not_found_atom
                if not self.atoms.missing_atom_types.get(atom.GetSymbol()):
                    self.atoms.missing_atom_types[atom.GetSymbol()] = set()
                self.atoms.missing_atom_types[atom.GetSymbol()].add(atom_info[-1])
                s = print_mol_ascii(Chem.MolFromSmiles(atom_info[-1], sanitize=False))
                logger.error(f"This is OPLS-aa type finder, atom "
                             f"{atom.GetIdx()}, {atom.GetIdx()} in {atom_info[-1]} is not found!\n"
                             f"HASH: \n{s}\n {atom_info[-1]}, {env_hash}")
                missing_atom_idx.append(atom.GetIdx())

        # find bonded
        # find bonds
        bond: Chem.Bond
        missing_bond = []  # get atomic idx i, j in missing bonds
        for bond in tqdm.tqdm(molecule.GetBonds(), total=len(molecule.GetBonds()), desc='searching bond'):
            _bond: Union[None, OplsBonded] = None
            bi = bond.GetBeginAtomIdx()
            bj = bond.GetEndAtomIdx()
            bond_type_1 = self.atoms[bi].bond_type
            bond_type_2 = self.atoms[bj].bond_type
            name1 = "-".join([bond_type_1, bond_type_2])
            name2 = "-".join([bond_type_2, bond_type_1])
            atom_i: Chem.Atom = bond.GetBeginAtom()
            atom_j: Chem.Atom = bond.GetEndAtom()

            # Start searching
            # custom first
            if self.custom_bonding is not None:
                _bond = custom_bonding(self.custom_bonding, molecule, (bi, bj))
                if _bond:
                    self.bond[(bi, bj)] = _bond

            # Finding by rules: custom -> rules (if applied, overwrite)
            _skip_bond = False
            for _atom in [atom_i, atom_j]:
                if self.atoms[_atom.GetIdx()].name == 'Opls_nfd':
                    logger.error(f"{_atom.GetIdx()}, {_atom.GetSymbol()} not found bond_type for gmx or boss!")
                    _skip_bond = True
                if self.atoms[_atom.GetIdx()].name.endswith('_ML'):
                    # make sure you custom atom-typing rules either give sound opls-bond-type, or
                    # add a mark to skip the matching proc.
                    logger.warning(f"{_atom.GetIdx()}, {_atom.GetSymbol()} not found bond_type for gmx or boss, but use ML prediction.")
                    _skip_bond = True
            _skip_bond = _skip_gmx_bonded and _skip_boss_bonded
            if (use_gmx or use_boss) and (not _skip_bond):
                # make a record when there is gmx/boss non-typed opls_atom
                # for a bonded gmx/boss parameters
                # however, since I have set '!bond_type' for gmx/boss non-typed
                # atoms, this should not be a problem
                _bond_gmx = _bond_boss = None
                # search gmx rules first, if boss, overwrite it.
                if use_gmx and (not _skip_gmx_bonded):
                    _bond_gmx = self.db.search("bonded_gmx", name=name1, stype='bond') or \
                                self.db.search("bonded_gmx", name=name2, stype='bond')
                    if _bond_gmx:
                        logger.debug(f"GMX bond found for {(bi, bj)}, {name1}, {_bond_gmx}")
                if use_boss and (not _skip_boss_bonded):
                    bond_type_1 = self.atoms[bi].bond_type
                    bond_type_2 = self.atoms[bj].bond_type
                    name1 = "-".join([bond_type_1, bond_type_2])
                    name2 = "-".join([bond_type_2, bond_type_1])
                    name1 = re.sub(r'_\d+', '', name1)
                    name2 = re.sub(r'_\d+', '', name2)
                    _hashes = [env_hashes[bi], env_hashes[bj]]
                    _hash = bonded_hash(_hashes)
                    _bond_from_hash = None
                    _bond_boss = self.db.search("bonded_boss", name=name1, stype='bond') or \
                                 self.db.search("bonded_boss", name=name2, stype='bond')
                    if _bond_boss:
                        logger.debug(f"BOSS bond found for {(bi, bj)}, {name1}, {_bond_boss}")

                    if not (_bond_from_hash or _bond_boss):
                        bond_hash = self.db.search('bonded_boss', hash=_hash, stype='bond')
                        if bond_hash:
                            logger.debug(f"Find bond for {bi}-{bj} as {bond_hash}")
                            _bond_boss = bond_hash

                    if self.bond_rules:
                        for rule in self.bond_rules:
                            regex = rule.name.replace('?', '.{1, 2}')
                            if re.search(rf"{regex}", name1) or re.search(rf"{regex}", name2):
                                _bond_boss = rule
                                logger.debug(f"BOSS bond for {(bi, bj)}, {name1}, {_bond_boss}"
                                             f" from rule {rule}")
                                break
                    if _bond_boss:
                        logger.debug(f"BOSS bond found for {(bi, bj)}, {name1}, {_bond_boss}")

                _bond = _bond_boss or _bond_gmx
                if _bond:
                    logger.debug(f"Bond for {(bi, bj)} {atom_i.GetSymbol()}-{atom_j.GetSymbol()} found: "
                                 f"{_bond}")
                    self.bond[(bi, bj)] = _bond
            # End of searching
            # handle missing bonds:

            if _bond is None:  # not find in db or custom rules
                if name1 in self.bond.missing:
                    self.bond.missing[name1].append((bi, bj))
                elif name2 in self.bond.missing:
                    self.bond.missing[name2].append((bi, bj))
                else:
                    self.bond.missing[name1] = [(bi, bj)]
                logger.warning(f"Bond between {bi}-{bj}, {name1} is not found.")
                missing_bond.append((bi, bj))
            else:
                self.bond[(bi, bj)] = _bond

        # make each searching part independent
        # find torsions via iterating over bonds.
        missing_torsion = []
        for bond in tqdm.tqdm(molecule.GetBonds(), total=len(molecule.GetBonds()), desc='searching torsions'):
            bi = bond.GetBeginAtomIdx()
            bj = bond.GetEndAtomIdx()
            atom_i: Chem.Atom = bond.GetBeginAtom()
            atom_j: Chem.Atom = bond.GetEndAtom()
            for nbr_i in atom_i.GetNeighbors():
                # a bond (i, j) and neighbor of i and j define a proper dihedral
                ni = nbr_i.GetIdx()
                if ni == bj:
                    continue
                for nbr_j in atom_j.GetNeighbors():
                    nj = nbr_j.GetIdx()
                    if nj == bi or ni == nj:
                        # avoid 3-ring
                        continue
                    # construct dihedral
                    _skip_torsion = False
                    dihedral_idx = [self.atoms[ni].bond_type,
                                    self.atoms[bi].bond_type,
                                    self.atoms[bj].bond_type,
                                    self.atoms[nj].bond_type]
                    name1 = "-".join(dihedral_idx)
                    if "Opls_nfd" in name1 or "_ML" in name1:
                        _skip_torsion = True
                    name2 = "-".join(dihedral_idx[::-1])
                    _all_names = [name1, name2,
                                  "-".join(dihedral_idx[:3]) + '-X',
                                  "-".join(dihedral_idx[::-1][1:]) + '-X',
                                  'X-' + "-".join(dihedral_idx[1:3]) + '-X',
                                  "X-" + "-".join(dihedral_idx[::-1][1:3]) + '-X']
                    _all_perms = [(ni, bi, bj, nj),
                                  (nj, bj, bi, ni),
                                  (nj, bj, bi, ni),
                                  (ni, bi, bj, nj),
                                  (nj, bj, bi, ni)]
                    _dih_hashes = [env_hashes[ni], env_hashes[bi], env_hashes[bj], env_hashes[nj]]
                    _dih_hash_str = bonded_hash(_dih_hashes)
                    dih = None
                    _perm = _all_perms[0]  # original permutation
                    orig_perm = _all_perms[0]
                    # Starg finding...
                    # Custom first
                    if self.custom_dihedral is not None:
                        # print(_perm)
                        dih, perm = custom_torsion(self.custom_dihedral, molecule, _perm)
                        if dih is not None:
                            dih.idx = perm
                            self.torsion[orig_perm] = dih

                    if _skip_torsion:
                        logger.info(f"{name1} contains invalid opls bond-types, skipping gmx or boss rule-searching.")

                    _skip_torsion = _skip_gmx_bonded and _skip_boss_bonded

                    if (use_gmx or use_boss) and (not _skip_torsion):
                        # make a record when there is gmx/boss non-typed opls_atom
                        # for a bonded gmx/boss parameters
                        # however, since I have set '!bond_type' for gmx/boss non-typed
                        # atoms, this should not be a problem

                        if use_boss and (not _skip_boss_bonded):
                            dih_boss = self.db.search('bonded_boss', hash=_dih_hash_str, stype='dihedral')
                            if dih_boss is not None:
                                logger.debug(f"Found boss dihedral by hash {_dih_hash_str} with {_perm}: {dih_boss}")

                        # print(_all_perms)
                        for _name, _perm in zip(_all_names, _all_perms):
                            dih_gmx = dih_boss = None

                            if use_gmx and (not _skip_gmx_bonded):
                                dih_gmx = self.db.search("bonded_gmx", name=_name, stype='dihedral')
                                if dih_gmx is not None:
                                    logger.debug(f"Found gmx dihedral {_name} with {_perm}: {dih_gmx}")

                            if use_boss and (not _skip_boss_bonded):
                                _name = re.sub(r'_\d+', '', _name)
                                dih_boss = self.db.search('bonded_boss', name=_name, stype='dihedral')
                                if dih_boss is not None:
                                    logger.debug(f"Found boss dihedral by name {_name} with {_perm}: {dih_boss}")
                                if self.dihedral_rules:
                                    for rule in self.dihedral_rules:
                                        regex = rule.name.replace('?', '.{1, 2}')
                                        if re.search(rf"{regex}", _name):
                                            dih_boss = rule
                                            logger.debug(f"BOSS dihedral for {_perm}, {_name}"
                                                         f" from rule {rule}")
                                            break
                            dih_ = dih_boss or dih_gmx
                            if dih_:
                                dih_.idx = _perm  # _perm was in the loop
                                dih = dih_
                                break
                    ####
                    #### End of searching gmx or boss rules

                    # if not found above, try hash last
                    # Even the gmx or boss rules are skipped, we can still use hash
                    if use_boss and (dih is None) and (not _skip_torsion):  # find by hash at last
                        _hashes = [env_hashes[ni], env_hashes[bi], env_hashes[bj], env_hashes[nj]]
                        _hash = bonded_hash(_hashes)
                        _dih_from_hash = self.db.search('bonded_boss', hash=_hash, stype='dihedral')
                        if _dih_from_hash:
                            logger.debug(f"Find dihedral for {ni}-{bi}-{bj}-{nj} as {_dih_from_hash}")
                            dih = _dih_from_hash
                            _perm = (ni, bi, bj, nj)  # _perm is out-of-the permutations
                            dih.idx = _perm

                        if dih:
                            self.torsion[orig_perm] = dih

                    if not dih:
                        logger.warning(f"This is OPLS-aa bonded finder, Dihedral "
                                       f"{ni}-{bi}-{bj}-{nj} {name1} not found!")
                        if name1 in self.torsion.missing:
                            self.torsion.missing[name1].append((ni, bi, bj, nj))
                        elif name2 in self.torsion.missing:
                            self.torsion.missing[name2].append((ni, bi, bj, nj))
                        else:
                            self.torsion.missing[name1] = [(ni, bi, bj, nj)]
                        missing_torsion.append((ni, bi, bj, nj))
                    else:
                        self.torsion[orig_perm] = dih  # this is for sure.

        # find angles
        missing_angle = []
        for atom in tqdm.tqdm(molecule.GetAtoms(), total=len(molecule.GetAtoms()), desc='searching angles'):
            # any atom with >=2 bonds can be a center of an angle
            if len(atom.GetNeighbors()) >= 2:
                i = atom.GetIdx()
                for nbr_i in atom.GetNeighbors():
                    ni = nbr_i.GetIdx()
                    for nbr_j in atom.GetNeighbors():
                        nj = nbr_j.GetIdx()
                        if ni <= nj:
                            continue
                        # construct angle
                        _skip_angle = False
                        angle = [self.atoms[ni].bond_type, self.atoms[i].bond_type,
                                 self.atoms[nj].bond_type]
                        name1 = "-".join(angle)
                        name2 = "-".join(angle[::-1])
                        perm1 = (ni, i, nj)
                        perm2 = (nj, i, ni)
                        ang = None
                        _perm = perm1  # original permutation
                        orig_perm = perm1

                        if self.custom_angles is not None:
                            ang, perm = custom_angle(self.custom_angles, molecule, _perm)
                            if ang is not None:
                                ang.idx = perm
                            self.angle[orig_perm] = ang

                        _skip_angle = ("Opls_nfd" in name1) or ("_ML" in name1)
                        _skip_angle = _skip_boss_bonded and _skip_gmx_bonded
                        if _skip_angle:
                            logger.debug(
                                f"{_atom.GetIdx()}, {atom.GetSymbol()} not found bond_type for gmx or boss!")

                        if (use_gmx or use_boss) and (not _skip_angle):
                            # make a record when there is gmx/boss non-typed opls_atom
                            # for a bonded gmx/boss parameters
                            # however, since I have set '!bond_type' for gmx/boss non-typed
                            # atoms, this should not be a problem
                            for _name, _perm in zip([name1, name2], [perm1, perm2]):
                                ang_gmx = ang_boss = None
                                if use_gmx and (not _skip_gmx_bonded):
                                    ang_gmx = self.db.search('bonded_gmx', name=_name, stype='angle')

                                if use_boss and (not _skip_boss_bonded):
                                    _name = re.sub(r'_\d+', '', _name)
                                    ang_boss = self.db.search('bonded_boss', name=_name, stype='angle')

                                if self.angle_rules:
                                    for rule in self.angle_rules:
                                        regex = rule.name.replace('?', '.{1, 2}')
                                        if re.search(rf"{regex}", _name):
                                            ang_boss = rule
                                            logger.debug(f"BOSS angle for {_perm}, {_name}"
                                                         f" from rule {rule}")
                                            break
                                ang_ = ang_boss or ang_gmx
                                if ang_:
                                    ang_.idx = _perm
                                    ang = ang_
                                    break
                        #####
                        # End of gmx or boss searching
                        # Even the gmx or boss rules are skipped, we can still use hash
                        if use_boss and (ang is None) and (not _skip_angle):  # same as torsion
                            _hashes = [env_hashes[ni], env_hashes[i], env_hashes[nj]]
                            _hash = bonded_hash(_hashes)
                            ang_hash = self.db.search('bonded_boss', hash=_hash)
                            if ang_hash:
                                logger.debug(f"Find angle for {ni}-{i}-{nj} as {ang_hash}")
                                ang = ang_hash
                                _perm = (ni, i, nj)
                                ang.idx = _perm

                        if ang is None:
                            logger.warning(f"This is OPLS-aa bonded finder, Angle {ni}-{i}-{nj} {name1} not found!")
                            if name1 in self.angle.missing:
                                self.angle.missing[name1].append((ni, i, nj))
                            elif name2 in self.angle.missing:
                                self.angle.missing[name2].append((ni, i, nj))
                            else:
                                self.angle.missing[name1] = [(ni, i, nj)]
                            missing_angle.append((ni, i, nj))
                        else:
                            # To make sure
                            self.angle[orig_perm] = ang

        # make all fucking searching independently
        # to avoid the fucking continue problem
        for atom in tqdm.tqdm(molecule.GetAtoms(), total=len(molecule.GetAtoms()), desc='searching impropers'):
            # improper, for center atom is j in i-j-k-l
            # always 180 to keep planar between i-j-k and j-k-l
            # rules from gromacs and oplsaa.par of BOSS
            # if len(atom.GetNeighbors()) == 3:
            hyb = atom.GetHybridization()
            if hyb.name == 'SP2' and len(atom.GetNeighbors()) == 3:
                # sp2 atoms are always planar, instead of using number of neighbors only
                nbr_hashes = []
                for nbr in atom.GetNeighbors():
                    nbr_hashes.append(env_hashes[nbr.GetIdx()])

                # construct improper

                perms = list(permutations(atom.GetNeighbors()))
                perm: tuple = perms[0]
                perm_idx = tuple([_.GetIdx() for _ in perm])
                j = atom.GetIdx()
                name = "-".join(
                    [self.atoms[_atom.GetIdx()].name for _atom in [atom] + list(atom.GetNeighbors())])
                _skip_improper = "Opls_nfd" in name  # or "_ML" in name
                _skip_improper = _skip_gmx_bonded and _skip_boss_bonded
                if _skip_improper:
                    logger.debug(f"{'-'.join([str(_atom.GetIdx()) for _atom in [atom] + list(atom.GetNeighbors())])} "
                                 f"with "
                                 f"name {name}"
                                 f" has invalid bond_type for gmx or boss!")
                if self.custom_improper:
                    _opls_improper, _perm = custom_improper(self.custom_improper, molecule, atom.GetIdx(), perm_idx)
                    if _opls_improper is not None:
                        self.improper[_perm] = _opls_improper

                if (use_gmx or use_boss) and (not _skip_improper):
                    # make a record when there is gmx/boss non-typed opls_atom
                    # for a bonded gmx/boss parameters
                    # however, since I have set 'bond_type_XXX' for gmx/boss non-typed
                    # atoms, this should not be a problem
                    _found = False  # this is dumb, however I don't want to change this shit
                    for _perm in perms:
                        i = _perm[0].GetIdx()
                        k = _perm[1].GetIdx()
                        l = _perm[2].GetIdx()
                        dihedral = [self.atoms[i].bond_type, self.atoms[j].bond_type,
                                    self.atoms[k].bond_type, self.atoms[l].bond_type]
                        name = "-".join(dihedral)
                        d_str = '-'.join(dihedral)
                        if re.search(r'O.*-(C|C_2|C_3)-.*-.*', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 43.93200, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True  # this is dumb, however I don't want to change this shit
                            break
                        elif re.search(r'.*-NO-ON-NO', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 43.93200, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'N2-.*-N2-N2', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 43.93200, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'.*-N.+-.*-.*', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 4.18400, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'.*-N-.*-.*', d_str):  # oplsaa.par
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 10.46, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'.*-(CM|C=)-.*-.*', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 62.76000, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'.*-(CM|CB|CN|CV|CW|CR|CK|CQ|CS|C*)-.*-.*', d_str):
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 4.60240, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif re.search(r'.*-(CA)-.*-.*', d_str):  # oplsaa.par
                            self.improper[(i, j, k, l)] = OplsBonded(name=name, hash="nil", ftype=4,
                                                                     param=[180.0, 10.46, 2],
                                                                     type='improper', idx=[i, j, k, l])
                            _found = True
                            break
                        elif self.db.search('improper_boss', name=d_str, stype='improper'):
                            self.improper[(i, j, k, l)] = self.db.search('improper_boss',
                                                                         name=d_str, stype='improper')
                            self.improper[(i, j, k, l)].idx = [i, j, k, l]
                            _found = True
                            break

                # Even the gmx or boss rules are skipped, we can still use hash
                if use_boss and (not _skip_improper):
                    _hash = improper_hash(env_hashes[atom.GetIdx()])
                    _improper_hash = self.db.search('improper_boss', hash=_hash, stype='improper')
                    if _improper_hash:
                        logger.debug(f"Find improper for {[_.GetIdx() for _ in perm]} as {_improper_hash}")
                        self.improper[(perm[0].GetIdx(), j, perm[1].GetIdx(), perm[2].GetIdx())] = _improper_hash
                        continue

        sound_p = self.stats()
        if sound_p:
            logger.info("The force field is sound, collecting the force field parameters")
            self.collect()
        else:
            logger.info("The force field is not sound, skip collecting the force field parameters")

    def stats(self):
        num_sp2_atoms = 0
        num_angles = 0
        num_dihedrals = 0
        self.molecule_charge = 0
        for atom in self.molecule.GetAtoms():
            if atom.GetHybridization().name == 'SP2':
                num_sp2_atoms += 1
            if len(atom.GetNeighbors()) >= 2:
                for nbr_i in atom.GetNeighbors():
                    ni = nbr_i.GetIdx()
                    for nbr_j in atom.GetNeighbors():
                        nj = nbr_j.GetIdx()
                        if ni <= nj:
                            continue
                        num_angles += 1
            _opls_atom: OplsAtom = self.atoms[atom.GetIdx()]
            self.molecule_charge += _opls_atom.charge

        for bond in self.molecule.GetBonds():
            atom_i: Chem.Atom = bond.GetBeginAtom()
            atom_j: Chem.Atom = bond.GetEndAtom()
            bi = atom_i.GetIdx()
            bj = atom_j.GetIdx()
            for nbr_i in atom_i.GetNeighbors():
                # a bond (i, j) and neighbor of i and j define a proper dihedral
                ni = nbr_i.GetIdx()
                if ni == bj:
                    continue
                for nbr_j in atom_j.GetNeighbors():
                    nj = nbr_j.GetIdx()
                    if nj == bi or ni == nj:
                        # avoid 3-ring
                        continue
                    num_dihedrals += 1
        logger.info(f"Molecule {self.molecule} has\n"
                    f" {self.molecule.GetNumAtoms()} atoms"
                    f"\n {self.molecule.GetNumBonds()} bonds"
                    f"\n {num_dihedrals} dihedrals\n "
                    f"{num_angles} angles\n {num_sp2_atoms} SP2 atoms.")
        logger.info("Atom typing stats:")
        missing_atoms = self.atoms.stats()
        logger.info(f"Total Charge is {self.molecule_charge:.3f}")
        logger.info("Bonded typing stats:")
        missing_bonds = self.bond.stats()
        missing_angles = self.angle.stats()
        self.torsion.stats()
        self.improper.stats()
        return missing_atoms == missing_bonds == missing_angles == 0


gmx_rules = opls_db.rules
opls_ff_pargen = OplsFF(gmx_rules=gmx_rules, database=opls_db, custom_typing=[MLModel], custom_angles=[MLModel],
                        custom_dihedrals=[MLModel], custom_bonding=[MLModel])

if __name__ == '__main__':
    #mol = Chem.MolFromSmiles('C(F)CC#CCc1ccccc1')
    mol = Chem.MolFromSmiles('C(F)(F)(F)F')
    mol = Chem.AddHs(mol)
    opls_ff_pargen.parameterize(mol)
