import json
import uuid
from hashlib import sha256
from typing import Any
from typing import List, Union

from rdkit import Chem
from sqlalchemy import JSON, Integer
from sqlalchemy import String, Column
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import sessionmaker

from misc.logger import logger

periodic_tbl = Chem.GetPeriodicTable()


class Base(DeclarativeBase):
    pass


class OplsAtomChargeDB(Base):
    __tablename__ = "opls_charge"
    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    charge: Mapped[float] = mapped_column()
    element: Mapped[str] = mapped_column(String)
    smarts: Mapped[str] = mapped_column(String)

    def __repr__(self):
        return f"OplsAtomChargeDB(element={self.element}, charge={self.charge}, smarts={self.smarts}, hash={self.hash})"


class OplsAtomChargeBOSS(OplsAtomChargeDB):
    __mapper_args__ = {
        'polymorphic_identity': 'boss',
    }


class OplsAtomChargeGMX(OplsAtomChargeDB):
    __mapper_args__ = {
        'polymorphic_identity': 'gmx',
    }


class OplsAtomDB(Base):
    __tablename__ = "opls_atom"
    name: Mapped[str] = mapped_column(String)
    bond_type: Mapped[str] = mapped_column(String)
    smarts: Mapped[str] = mapped_column(String)
    hash: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    epsilon: Mapped[float] = mapped_column()
    sigma: Mapped[float] = mapped_column()
    charge: Mapped[float] = mapped_column()
    ptype: Mapped[str] = mapped_column(String)
    desc: Mapped[str] = mapped_column(String)
    is_rule: Mapped[bool] = mapped_column()
    atomic_num: Mapped[int] = mapped_column()
    mass: Mapped[float] = mapped_column()
    element: Mapped[str] = mapped_column()

    def __repr__(self) -> str:
        return f"OplsAtomDB(name={self.name!r}, bond_type={self.bond_type!r}, mass={self.mass!r}, " \
               f"smarts={self.smarts!r}, hash={self.hash!r}, element={self.element!r}, " \
               f"atomic_num={self.atomic_num!r}, " \
               f"epsilon={self.epsilon!r}, sigma={self.sigma!r}, charge={self.charge!r}, ptype={self.ptype!r}, " \
               f"desc={self.desc!r}, is_rule={self.is_rule!r})"


class OplsAtomBOSS(OplsAtomDB):
    __mapper_args__ = {
        'polymorphic_identity': 'boss_atom',
    }


class OplsAtomGMX(OplsAtomDB):
    __mapper_args__ = {
        'polymorphic_identity': 'gmx_atom',
    }


class OplsBondedDB(Base):
    __tablename__ = "opls_bonded"
    # keep name for a while, I don't think it is still necessary to search by name while we have hash.
    _id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, index=True)
    ftype: Mapped[int] = mapped_column()
    param: Mapped[str] = Column(JSON)
    hash: Mapped[str] = mapped_column(String, index=True)
    type: Mapped[str] = Column(String)
    is_rule: Mapped[bool] = mapped_column()
    _type: Mapped[str] = Column(String)
    __mapper_args__ = {
        'polymorphic_on': _type,
        'polymorphic_identity': 'opls_bonded',
    }

    def __repr__(self) -> str:
        return f"OplsBondedDB(name={self.name!r}, hash={self.hash!r}, type={self.type!r}, " \
               f"ftype={self.ftype!r}, param={self.param!r}, is_rule={self.is_rule!r})"


class OplsBondedGMX(OplsBondedDB, Base):
    __mapper_args__ = {
        'polymorphic_identity': 'gmx_bonded',
    }


class OplsBondedBOSS(OplsBondedDB, Base):
    __mapper_args__ = {
        'polymorphic_identity': 'boss_bonded',
    }


class OplsImproperBOSS(OplsBondedDB, Base):
    __mapper_args__ = {
        'polymorphic_identity': 'improper_boss',
    }


class OplsAtom(object):
    def __init__(self, name: str = 'nil', bond_type: str = 'nil', smarts: str = 'nil', element: str = 'nil',
                 hash: str = 'nil', charge: float = 0, epsilon: float = 0, sigma: float = 0,
                 ptype: str = 'A', mass: float = 0, atomic_num: int = 0, desc: str = "nil",
                 is_rule: bool = False):
        self.name = name
        self.bond_type = bond_type
        self.smarts = smarts
        self.hash = hash
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.ptype = ptype
        self.desc = desc
        self.element = element
        self.atomic_num = atomic_num
        self.mass = mass
        self.is_rule = is_rule
        self.type = 'atom'

    def __repr__(self):
        return f"OplsAtom(name={self.name}, bond_type={self.bond_type}, smarts={self.smarts}, " \
               f"hash={self.hash}, epsilon={self.epsilon}, sigma={self.sigma}, charge={self.charge}, " \
               f"ptype={self.ptype}, element={self.element}, atomic_num={self.atomic_num}, " \
               f"mass={self.mass})"


class OplsBonded(object):
    def __init__(self, name: str, hash: str, ftype: int, param: list, type: str, idx=None, is_rule: bool = False):
        if type not in {"angle", "bond", "dihedral", "improper"}:
            raise ValueError("The type of a bonded object should be `angle', `bond' or 'dihedral'.")
        self.idx = idx
        self.name = name
        self.hash = hash
        self.ftype = ftype
        self.param = param
        self.type = type
        self.is_rule = is_rule

    def __repr__(self):
        return f"OplsBonded(idx={self.idx}, name={self.name}, hash={self.hash}, " \
               f"ftype={self.ftype}, param={self.param}, type={self.type}, is_rule={self.is_rule})"


class Rule:
    def __init__(self, patt: Chem.Mol, atom: OplsAtom, desc: str):
        self.patt = patt
        self.opls_atom = atom
        self.desc = desc

    def __repr__(self):
        return f"Rule(patt={self.desc}, opls_atom={self.opls_atom})"


_data_types = dict(atom_gmx=OplsAtomGMX, atom_boss=OplsAtomBOSS, bonded_gmx=OplsBondedGMX,
                   bonded_boss=OplsBondedBOSS, charge_gmx=OplsAtomChargeGMX, charge_boss=OplsAtomChargeBOSS,
                   improper_boss=OplsImproperBOSS)
_prop_list = dict(atom={"name", "bond_type", "smarts", 'desc', "is_rule", "mass", "element",
                        "hash", "epsilon", "sigma", "charge", "ptype", "atomic_num"},
                  bonded={"name", "ftype", "hash", "param", "hash", "type", "is_rule"},
                  improper={"name", "ftype", "hash", "param", "hash", "type", "is_rule"},
                  charge={"hash", "charge", "element", "smarts"},
                  angle={"name", "ftype", "hash", "param", "hash", "type", "is_rule"},
                  dihedral={"name", "ftype", "hash", "param", "hash", "type", "is_rule"})


class OplsDB:
    def __init__(self, filename, create=False):
        self._rules_found: list[Rule] = []
        self.engine = create_engine(f"sqlite:///{filename}")
        session = sessionmaker(bind=self.engine)
        self.session = session()
        if create:
            Base.metadata.drop_all(bind=self.engine)
            Base.metadata.create_all(bind=self.engine)

    def add(self, target='atom_gmx', uniq=False, **kwargs):
        data_type = _data_types.get(target)
        if not data_type:
            raise ValueError(f"The target has to be one of {_data_types.keys()}!")
        target, _ = target.split('_')
        prop_list = _prop_list.get(target)
        if target == 'atom':
            if kwargs.get('desc'):
                _hash = sha256(kwargs['desc'].encode("utf8")).digest().hex()
                kwargs['hash'] = _hash
                kwargs['is_rule'] = True
            else:
                kwargs['desc'] = 'nil'
                kwargs['is_rule'] = False
        if target == 'bonded':
            kwargs['param'] = json.dumps(kwargs['param'])
            if kwargs.get('is_rule') is None:
                kwargs['is_rule'] = False
            if kwargs.get('hash') is None:
                kwargs['hash'] = kwargs['name']


        if not set(kwargs.keys()) == prop_list:
            raise ValueError(f"Missing {prop_list - set(kwargs.keys())} for {target}.\n"
                             f"The property list of data {target} is: \n"
                             f"{prop_list}\n"
                             f"You give {set(kwargs.keys())}")
        obj: data_type = data_type(**kwargs)
        count = 0
        if not uniq:
            search = self.session.query(data_type).filter_by(hash=obj.hash)
            logger.debug(f"Search for {obj.name} with hash {obj.hash} in {data_type} and res: {list(search)}")
            count = search.count()
        if count < 1:
            self.session.add(obj)
            logger.debug(f"Target {target} add {obj} in {data_type}")
            self.session.commit()
        else:
            res: data_type = list(search)[0]
            logger.warning(f"{obj.__repr__()} has same hash with {res.__repr__()}, "
                           f"skip adding to database {data_type}.")

    def search(self, target='atom_gmx', hash=None, name=None, smarts=None, stype=None) -> Union[None, Any]:
        data_type = _data_types.get(target)
        if stype is None:
            if not data_type:
                raise ValueError(f"The target has to be one of {_data_types.keys()}!")

            if name is not None:
                res = list(self.session.query(data_type).filter(data_type.name == name))
            elif smarts is not None:
                res = list(self.session.query(data_type).filter(data_type.smarts == smarts))
            else:
                res = list(self.session.query(data_type).filter(data_type.hash == hash))
        else:
            if not data_type:
                raise ValueError(f"The target has to be one of {_data_types.keys()}!")
            if name is not None:
                res = list(self.session.query(data_type).filter(data_type.name == name, data_type.type == stype))
            elif smarts is not None:
                res = list(self.session.query(data_type).filter(data_type.smarts == smarts, data_type.type == stype))
            else:
                res = list(self.session.query(data_type).filter(data_type.hash == hash, data_type.type == stype))

        if not res:
            return

        var_dict = vars(res[0])
        var_dict.pop('_sa_instance_state', None)
        var_dict.pop('_id', None)
        var_dict.pop('_type', None)

        if 'bond' in target or 'improper' in target:
            data_type = OplsBonded
        if 'atom' in target:
            data_type = OplsAtom
        if 'charge' in target:
            data_type = OplsAtom
            var_dict['name'] = "boss_charge"
            var_dict["bond_type"] = var_dict['element']
            var_dict['epsilon'] = 0.0
            var_dict['sigma'] = 0.0
            var_dict['ptype'] = 'A'
            var_dict['atomic_num'] = 0
            var_dict['mass'] = 0
        return data_type(**var_dict)

    def get_rules(self) -> List[Rule]:
        if self._rules_found:
            return self._rules_found
        ret = []
        atoms = list(self.session.query(OplsAtomGMX).filter_by(is_rule=True))
        for atom in atoms:
            patt = Chem.MolFromSmarts(atom.desc)
            var_dict = vars(atom)
            var_dict.pop('_sa_instance_state', None)
            oa = OplsAtom(**var_dict)
            ret.append(Rule(patt, oa, atom.desc))
        self._rules_found = ret
        return ret

    @property
    def rules(self) -> List[Rule]:
        return self.get_rules()

    def bonded_rules(self, target: str):
        res = self.session.query(OplsBondedDB).filter_by(type=target, is_rule=True)
        if not res:
            return []
        return list(res)

    def stats(self):
        n_rules = len(self.rules)
        n_atoms = self.session.query(OplsAtomDB).filter_by(is_rule=False).count()
        n_bonds = self.session.query(OplsBondedDB).filter_by(type='bond').count()
        n_angle = self.session.query(OplsBondedDB).filter_by(type='angle').count()
        n_dihed = self.session.query(OplsBondedDB).filter_by(type='dihedral').count()
        n_chars = self.session.query(OplsAtomChargeDB).count()
        return f"The database contains {n_rules} gmx atom typing rules, {n_atoms} atoms, " \
               f"{n_bonds} bonds, {n_angle} angles, {n_dihed} dihedrals, {n_chars} charges."

if __name__ == "__main__":
    import os
    this_dir, this_file = os.path.split(__file__)
    db = OplsDB(os.path.join(this_dir, 'resources', 'test.db'), create=True)
    #db.add('bonded_gmx', name='C-N-C-CA', hash='C-N-C-CA', ftype=1, param=json.dumps([1, 2, 3]), type='bond')
    # db.add('atom_gmx', name='gmx_opls_145', ptype='A',
    #      epsilon=0.33, sigma=0.33, charge=-0.115,
    #      desc="[$([c]([c])[c])]",
    #      hash=None, smarts='nil', bond_type='CA', mass=12, atomic_num=6, element='C')
    # db.add('atom_boss', name='gmx_opls_145', ptype='A',
    #       epsilon=0.33, sigma=0.33, charge=-0.115,
    #       hash='1' * 32, smarts='nil', bond_type='CA', mass=12, atomic_num=6, element='C')
    r = db.search(target='bonded_gmx', name='C-N-C-CA', stype='bond')
    rules = db.get_rules()
    print(db.stats())
    print(rules)
    print(r)
    print(OplsBondedDB.name == 'ab')
