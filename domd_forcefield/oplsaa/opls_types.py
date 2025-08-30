from typing import Union, Dict, Any

from domd_forcefield.oplsaa.database import OplsAtom
from misc.logger import logger


class OplsAtomTypes(object):
    def __init__(self):
        self.atoms: dict[int, Union[OplsAtom, None]] = {}
        self.missing_atoms: dict[int, OplsAtom] = {}
        self.missing_atom_types: dict[str, set] = {}
        self.charge: Dict[int, float] = {}

    def __getitem__(self, item: int) -> Union[OplsAtom, None]:
        return self.atoms[item]

    def __setitem__(self, key: int, value: Union[OplsAtom, None]):
        self.atoms[key] = value

    def keys(self):
        return self.atoms.keys()

    def __iter__(self):
        return (_ for _ in self.atoms)

    def __len__(self):
        return len(self.atoms)

    def stats(self):
        ms = ""
        for m in self.missing_atom_types:
            lst = list(self.missing_atom_types[m])
            ms += f"{m:5}: {lst[0]}\n"
            for item in lst[1:]:
                ms += f"       {item}\n"
        logger.info(f"The sum of charges is {sum(self.charge.values())}\n" \
                    f"{len(self.missing_atoms)} atoms are un-typed.\n" \
                    f"Missing Atoms: \n{self.missing_atoms.keys()}\n" \
                    f"Missing Types:\n{ms}")
        return len(self.missing_atoms)


class OplsBondedTypes(object):
    def __init__(self, name: str):
        self.name = name
        self.idx: dict[tuple, tuple] = {}
        self.res: dict[tuple, Any] = {}
        self.missing: dict[str, list] = {}

    def keys(self):
        return self.res.keys()

    def __setitem__(self, key: tuple, value: Any):
        skey = tuple(sorted(key))
        self.idx[skey] = key
        self.res[key] = value

    def __getitem__(self, key: tuple) -> Any:
        skey = tuple(sorted(key))
        key = self.idx.get(skey)
        return self.res.get(key)

    def __len__(self):
        return len(self.res)

    def stats(self):
        sb = sa = sd = ''
        for m in self.missing:
            if len(m.split('-')) == 2:
                sb += f"{m:15}: {self.missing[m][0]}\n"
                for j in self.missing[m][1:]:
                    sb += " " * 17 + f"{j}\n"
            if len(m.split('-')) == 3:
                sa += f"{m:15}: {self.missing[m][0]}\n"
                for j in self.missing[m][1:]:
                    sa += " " * 17 + f"{j}\n"
            if len(m.split('-')) == 4:
                sd += f"{m:15}: {self.missing[m][0]}\n"
                for j in self.missing[m][1:]:
                    sd += " " * 17 + f"{j}\n"

        logger.info(f"\nFound {len(self.res)} {self.name}s\n"
                    f"Missing types:\n{sb}{sa}{sd}")
        return len(self.missing)
