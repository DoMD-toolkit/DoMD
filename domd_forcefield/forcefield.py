from abc import ABCMeta, abstractmethod
from typing import Any


class ForceField(metaclass=ABCMeta):

    def __init__(self, name: str):
        self.all_params = {}
        self.name = name

    @abstractmethod
    def parameterize(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def stats(self):
        pass


class CustomRule(metaclass=ABCMeta):
    __doc__ = """Use Custom Rule to define a per-atom, per-bond, per-angle, per-torsion and per-improper
    method.
    """

    def __init__(self, name: str):
        self.name = name
        return

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.process(*args, **kwargs)

    def __repr__(self):
        return self.__class__.__name__
