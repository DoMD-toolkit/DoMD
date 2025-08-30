from .forcefield import CustomRule

__doc__ = """
Since ML was applied for atom typing, there is no need for a database-searching charge method.
"""
class DBChargeModel(CustomRule):
    def __init__(self, name: str = 'DBCharge'):
        super().__init__(name)
        return
