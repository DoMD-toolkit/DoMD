import os

from domd_forcefield.oplsaa.database import OplsDB
from misc.logger import logger

this_dir, this_file = os.path.split(__file__)
logger.info(f"Loading {os.path.join(this_dir, 'resources', 'opls.db')}")
opls_db = OplsDB(os.path.join(this_dir, 'resources', 'opls.db'), create=False)
