# -*- coding: utf-8 -*-

import getpass
import os
import socket
import sys

from ncc.utils.logging import LOGGER
from ncc.utils.logging import meters, metrics, progress_bar  # noqa

sys.modules['ncc.meters'] = meters
sys.modules['ncc.metrics'] = metrics
sys.modules['ncc.progress_bar'] = progress_bar

# ============== NCC Library info ============== #
# library properties
__NAME__ = 'NaturalCC'
__FULL_NAME__ = 'Natural Code Comprehension Lib.'
__DESCRIPTION__ = '''
NaturalCC: A Toolkit to Naturalize the Source Code Corpus.
'''
# __VERSION__ = ‘x,y(a,b,c)’, e.g. '1.2c'
# x for framework update or a considerable update,
# y for internal update, e.g. naturalcodev3 is our 3rd Version of coding,
__VERSION__ = '0.6'

# directories to save/download dataset or files
__HOSTNAME__ = socket.gethostname()
__USERNAME__ = getpass.getuser()
__NCC_DIR__ = os.environ.get('NCC', '~')
assert __NCC_DIR__, FileNotFoundError("No such directory")
"""
Pycharm cannot get user-defined environment variables, if you want to use user-defined variables.
PLZ, launch pycharm at {your pycharm directory}/pycharm.sh.
"""

# __NCC_DIR__ = '/data/ncc_data'
# wtf?
__TREE_SITTER_LIBS_DIR__ = os.path.join(__NCC_DIR__, 'tree_sitter_libs')
__BPE_DIR__ = os.path.join(__NCC_DIR__, 'byte_pair_encoding')
__JAVA_HOME__ = os.path.join(os.getenv('JAVA_HOME', '/usr'), 'bin/java')

LOGGER.debug(f"Host name: {__HOSTNAME__}; user name: {__USERNAME__}")
LOGGER.debug(f"{__NAME__} version: {__VERSION__}, Data directory: {__NCC_DIR__};")
LOGGER.debug(f"TreeSitter so directory: {__TREE_SITTER_LIBS_DIR__};")
LOGGER.debug(f"BytePairEncoding dictionaries directory: {__BPE_DIR__}; ")
LOGGER.debug(f"JAVA_HOME(for Meteor): {__JAVA_HOME__};")

__all__ = [
    "LOGGER",
    "__NAME__", "__VERSION__",
    "__HOSTNAME__", "__USERNAME__", "__NCC_DIR__",
    "__TREE_SITTER_LIBS_DIR__", "__BPE_DIR__", "__JAVA_HOME__",
]
