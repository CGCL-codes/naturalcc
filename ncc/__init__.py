# -*- coding: utf-8 -*-

import getpass
import logging
import os
import socket
import sys

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from ncc.utils.logging import meters, metrics, progress_bar  # noqa

sys.modules['ncc.meters'] = meters
sys.modules['ncc.metrics'] = metrics
sys.modules['ncc.progress_bar'] = progress_bar

# ============== NCC Library info ============== #
# library properties
__NAME__ = 'NaturalCC'
__FULL_NAME__ = 'Natural Code and Comment Lib.'
__DESCRIPTION__ = '''
NaturalCC: A Toolkit to Naturalize the Source Code Corpus.
'''
# __VERSION__ = ‘x,y(a,b,c)’, e.g. '1.2c'
# x for framework update or a considerable update,
# y for internal update, e.g. naturalcodev3 is our 3rd Version of coding,
__VERSION__ = '0.5b'

# directories to save/download dataset or files
_HOSTNAME = socket.gethostname()
_USERNAME = getpass.getuser()
"""
Pycharm cannot get user-defined environment variables, if you want to use user-defined variables.
PLZ, launch pycharm at {your pycharm directory}/pycharm.sh.
"""
__DEFAULT_DIR__ = os.environ.get('NCC', '~')
if str.endswith(_HOSTNAME, '.uts.edu.au'):
    __DEFAULT_DIR__ = '/data/yanghe'
elif _HOSTNAME == 'GS65':
    __DEFAULT_DIR__ = '/data'
elif _HOSTNAME == 'node14':
    __DEFAULT_DIR__ = '/mnt/wanyao'
elif _HOSTNAME == 'node13':
    __DEFAULT_DIR__ = '/mnt/wanyao'

__DEFAULT_DIR__ = os.path.expanduser(__DEFAULT_DIR__)
__CACHE_NAME__ = 'ncc_data'
__CACHE_DIR__ = os.path.join(__DEFAULT_DIR__, __CACHE_NAME__)
__LIBS_DIR__ = os.path.join(__CACHE_DIR__, 'tree_sitter_libs')
__BPE_DIR__ = os.path.join(__CACHE_DIR__, 'byte_pair_encoding')
__JAVA_HOME__ = os.path.join(os.getenv('JAVA_HOME', '/usr'), 'bin/java')
LOGGER.debug(
    f"Host Name: {_HOSTNAME}; User Name: {_USERNAME}; Data directory: {__DEFAULT_DIR__}; " \
    f"Cache directory: {__CACHE_DIR__}; TreeSitter directory: {__LIBS_DIR__}; BPE directory: {__BPE_DIR__}; " \
    f"Java_home: {__JAVA_HOME__}; "
)
