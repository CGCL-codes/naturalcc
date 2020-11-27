# -*- coding: utf-8 -*-

import os
import socket
import getpass
from ncc import LOGGER

HOSTNAME = socket.gethostname()
USERNAME = getpass.getuser()
# register your hostname or username
DEFAULT_DIR = '~/.ncc'
DEFAULT_DIR = os.path.expanduser(DEFAULT_DIR)
LIBS_DIR = os.path.join(os.path.dirname(__file__), 'tree-sitter-libs')
LOGGER.debug('Host Name: {}; User Name: {}; Default data directory: {}'.format(HOSTNAME, USERNAME, DEFAULT_DIR))

__all__ = (
    HOSTNAME, USERNAME,
    DEFAULT_DIR, LIBS_DIR,
    LOGGER,
)
