# -*- coding: utf-8 -*-

import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)-15s] %(levelname)7s >> %(message)s (%(filename)s:%(lineno)d, %(funcName)s())',
    datefmt='%Y-%m-%d %H:%M:%S',
)
LOGGER = logging.getLogger(__name__)

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from ncc.logging import meters, metrics, progress_bar  # noqa

sys.modules['ncc.meters'] = meters
sys.modules['ncc.metrics'] = metrics
sys.modules['ncc.progress_bar'] = progress_bar

# ============== NCC Library info ============== #
# library properties
__NAME__ = 'NCC'
__FULL_NAME__ = 'Natural Code and Comment Lib.'
__DESCRIPTION__ = '''
NCC (Natural Code and Comment Lib.) is developed for Programming Language Processing and Code Analysis.
'''
__VERSION__ = '0.4.0'
