import os
import glob

from preprocess import LIBS_DIR

SO_FILES = [so_file for so_file in glob.glob(f'{LIBS_DIR}/*') if so_file.endswith('.so')]
LANGUAGES = [os.path.basename(so_file[:so_file.find('.so')]) for so_file in SO_FILES]
SO_FILES_MAP = {
    lang: file
    for lang, file in zip(LANGUAGES, SO_FILES)
}
