import os
import wget
from tree_sitter import Language
from ncc import (
    __LIBS_DIR__,
    LOGGER,
)
import shutil
import zipfile

# define your config
YOUR_LANGUAGE = 'csharp'
TREE_SITTER_LIB_URL = 'https://github.com/tree-sitter/tree-sitter-c-sharp/archive/master.zip'
os.makedirs(__LIBS_DIR__, exist_ok=True)
so_file = os.path.join(__LIBS_DIR__, f'{YOUR_LANGUAGE}.so')

# download
lib_filename = os.path.join(__LIBS_DIR__, f'{YOUR_LANGUAGE}.zip')
if os.path.exists(lib_filename):
    shutil.rmtree(lib_filename)
LOGGER.info(f"Download TreeSitter-{YOUR_LANGUAGE}-Parser from {TREE_SITTER_LIB_URL}")
wget.download(TREE_SITTER_LIB_URL, lib_filename)

# decompress
decompress_dir = os.path.join(__LIBS_DIR__, 'tmp')
with zipfile.ZipFile(lib_filename, 'r') as zip_file:
    zip_file.extractall(path=decompress_dir)
lib_dir = os.path.join(decompress_dir, os.listdir(decompress_dir)[0])

# build
LOGGER.info(f"Build {YOUR_LANGUAGE}.so, and save it at {__LIBS_DIR__}")
Language.build_library(
    # your language parser file, we recommend buidl *.so file for each language
    so_file,
    # Include one or more languages
    [lib_dir],
)

# delete lib zip
shutil.rmtree(decompress_dir)  # remove tmp directory
os.remove(lib_filename)  # remove zip file
