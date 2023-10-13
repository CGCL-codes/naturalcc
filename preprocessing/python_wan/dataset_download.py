import os
import shutil
from preprocessing.utils import zenodo_get


from ncc import __NCC_DIR__
PYTHON_WAN_DIR = os.path.join(__NCC_DIR__, 'python_wan')
def download():
    filename = 'python_wan.tar.gz'
    record = '7202649'
    print("Fetching the python-wan dataset...")
    if(not os.path.exists(PYTHON_WAN_DIR)):
        os.mkdir(PYTHON_WAN_DIR)
    if(not os.path.exists(os.path.join(PYTHON_WAN_DIR,filename))):
        result_file = zenodo_get(record, filename)
        shutil.move(str(result_file), PYTHON_WAN_DIR)
    print("Extracting the python-wan dataset...")
    os.system(f"cd '{PYTHON_WAN_DIR}' && tar xzvf '{filename}' -C '{PYTHON_WAN_DIR}'")
    print("The python-wan dataset is prepared.")




