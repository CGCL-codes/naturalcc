import os.path
import subprocess
import pathlib
def download():
    current_path=pathlib.Path(__file__).parent.resolve()
    subprocess.call(['bash',os.path.join(current_path, 'download.sh')])