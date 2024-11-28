import sys
from pathlib import Path
sys.path.append(str(Path("").absolute().parent))

from ncc3.models import model_zoo
print(model_zoo)