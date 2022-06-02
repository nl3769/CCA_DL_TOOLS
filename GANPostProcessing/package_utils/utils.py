import os
from pathlib import Path

def check_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)