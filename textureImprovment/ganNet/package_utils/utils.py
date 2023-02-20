'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
def check_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)