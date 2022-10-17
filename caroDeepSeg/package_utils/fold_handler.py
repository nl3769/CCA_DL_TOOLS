import os

# ----------------------------------------------------------------------------------------------------------------------
def create_dir(path):
    """ Create directory. """
    isExist = os.path.exists(path)

    if not isExist:
        # ---  create a new directory because it does not exist
        os.makedirs(path)
