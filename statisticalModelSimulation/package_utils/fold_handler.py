import os

# ----------------------------------------------------------------------------------------------------------------------
def create_dir(path):
    ''' Takes string and create directory. '''

    is_exist = os.path.exists(path)

    if not is_exist:
        # ---  create a new directory because it does not exist
        os.makedirs(path)

# ----------------------------------------------------------------------------------------------------------------------