import os

# ----------------------------------------------------------------------------------------------------------------------
def create_dir(path):

    try:
        os.makedirs(path)
    except OSError as error:
        print(error)