import os
import shutil

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pres = "/home/laine/PROJECTS_IO/SIMULATION/MEIBURGER_1_FRAME"

    files = os.listdir(pres)
    for file in files:
        files_ = os.listdir(os.path.join(pres, file))
        rm = None
        for key in files_:
            if key.find('id') == -1:
                rm = key
        if rm != None:
            shutil.rmtree(os.path.join(pres, file, rm))

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
