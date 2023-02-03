import matplotlib.pyplot    as plt
import pickle               as pkl

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    fname = "/run/media/laine/HDD/PROJECTS_IO/SIMULATION/DATABASE/tech_022/id_008/M2/pos_0_2.pkl"

    with open(fname, 'rb') as f:
        x = pkl.load(f)

    plt.imshow(x, cmap='gray')
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------