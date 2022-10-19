import package_utils.signal_processing      as pusp
import package_utils.loader                 as pul

import numpy                                as np
import matplotlib.pyplot                    as plt

from tqdm                                   import tqdm

def plot_pts(pts):
    nb_pts = len(pts.keys())

    for id_pts, key in enumerate(pts.keys()):
        x = [val[0] for val in pts[key]]
        z = [val[1] for val in pts[key]]

        plt.figure()
        plt.scatter(x, z)

    a=1

# ----------------------------------------------------------------------------------------------------------------------
def get_new_pos(motion, pos):

    height, width = motion.shape
    X = np.linspace(0, width-1, width)
    Z = np.linspace(0, height-1, height)

    x, z = np.meshgrid(X, Z)
    x_q = np.ones(1) * pos[0]
    z_q = np.ones(1) * pos[1]

    val = pusp.grid_interpolation_2D(x_q, z_q, motion, x, z)

    return val

# ----------------------------------------------------------------------------------------------------------------------
def track_pts(pts, motion):

    nb_frames = motion.shape[-1]
    nb_pts = len(pts)

    pts_tracking = {}

    for id_pts in range(nb_pts):
        pt = pts[id_pts]
        key_name = "pt_" + str(id_pts)
        pts_tracking[key_name] = []
        pts_tracking["pt_" + str(id_pts)].append(pt)
        for id_motion in tqdm(range(nb_frames)):
        # for id_motion in tqdm(range(5)):

            xmotion = motion[:, :, 0, id_motion]
            zmotion = motion[:, :, 2, id_motion]
            pos = pts_tracking["pt_" + str(id_pts)][-1]
            xmotion_ = get_new_pos(xmotion, pos)
            zmotion_ = get_new_pos(zmotion, pos)

            posx = pos[0] + xmotion_
            posz = pos[1] + zmotion_

            pts_tracking[key_name].append((posx, posz))


    return pts_tracking

# ----------------------------------------------------------------------------------------------------------------------
def main():

    pmotion = "/home/laine/Desktop/motion.pkl"
    motion = pul.load_pickle(pmotion)
    pts = [(100, 100)]
    traj = track_pts(pts, motion)
    plot_pts(traj)
    plt.show()
    a = 1
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()