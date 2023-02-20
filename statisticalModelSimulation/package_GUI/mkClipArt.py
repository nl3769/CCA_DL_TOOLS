'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import random
import cv2
import numpy                as np
from scipy                  import ndimage
import matplotlib.pyplot    as plt
from scipy.interpolate      import PchipInterpolator, interp1d, griddata
import matplotlib.pyplot    as plt

class mkClipArt:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, window_name: str, clip_size: list, range_CF = [20, 50]):

        image = np.zeros(clip_size)
        self.img_to_fill = image[..., 0].copy().squeeze()
        self.org_img = image.copy()
        self.current_img = image.copy()
        self.window_name = window_name
        self.CF = random.uniform(range_CF[0], range_CF[1])*1e-6
        self.nb_pixel = int(np.round(0.8*1e-3/self.CF))
        self.x_new = np.linspace(0, self.org_img.shape[1] - 1, self.org_img.shape[1])
        self.adventicia_depth = 10 * 1e-3 # to compute depth inside the adventicia

        # --- display
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, self.current_img)

        # --- name of the current processed interface
        self.interfaces = {'LI_bottom': [], 'MA_bottom': [], 'LI_top': [], 'MA_top': []}
        self.pos = {'LI_bottom': [], 'MA_bottom': [], 'LI_top': [], 'MA_top': []}
        self.keys_interfaces = list(self.interfaces.keys())
        self.incr = 0

        self.color_cirle_top = (0, 0, 250)  # BGR format
        self.thickness_circle = 1
        self.radius_circle = 1
        self.stop = False
        self.top_first = True
        self.bottom_first = True
        
    # ------------------------------------------------------------------------------------------------------------------
    def select_points(self, event, x: int, y: int, flags: int, param):

        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            self.img = self.org_img.copy()
            self.pts_bottom, self.pts_top = [], []
            self.top_val, self.bottom_val = [], []
        elif event == cv2.EVENT_LBUTTONUP:
            self.pos[self.keys_interfaces[self.incr]].append([x, y])
            cv2.circle(self.current_img,
                       (x, y),
                       self.radius_circle,
                       self.color_cirle_top,
                       self.thickness_circle)
            if len(self.pos[self.keys_interfaces[self.incr]]) > 2:
                z_new = update_val(self.pos[self.keys_interfaces[self.incr]], self.x_new)
                self.interfaces[self.keys_interfaces[self.incr]] = z_new
                self.add_interfaces()
        if event == cv2.EVENT_MBUTTONUP:
            if len(self.pos[self.keys_interfaces[self.incr]]) > 0:
                self.pos[self.keys_interfaces[self.incr]].pop(-1)
            elif self.incr > 0:
                self.incr -= 1
                if len(self.pos[self.keys_interfaces[self.incr]]) > 0:
                    self.pos[self.keys_interfaces[self.incr]].pop(-1)
            else:
                raise Exception("can't go back")
            if len(self.pos[self.keys_interfaces[self.incr]]) < 2:
                self.pos[self.keys_interfaces[self.incr]] = []
                self.interfaces[self.keys_interfaces[self.incr]] = []
            else:
                z_new = update_val(self.pos[self.keys_interfaces[self.incr]], self.x_new)
                self.interfaces[self.keys_interfaces[self.incr]] = z_new
            self.add_interfaces()
        if event == cv2.EVENT_MOUSEWHEEL:
            self.incr += 1
            if self.incr == 1:
                self.interfaces[self.keys_interfaces[self.incr]] = self.interfaces[self.keys_interfaces[self.incr-1]] + self.nb_pixel
                self.pos[self.keys_interfaces[self.incr]] = [[key[0], key[1] + self.nb_pixel] for key in self.pos[self.keys_interfaces[self.incr-1]]]
            elif self.incr == 3:
                self.interfaces[self.keys_interfaces[self.incr]] = self.interfaces[self.keys_interfaces[self.incr-1]] - self.nb_pixel
                self.pos[self.keys_interfaces[self.incr]] = [[key[0], key[1] - self.nb_pixel] for key in self.pos[self.keys_interfaces[self.incr-1]]]
            self.add_interfaces()
        if self.incr == 4:
            self.stop = True

    # ------------------------------------------------------------------------------------------------------------------
    def add_interfaces(self):

        self.current_img = self.org_img.copy()
        for key in self.interfaces.keys():
            if len(self.interfaces[key]) > 2:
                for x, z in enumerate(self.interfaces[key]):
                    if 0 < int(z) < self.current_img.shape[0]:
                        self.current_img[int(z), x] = 255

    # ------------------------------------------------------------------------------------------------------------------
    def getpts(self):

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.current_img)
        cv2.setMouseCallback(self.window_name, self.select_points)
        self.stop = False
        while (not self.stop):
            cv2.imshow(self.window_name, self.current_img)
            k = cv2.waitKey(2)
        cv2.destroyWindow(self.window_name)

    # ------------------------------------------------------------------------------------------------------------------
    def set_gray_scale(self, IMC_density, adventicia_density, scale_rayleigh):

        k_factor = 3
        x_grid = np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1] * k_factor)
        z_grid = np.linspace(0, self.org_img.shape[0] * self.CF, self.org_img.shape[0] * k_factor)
        f = interp1d(np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1]), self.interfaces["MA_top"])
        self.interfaces["MA_top"] = f(x_grid)
        f = interp1d(np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1]), self.interfaces["LI_top"])
        self.interfaces["LI_top"] = f(x_grid)
        f = interp1d(np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1]), self.interfaces["MA_bottom"])
        self.interfaces["MA_bottom"] = f(x_grid)
        f = interp1d(np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1]), self.interfaces["LI_bottom"])
        self.interfaces["LI_bottom"] = f(x_grid)
        [X, Z] = np.meshgrid(x_grid, z_grid)
        class_arr = np.zeros(X.shape + (3,))
        # --- classify region in the image
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                if Z[j, i] <= (self.interfaces["MA_top"][i] * self.CF):
                    class_arr[j, i, 0] = 0
                elif (self.interfaces["MA_top"][i] * self.CF) < Z[j, i] < (self.interfaces["LI_top"][i] * self.CF):
                    class_arr[j, i, 0] = 1
                elif (self.interfaces["LI_top"][i] * self.CF) < Z[j, i] < (self.interfaces["LI_bottom"][i] * self.CF):
                    class_arr[j, i, 0] = 2
                elif (self.interfaces["LI_bottom"][i] * self.CF) < Z[j, i] < (self.interfaces["MA_bottom"][i] * self.CF):
                    class_arr[j, i, 0] = 3
                elif (Z[j, i] >= (self.interfaces["MA_bottom"][i] * self.CF)):
                    class_arr[j, i, 0] = 4
        # --- compute relative position for each pixel
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                if class_arr[j, i, 0] == 0: # --- top of the image
                    pos_rel = round((self.interfaces["LI_top"][i] * self.CF - Z[j, i]) / self.adventicia_depth * 100)
                    if pos_rel <= 100:
                        class_arr[j, i, 1] = pos_rel
                    else:
                        class_arr[j, i, 1] = 100
                elif class_arr[j, i, 0] == 1: # --- IMC top
                    pos_rel = round((self.interfaces["LI_top"][i] * self.CF - Z[j, i]) / ((self.interfaces["LI_top"][i] - self.interfaces["MA_top"][i]) * self.CF) * 100)
                    class_arr[j, i, 1] = 100 - pos_rel
                elif class_arr[j, i, 0] == 3: # --- IMC bottom
                    pos_rel = round((self.interfaces["MA_bottom"][i] * self.CF - Z[j, i]) / ((self.interfaces["MA_bottom"][i] - self.interfaces["LI_bottom"][i]) * self.CF) * 100)
                    class_arr[j, i, 1] = pos_rel
                elif class_arr[j, i, 0] == 4: # --- below IMC bottom
                    p_max = self.interfaces["MA_bottom"][i] * self.CF + self.adventicia_depth
                    pos_rel = round((1 - (p_max - Z[j, i]) / self.adventicia_depth) * 100)
                    if pos_rel >= 0:
                        class_arr[j, i, 1] = pos_rel
                    elif pos_rel > 100:
                        class_arr[j, i, 1] = 100
                    else:
                        class_arr[j, i, 1] = 100
        # --- set gray value for each pixel
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                if class_arr[j, i, 0] == 0:
                    dist = adventicia_density[str(int(class_arr[j, i, 1]))]
                    class_arr[j, i, 2] = np.random.normal(dist['mu'], dist['std'])
                elif class_arr[j, i, 0] == 1:
                    dist = IMC_density[str(int(class_arr[j, i, 1]))]
                    class_arr[j, i, 2] = np.random.normal(dist['mu'], dist['std'])
                elif class_arr[j, i, 0] == 2:
                    class_arr[j, i, 2] = np.random.rayleigh(scale_rayleigh)
                elif class_arr[j, i, 0] == 3:
                    dist = IMC_density[str(int(class_arr[j, i, 1]))]
                    class_arr[j, i, 2] = np.random.normal(dist['mu'], dist['std'])
                elif class_arr[j, i, 0] == 4:
                    dist = adventicia_density[str(int(class_arr[j, i, 1]))]
                    class_arr[j, i, 2] = np.random.normal(dist['mu'], dist['std'])

        I = ndimage.gaussian_filter(class_arr[..., 2], sigma=0.8)
        x_q = np.linspace(0, self.org_img.shape[1] * self.CF, self.org_img.shape[1])
        z_q = np.linspace(0, self.org_img.shape[0] * self.CF, self.org_img.shape[0])
        [Z_q, X_q] = np.meshgrid(x_q, z_q)
        I = griddata((X.flatten(), Z.flatten()), I.flatten(), (Z_q.flatten(), X_q.flatten()), method='linear', fill_value=0)
        I = I.reshape(Z_q.shape)

        return I

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, IMC_density, adventicia_density, scale_rayleigh):

        self.getpts()
        I = self.set_gray_scale(IMC_density, adventicia_density, scale_rayleigh)
        I = np.round(I).astype(int)

        return I, self.CF, self.interfaces

# ----------------------------------------------------------------------------------------------------------------------
def update_val(pos, x_new):

    x_pts = [key[0] for key in pos]
    z_pts = [key[1] for key in pos]

    return get_interp_values(x_pts, z_pts, x_new)

# ----------------------------------------------------------------------------------------------------------------------
def get_interp_values(x_pts, z_pts, x_new):

    idx = np.argsort(x_pts)
    x_pts_pv, z_pts_pv = x_pts.copy(), z_pts.copy()
    x_pts = [x_pts_pv[id] for id in idx]
    z_pts = [z_pts_pv[id] for id in idx]
    f = PchipInterpolator(x_pts, z_pts)

    return f(x_new)

# ----------------------------------------------------------------------------------------------------------------------
def get_val_gauss(y_q, distribution):

    y = np.linspace(0, 1, len(distribution.keys()))
    val = np.zeros(len(distribution.keys()))
    for i, key in enumerate(distribution.keys()):
        val[i] = np.random.normal(distribution[key]['mu'], distribution[key]['std'])

    f = interp1d(y, val)

    return f(y_q)