'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import cv2
import numpy                as np
import matplotlib.pyplot    as plt
from scipy.interpolate      import PchipInterpolator, interp1d

class getLumen:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, windowName: str, image: np.ndarray, id_LI, pos_LI, id_MA, pos_MA):

        self.org_img = image.copy()
        self.id_LI, self.pos_LI = id_LI, pos_LI
        self.id_MA, self.pos_MA = id_MA, pos_MA
        self.windowName = windowName
        self.img = image.copy()
        self.imgbak = image.copy()
        self.id_top, self.pos_top = [], []
        self.id_bottom, self.pos_bottom = [], []
        self.pts_bottom, self.pts_top = [], []
        self.top_val, self.bottom_val = [], []
        # --- display
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, image)
        self.colorCirleTop = (0, 0, 250)  # BGR format
        self.colorCirleBottom = (100, 100, 0)  # BGR format
        self.thicknessCircle = 1
        self.radiusCircle = 1
        self.stop = False
        self.top_first = True
        self.bottom_first = True
        
    # ------------------------------------------------------------------------------------------------------------------
    def select_points(self, event, x: int, y: int, flags: int, param):

        update = False

        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            self.img = self.org_img.copy()
            self.add_annotation()
            self.pts_bottom, self.pts_top = [], []
            self.top_val, self.bottom_val = [], []
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.pts_bottom.append([x, y])
            cv2.circle(self.img,
                       (x, y),
                       self.radiusCircle,
                       self.colorCirleBottom,
                       self.thicknessCircle)
            # --- get new value
            x_pts_bottom = [key[0] for key in self.pts_bottom]
            z_pts = [key[1] for key in self.pts_bottom]
            if len(x_pts_bottom) > 3:
                idx = np.argsort(x_pts_bottom)
                x_pts_pv, z_pts_pv = x_pts_bottom.copy(), z_pts.copy()
                x_pts_bottom = [x_pts_pv[id] for id in idx]
                z_pts_bottom = [z_pts_pv[id] for id in idx]
                f = PchipInterpolator(x_pts_bottom, z_pts_bottom)
                x_new = np.linspace(min(x_pts_bottom), max(x_pts_bottom), max(x_pts_bottom)-min(x_pts_bottom)+1)
                z_new = f(x_new)
                self.bottom_val = []
                self.bottom_val.append(x_new)
                self.bottom_val.append(z_new)
                update = True
        elif event == cv2.EVENT_MBUTTONUP:
            self.pts_top.append([x, y])
            cv2.circle(self.img,
                       (x, y),
                       self.radiusCircle,
                       self.colorCirleTop,
                       self.thicknessCircle)
            # --- get new value
            x_pts_top = [key[0] for key in self.pts_top]
            z_pts_top = [key[1] for key in self.pts_top]
            if len(x_pts_top) > 3:
                idx = np.argsort(x_pts_top)
                x_pts_pv, z_pts_pv = x_pts_top.copy(), z_pts_top.copy()
                x_pts_top = [x_pts_pv[id] for id in idx]
                z_pts_top = [z_pts_pv[id] for id in idx]
                f = PchipInterpolator(x_pts_top, z_pts_top)
                x_new = np.linspace(min(x_pts_top), max(x_pts_top), max(x_pts_top)-min(x_pts_top)+1)
                z_new = f(x_new)
                self.top_val = []
                self.top_val.append(x_new)
                self.top_val.append(z_new)
                update = True
        if not self.top_val and not self.bottom_val:
            update = False
        if update:
            if len(list(self.bottom_val[0])) > 3 or len(list(self.top_val[0])) > 3:
                self.update_full(self.top_val, self.bottom_val)
        if flags == cv2.EVENT_FLAG_SHIFTKEY + cv2.EVENT_FLAG_LBUTTON:
            self.stop = True

    # ------------------------------------------------------------------------------------------------------------------
    def update_full(self, val_top, val_bottom):

        self.img = self.org_img.copy()
        self.add_annotation()
        # --- top val in blue
        if val_top:
            idx = list(val_top[0])
            poss = list(val_top[1])
            if len(idx) > 3:
                for id, pos in zip(idx, poss):
                    self.img[int(pos), int(id), 0] = 250
                    self.img[int(pos), int(id), 1] = 0
                    self.img[int(pos), int(id), 2] = 0
        # --- bottom val in red
        if val_bottom:
            idx = list(val_bottom[0])
            poss = list(val_bottom[1])
            if len(idx) > 3:
                for id, pos in zip(idx, poss):
                    self.img[int(pos), int(id), 0] = 0
                    self.img[int(pos), int(id), 1] = 0
                    self.img[int(pos), int(id), 2] = 250

    # ------------------------------------------------------------------------------------------------------------------
    def update_disp(self, val):

        self.img = self.org_img.copy()
        self.add_annotation()
        idx = list(val[0])
        poss = list(val[1])
        for id, pos in zip(idx, poss):
            self.img[int(pos), int(id), 0] = 0
            self.img[int(pos), int(id), 1] = 0
            self.img[int(pos), int(id), 2] = 250

    # ------------------------------------------------------------------------------------------------------------------
    def add_annotation(self):

        for id, pos in zip(self.id_LI, self.pos_LI):
            if pos>0:
                self.img[int(pos), id, 0] = 0
                self.img[int(pos), id, 1] = 180
                self.img[int(pos), id, 2] = 180
        for id, pos in zip(self.id_MA, self.pos_MA):
            if pos > 0:
                self.img[int(pos), id, 0] = 0
                self.img[int(pos), id, 1] = 180
                self.img[int(pos), id, 2] = 180
        self.img1 = self.img.copy()

    # ------------------------------------------------------------------------------------------------------------------
    def getpt(self):

        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowName, self.img)
        cv2.setMouseCallback(self.windowName, self.select_points)
        self.point = []
        self.stop = False
        while (not self.stop):
            self.add_annotation()
            cv2.imshow(self.windowName, self.img)
            k = cv2.waitKey(2)
        cv2.destroyWindow(self.windowName)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.getpt()

        return self.top_val, self.bottom_val
    # ------------------------------------------------------------------------------------------------------------------