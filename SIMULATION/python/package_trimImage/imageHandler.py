'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import cv2
import numpy as np

import matplotlib.pyplot as plt
from scipy import interpolate

class getBorders:

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, windowName: str, image: np.ndarray, seg1, seg2):


        self.seg1 = seg1
        self.seg2 = seg2
        self.windowName = windowName
        self.backup = image.copy()
        self.img1 = image.copy()
        self.img = self.img1.copy()
        self.I = image[..., 0]
        self.ROI = []
        self.channel = image[:, :, 0].copy()

        self.dim = image.shape

        print(self.dim)

        self.xLeft = 0
        self.xRight = 0
        self.zTop = 0
        self.zBottom = 0

        self.xLeftSeg = 0
        self.xRighSeg = 0

        self.curr_pt = []
        self.point = []
        self.restart = False

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, image)

        self.colorCirle = (0, 0, 255)  # BGR format

        self.thicknessCircle = -1
        self.radiusCircle = 1
        self.stop = False

    # ------------------------------------------------------------------------------------------------------------------
    def select_points(self, event, x: int, y: int, flags: int, param):

        if flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_FLAG_LBUTTON:
            self.I = None
            self.seg1 = None
            self.seg2 = None
            self.stop = True

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])

            cv2.circle(self.img,
                       (x, y),
                       self.radiusCircle,
                       self.colorCirle,
                       self.thicknessCircle)

            if len(self.point) < 5:

                if len(self.point) == 1:
                    self.xLeft = self.point[0][0]
                    self.img[:, x, 0:2] = 0
                    self.img[:, x, 2] = 255
                if len(self.point) == 2:
                    self.xRight = self.point[1][0]
                    self.img[:, x, 0:2] = 0
                    self.img[:, x, 2] = 255
                if len(self.point) == 3:
                    self.zTop = self.point[2][1]
                    self.img[y, :, 0:2] = 0
                    self.img[y, :, 2] = 255
                if len(self.point) == 4:
                    self.zBottom = self.point[3][1]
                    self.img[y, :, 0:2] = 0
                    self.img[y, :, 2] = 255

            elif len(self.point) == 5:

                # check values
                if self.xRight < self.xLeft:
                    bckup = self.xLeft
                    self.xLeft = self.xRight
                    self.xRight = bckup

                if self.zTop < self.zBottom:
                    bckup = self.zTop
                    self.zTop = self.zBottom
                    self.zBottom = bckup

                # --- crop image
                self.I = self.I[self.zBottom:self.zTop, self.xLeft:self.xRight]
                self.img = np.repeat(self.I[:, :, np.newaxis], 3, axis=2)
                self.seg1 = self.seg1[self.xLeft:self.xRight] - self.zBottom
                self.seg2 = self.seg2[self.xLeft:self.xRight] - self.zBottom
                self.add_annotation()

            else:
                self.stop = True

        elif event == cv2.EVENT_MBUTTONUP or self.restart == False:

            self.restart = True
            self.img = self.backup.copy()
            self.img1 = self.backup.copy()
            self.channel = self.backup[:, :, 0].copy()
            self.point = []
            self.curr_pt = []

    # ------------------------------------------------------------------------------------------------------------------
    def add_annotation(self):

        dim = self.seg1.shape[0]

        for id in range(dim):

            z1 = int(round(self.seg1[id]))
            z2 = int(round(self.seg2[id]))

            if z1>0 and z2 > 0:
                self.img[z1, id, 0] = 0
                self.img[z1, id, 1] = 255
                self.img[z1, id, 2] = 0

                self.img[z2, id, 0] = 0
                self.img[z2, id, 1] = 255
                self.img[z2, id, 2] = 0

        self.img1 = self.img.copy()

    # ------------------------------------------------------------------------------------------------------------------
    def getpt(self, img=None):

        '''
        :return self.wallPosition: the position of the far in an np.array with dimension (1, self.xRightSeg-self.xLeftSeg)
        :return self.img: the image one which we estimate the far wall
        :return [self.xLeft, self.xRight]: the region of interest
        :return [self.xLeftSeg, self.xRightSeg]: are of 128 at least to launch the segmentation
        '''

        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()


        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowName, self.img)
        cv2.setMouseCallback(self.windowName, self.select_points)
        self.point = []

        while (1):
            self.add_annotation()
            cv2.imshow(self.windowName, self.img)
            k = cv2.waitKey(2)

            if self.stop == True:
                break

        cv2.destroyWindow(self.windowName)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        self.getpt()

        return self.I, self.seg1, self.seg2
    # ------------------------------------------------------------------------------------------------------------------