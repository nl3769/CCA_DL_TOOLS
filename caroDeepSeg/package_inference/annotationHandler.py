"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import scipy.io
import os
import numpy                                            as np
from scipy                                              import interpolate
from package_inference.gui                              import cv2Annotation
from package_utils.get_biggest_connected_region         import get_biggest_connected_region
from numba                                              import jit

# --- window of +/- neighbours pixels where the algorithm searches the borders
NEIGHBOURS = 15

# ----------------------------------------------------------------------------------------------------------------------
def get_gt_from_txt(path):

    contour_ = []
    contour = []
    with open(path, 'r') as f:
        contour_gt = f.readlines()

    contour_.append(contour_gt[0].split(' '))
    contour_.append(contour_gt[1].split(' '))
    tmp = [val.split('.')[0] for val in contour_[0]]
    contour.append(tmp)
    contour.append(contour_[1])

    dim_val = (2, len(contour_[0])-1)
    val = np.zeros(dim_val)
    for i in range(len(contour[0])-1):
        val[0, i] = float(contour[0][i]) - 1
        val[1, i] = float(contour[1][i])

    return val

# ----------------------------------------------------------------------------------------------------------------------
def load_borders(borders_path):
    """ Load the right and left border from expert annotation instead to use GUI. """

    mat_b = scipy.io.loadmat(borders_path)
    right_b = mat_b['border_right']
    right_b = right_b[0, 0] - 1
    left_b = mat_b['border_left']
    left_b = left_b[0, 0] - 1

    # --- we increase the size of the borders if they are not big enough (128 is the width of the patch)
    if right_b-left_b<128:
        k=round((right_b-left_b)/2)+1
        right_b=right_b+k
        left_b=left_b-k

    return {"leftBorder": left_b,
            "rightBorder": right_b}

# ----------------------------------------------------------------------------------------------------------------------
def load_FW_prediction(path: str):
    """ Load the far wall prediction. """


    predN = open(path, "r")
    prediction = predN.readlines()
    predN.close()
    pred = np.zeros(len(prediction))
    for k in range(len(prediction)):
        pred[k] = prediction[k].split('\n')[0].split(' ')[-1]

    return pred

# ----------------------------------------------------------------------------------------------------------------------
class annotationClassIMC():

    """ annotationClass contains functions to:
        - update annotations
        - initialize the annotation maps
        - compute the intima-media thickness """

    def __init__(self, dimension: tuple, first_frame: np.ndarray, scale: dict, patient_name: str, CF_org: float, p=None):

        self.map_annotation = np.zeros((dimension[0] + 1, dimension[2], 2))
        self.map_annotation_org = np.zeros((dimension[0],) + (first_frame.shape[1],) + (2,))
        self.patient = patient_name
        self.overlay = p.SHIFT_X
        self.seq_dimension = dimension
        self.borders_ROI = {}
        pos, borders_seg, borders_ROI = self.get_far_wall(img=first_frame, patient_name=patient_name, CF_org=CF_org, p=p)
        self.org_grid = np.linspace(0, pos.shape[0]-1, pos.shape[0])
        self.query_grid = np.linspace(0, pos.shape[0]-1, self.seq_dimension[2])
        self.borders_org = {"leftBorder": borders_ROI[0], "rightBorder": borders_ROI[1]}
        self.initialization(localization=pos, scale=scale)
        self.borders = {"leftBorder": round(borders_seg[0] * scale['scale_x']), "rightBorder": round(borders_seg[1] * scale['scale_x'])}
        self.borders_ROI = {"leftBorder": round(borders_ROI[0] * scale['scale_x']), "rightBorder": round(borders_ROI[1] * scale['scale_x'])}

    # ------------------------------------------------------------------------------------------------------------------
    def initialization(self, localization: np.ndarray, scale: float):
        """ Initialize map_annotation with the manual delineation. """

        f = interpolate.interp1d(self.org_grid, localization)
        self.map_annotation[0, :, 0] = f(self.query_grid) * scale['scale_y']
        self.map_annotation[0, :, 1] = f(self.query_grid) * scale['scale_y']

    # ------------------------------------------------------------------------------------------------------------------
    def update_annotation(self, previous_mask: np.ndarray, frame_ID: int, offset: int):
        """ Computes the position of the LI and MA interfaces according to the predicted mask. """

        # --- the algorithm starts from the left to the right
        x_start = self.borders['leftBorder']
        x_end = self.borders['rightBorder']
        # --- dimension of the mask
        dim = previous_mask.shape
        # --- we extract the biggest connected region
        previous_mask[previous_mask > 0.5] = 1
        previous_mask[previous_mask < 1] = 0
        previous_mask = get_biggest_connected_region(previous_mask)
        # --- we count the number of white pixels to localize the seed
        white_pixels = np.array(np.where(previous_mask == 1))
        seed = (round(np.mean(white_pixels[0,])), round(np.mean(white_pixels[1,])))
        j, limit=0, dim[0]-1
        self.LI_center_to_left_propagation(j, seed, x_start, dim, previous_mask, offset, self.map_annotation[frame_ID,], NEIGHBOURS, limit)
        j, limit=0, dim[0]-1
        self.LI_center_to_right_propagation(j, seed, x_end, dim, previous_mask, offset, self.map_annotation[frame_ID,], NEIGHBOURS, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_left_propagation(j, seed, x_start, dim, previous_mask, offset, self.map_annotation[frame_ID,], NEIGHBOURS, limit)
        j, limit = 0, dim[0]-1
        self.MA_center_to_right_propagation(j, seed, x_end, dim, previous_mask, offset, self.map_annotation[frame_ID,], NEIGHBOURS, limit)

        return previous_mask

    # ------------------------------------------------------------------------------------------------------------------
    def get_far_wall(self, img: np.ndarray, patient_name: str, CF_org: float, p):

        if p.INITIALIZATION_FROM_REFERENCES:

            LI_contour_ = get_gt_from_txt(os.path.join(p.PATH_FW_REFERENCES, patient_name.split('.tiff')[0] + "-LI.txt"))
            MA_contour_ = get_gt_from_txt(os.path.join(p.PATH_FW_REFERENCES, patient_name.split('.tiff')[0] + "-MA.txt"))

            LI = np.zeros(img.shape[1])
            MA = np.zeros(img.shape[1])
            for i in range(LI_contour_.shape[1]):
                LI[int(LI_contour_[0, i])-1] = LI_contour_[1, i]
            for i in range(MA_contour_.shape[1]):
                MA[int(MA_contour_[0, i])-1] = MA_contour_[1, i]

            borders_ROI = [int(max(MA_contour_[0, 0], LI_contour_[0, 0])), int(min(MA_contour_[0, -1]-1, LI_contour_[0, -1]))]
            borders_seg = borders_ROI
            fw_approx = np.zeros(img.shape[1])

            for i in range(borders_seg[0], borders_seg[1]):
                fw_approx[i] = (MA[i] + LI[i])/2

            # --- check if roi is width enought to segment the artery, else we extend the median axis using the same value
            recquired_width = p.ROI_WIDTH
            roi_width = (borders_seg[1] - borders_seg[0]) * CF_org
            if roi_width <= recquired_width:
                min_pxl_recq = np.floor(recquired_width/CF_org)
                mean_pos = int((borders_seg[1] + borders_seg[0]))/2
                # --- clearly more than the minimum recquired, but it allows overlaping
                for i in range((borders_seg[0] - int(min_pxl_recq/2)) - 10, borders_seg[0]):
                    fw_approx[i] = fw_approx[borders_seg[0]]
                for i in range(borders_seg[1], (borders_seg[1] + int(min_pxl_recq/2)) + 10):
                    fw_approx[i] = fw_approx[borders_seg[1]-1]

                borders_seg[0] = borders_seg[0] - int(min_pxl_recq/2) - 10
                borders_seg[1] = borders_seg[1] + int(min_pxl_recq / 2) + 10

        else:

            img = np.array(img)
            I_ = img
            image = np.zeros(img.shape + (3,))

            image[:, :, 0] = I_.copy()
            image[:, :, 1] = I_.copy()
            image[:, :, 2] = I_.copy()

            coordinateStore = cv2Annotation("Far wall manual detection", image.astype(np.uint8))
            pos = coordinateStore.getpt()

            borders_seg = pos[3]
            borders_ROI = pos[2]
            fw_approx = np.zeros(I_.shape[1])
            fw_approx[borders_seg[0]:borders_seg[1]] = pos[0]

        return fw_approx, borders_seg, borders_ROI

    # ------------------------------------------------------------------------------------------------------------------
    def IMT(self):
        """ Compute the IMT. """

        xLeft = self.borders['leftBorder']
        xRight = self.borders['rightBorder']

        IMT = self.map_annotation[:, xLeft:xRight, 1] - self.map_annotation[:, xLeft:xRight, 0]

        return np.mean(IMT, axis=1), np.median(IMT, axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    def inter_contours(self, id_seq, scale):

        f_LI = interpolate.interp1d(self.query_grid, self.map_annotation[id_seq+1, :, 0])
        f_MA = interpolate.interp1d(self.query_grid, self.map_annotation[id_seq+1, :, 1])

        self.map_annotation_org[0, :, 0] = f_LI(self.org_grid) / scale['scale_y']
        self.map_annotation_org[0, :, 1] = f_MA(self.org_grid) / scale['scale_y']

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def yPosition(xLeft: int, width: int, height: int, map: np.ndarray):
        """ Compute the y position on which the current patch will be centered. """

        xRight = xLeft + width

        # --- we load the position of the LI and MA interfaces
        posLI = map[:, 0][xLeft:xRight]
        posMA = map[:, 1][xLeft:xRight]

        # --- we compute the mean value and retrieve the half height of a patch
        concatenation = np.concatenate((posMA, posLI))
        y_mean = round(np.mean(concatenation) - height / 2)
        y_max = round(np.max(concatenation) - height / 2)
        y_min = round(np.min(concatenation) - height / 2)

        # --- we check if the value is greater than zero to avoid problems
        max_height = map.shape[1] - 1
        if y_mean < 0 and y_mean > max_height or y_min < 0 or y_max < 0:
            print("Problem with yPosition !!!!!!")
            y_mean, y_min, y_max = 0, 0, 0

        return y_mean, y_min, y_max

    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def LI_center_to_right_propagation(j: int, seed: tuple, x_end: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighbours: int, limit: int):
        """ Computes the LI interface from the center to the right. """

        for i in range(seed[1] + 1, x_end):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previous_mask[j, i] == 1):
                    map_annotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    map_annotation[i, 0] = map_annotation[i - 1, 0]
                    condition = False

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighbours + 1
            limit = j + 2 * neighbours

    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def LI_center_to_left_propagation(j: int, seed: tuple, x_start: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighbours: int, limit: int):
        """ Computes the LI interface from the center to the left. """

        for i in range(seed[1], x_start - 1, -1):
            # --- if condition while a boundary is found
            condition = True
            while condition == True:

                # --- the boundary is found, while we change the column
                if (j < dim[0] and previous_mask[j, i] == 1):
                    map_annotation[i, 0] = j + offset
                    condition = False
                # --- if any boundary is found, the current boundary is equal to the previous one. Note that it is a problem if a boundary is not found at the first step.
                elif j == limit:
                    map_annotation[i, 0] = map_annotation[i + 1, 0]
                    condition = False
                    # previous_mask[j, i] = 100 # for debug

                j += 1

            # --- we initialize the new neighbours windows as well as the new limit value (+1 to compensate j+=1)
            j -= neighbours + 1
            limit = j + 2 * neighbours

    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def MA_center_to_right_propagation(j: int, seed: tuple, x_end: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighbours: int, limit: int):
        """ Computes the MA interface from the center to the right. """

        for i in range(seed[1] + 1, x_end):
            condition = True

            while condition == True:

                if (j < dim[0] and previous_mask[dim[0] - 1 - j, i] == 1):
                    map_annotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    map_annotation[i, 1] = map_annotation[i - 1, 1]
                    condition = False
                    # previous_mask[j, i] = 100

                j += 1

            j -= neighbours + 1
            limit = j + 2 * neighbours

    # ------------------------------------------------------------------------------------------------------------------
    # --- use numba to be faster
    @staticmethod
    @jit(nopython=True)
    def MA_center_to_left_propagation(j: int, seed: tuple, x_start: int, dim: tuple, previous_mask: np.ndarray, offset: int, map_annotation: np.ndarray, neighbours: int, limit: int):
        """ Computes the MA interface from the center to the left. """

        for i in range(seed[1], x_start - 1, -1):
            condition = True

            while condition == True:

                if (j < dim[0] and previous_mask[dim[0] - 1 - j, i] == 1):
                    map_annotation[i, 1] = dim[0] - 1 - j + offset
                    condition = False

                elif j == limit:
                    map_annotation[i, 1] = map_annotation[i + 1, 1]
                    condition = False

                j += 1

            j -= neighbours + 1
            limit = j + 2 * neighbours