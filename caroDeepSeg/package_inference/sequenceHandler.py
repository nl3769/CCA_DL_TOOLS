"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import time
import numpy                                            as np
import matplotlib.pyplot                                as plt
from package_inference.predictionHandler                import predictionClassIMC
from package_inference.annotationHandler                import annotationClassIMC
from package_utils.load_data                            import load_data

# ----------------------------------------------------------------------------------------------------------------------
class sequenceClassIMC():
    """ sequenceClass calls all the other classes to perform the calculations. This class contains all the results and runs the sliding window (sliding_window_vertical_scan). """
    def __init__(self, path_seq: str, p):

        patient_name = path_seq.split('/')[-1]

        self.PATCH_WIDTH = p.PIXEL_WIDTH
        self.PATCH_HEIGHT = p.PIXEL_HEIGHT
        self.SHIFT_X = p.SHIFT_X
        self.SHIFT_Z = p.SHIFT_Z
        self.sequence, self.firstFrame, self.scale, self.CF, self.CF_org = load_data(path=path_seq, param=p)    # load data
        self.annotationClass = annotationClassIMC(
            dimension=self.sequence.shape,
            first_frame=self.firstFrame,
            scale=self.scale,
            patient_name=patient_name,
            CF_org=self.CF_org,
            p=p)
        self.predictionClass = predictionClassIMC(
            dimensions=self.sequence.shape,
            borders=self.annotationClass.borders_ROI,
            p=p,
            img=self.sequence[0, ])
        DEBUG = False
        if DEBUG:
            for id in range(self.annotationClass.borders['leftBorder'], self.annotationClass.borders['rightBorder']):
                self.sequence[0, round(self.annotationClass.map_annotation[0, id, 0]), id] = 255
            plt.imshow(self.sequence[0, ])
            plt.show()
        self.patch = np.empty((self.PATCH_WIDTH, self.PATCH_HEIGHT), dtype=np.float32)
        self.step = 0
        self.current_frame = 0
        self.final_mask_after_post_processing = np.zeros(self.sequence.shape[1:])

    # ------------------------------------------------------------------------------------------------------------------
    def sliding_window_vertical_scan(self):
        """ At each position, three patches are extracted, and if the difference between the min and the max then the vertical scanning is automatically adjusted. """

        condition = True
        for frame_ID in range(self.sequence.shape[0]):
            self.current_frame = frame_ID
            self.predictionClass.patches = []
            self.step = 0
            y_pos_list = []
            median = (self.annotationClass.map_annotation[frame_ID, :, 0] + self.annotationClass.map_annotation[frame_ID, :, 1]) / 2
            vertical_scanning = True
            # --- condition give the information if the frame is segmented
            while condition:

                if self.step == 0:  # initialization step
                    x = self.initialization_step()
                    overlay_ = self.SHIFT_X
                median_min = np.min(median[x:x+self.PATCH_WIDTH])
                median_max = np.max(median[x:x+self.PATCH_WIDTH])
                y_mean, _, _ = self.annotationClass.yPosition(
                    xLeft=x,
                    width=self.PATCH_WIDTH,
                    height=self.PATCH_HEIGHT,
                    map=self.annotationClass.map_annotation[frame_ID, ])
                y_pos = y_mean
                # --- by default, we take three patches for a given position x. If this is not enough, the number of patches is dynamically adjusted.
                if self.PATCH_HEIGHT * 2 > (median_max - median_min):
                    self.predictionClass.patches.append({
                        "patch": self.extract_patch(x, y_pos, image = self.sequence[frame_ID, ]),
                        "frameID": frame_ID,
                        "Step": self.step,
                        "Overlay": overlay_,
                        "(x, y)": (x, y_pos)})
                    y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean - self.SHIFT_Z > int(self.PATCH_HEIGHT/2):
                        y_pos = y_mean + self.SHIFT_Z
                        patch_ = self.extract_patch(x, y_pos, image=self.sequence[frame_ID,])
                        if patch_.shape == (self.PATCH_WIDTH, self.PATCH_HEIGHT):
                            self.predictionClass.patches.append({
                                "patch": patch_,
                                "frameID": frame_ID,
                                "Step": self.step,
                                "Overlay": overlay_,
                                "(x, y)": (x, y_pos)})
                            y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                    if y_mean + int(self.PATCH_HEIGHT/2) < self.sequence.shape[1] - 1:
                        y_pos = y_mean - self.SHIFT_Z
                        patch_ = self.extract_patch(x, y_pos, image=self.sequence[frame_ID, ])
                        if patch_.shape == (self.PATCH_WIDTH, self.PATCH_HEIGHT):
                            self.predictionClass.patches.append({
                                 "patch": patch_,
                                 "frameID": frame_ID,
                                 "Step": self.step,
                                 "Overlay": overlay_,
                                 "(x, y)": (x, y_pos)})
                            y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                # --- if the condition is not verified, the artery wall is not fully considered and a vertical scan is applied
                else:
                    y_inc = median_min - 128
                    while(vertical_scanning):
                        patch_ = self.extract_patch(x, round(y_inc), image=self.sequence[frame_ID, ])
                        if patch_.shape == (self.PATCH_WIDTH, self.PATCH_HEIGHT):
                            self.predictionClass.patches.append({
                                "patch": patch_,
                                 "frameID": frame_ID,
                                 "Step": self.step,
                                 "Overlay": overlay_,
                                 "(x, y)": (x, round(y_inc))})
                            y_inc += 32
                            y_pos_list.append(self.predictionClass.patches[-1]["(x, y)"][-1])
                        else:
                            vertical_scanning = False
                self.step += 1
                vertical_scanning = True
                if ((x + self.PATCH_WIDTH) == self.annotationClass.borders_ROI['rightBorder']):  # if we reach the last position (on the right)
                    condition = False
                elif (x + self.SHIFT_X + self.PATCH_WIDTH) < (self.annotationClass.borders_ROI['rightBorder']):  # we move the patch from the left to the right with an overlay
                    x += self.SHIFT_X
                    overlay_ = self.SHIFT_X
                else:
                    tmp = x + self.PATCH_WIDTH - self.annotationClass.borders_ROI['rightBorder']  # we adjust to reach the right border
                    x -= tmp
                    overlay_ = tmp
            condition = True
            min_y = min(y_pos_list)
            max_y = max(y_pos_list)
            self.predictionClass.prediction_masks(
                id=frame_ID,
                pos={"min": min_y, "max": max_y+self.PATCH_HEIGHT})
            mask_ = self.predictionClass.map_prediction[str(frame_ID)]["prediction"]
            t = time.time()
            mask_tmp = self.annotationClass.update_annotation(
                previous_mask=mask_,
                frame_ID=frame_ID + 1,
                offset=self.predictionClass.map_prediction[str(frame_ID)]["offset"]).copy()
            mask_tmp_height = mask_tmp.shape[0]
            # --- for display only
            self.final_mask_after_post_processing[self.predictionClass.map_prediction[str(frame_ID)]["offset"]:self.predictionClass.map_prediction[str(frame_ID)]["offset"]+mask_tmp_height,:] = mask_tmp
            self.annotationClass.inter_contours(frame_ID, self.scale)
            postprocess_time = time.time() - t

        return postprocess_time

    # ------------------------------------------------------------------------------------------------------------------
    def initialization_step(self):
        """ Returns the left border. """

        return self.annotationClass.borders_ROI['leftBorder']

    # ------------------------------------------------------------------------------------------------------------------
    def extract_patch(self, x: int, y: int, image: np.ndarray):
        """ Extracts a patch at a given (x, y) coordinate. """

        img = image[y:(y + self.PATCH_HEIGHT), x:(x + self.PATCH_WIDTH)]

        return img

    # ------------------------------------------------------------------------------------------------------------------
    def compute_IMT(self, p, patient: str):
        """ Saves IMT value using annotationClass. """

        self.annotationClass.mapAnnotation = self.annotationClass.mapAnnotation[1:, ]
        IMTMeanValue, IMTMedianValue = self.annotationClass.IMT()
        spatial_res = self.CF /self.scale * 10000 # in micro meter

        plt.rcParams.update({'font.size': 16})
        plt.subplot(211)
        plt.plot(IMTMeanValue*spatial_res, "b")
        plt.ylabel('Thickness in $\mu$m')
        plt.legend(['Mean'], loc='lower right')
        plt.subplot(212)
        plt.plot(IMTMedianValue*spatial_res, "r")
        plt.ylabel('Thickness in $\mu$m')
        plt.xlabel('Frame ID')
        plt.legend(['Median'], loc='lower right')
        plt.savefig(os.path.join(p.PATH_TO_SAVE_RESULTS_COMPRESSION, patient + '_IMT_compression' + '.png'))
        plt.close()

        return IMTMeanValue, IMTMedianValue

# ----------------------------------------------------------------------------------------------------------------------