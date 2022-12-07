"""
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
"""

import os
import torch
import numpy                                        as np

from package_network.network_dilatedUnet            import dilatedUnet

# ----------------------------------------------------------------------------------------------------------------------
class predictionClassIMC():
    """ The prediction class contains the trained architecture and performs the following calculations:
    - prediction of masks
    - compute overlay and prediction maps """

    def __init__(self, dimensions: tuple, borders: dict, p, img=None):

        self.patch_height = p.PIXEL_HEIGHT              # height of the patch
        self.dim = dimensions                           # dimension of the interpolated image
        self.patch_width = p.PIXEL_WIDTH                # width of the patch
        self.patches = []                               # list in which extracted patches are stored
        self.predicted_masks = []                       # array in which we store the prediction
        self.final_mask_org = []                        # array in which we store the final combinaison of all prediction
        self.borders = borders                          # left and right borders
        self.map_overlay, self.map_prediction = {}, {}  # dictionaries evolve during the inference phase
        self.img = img
        self.param = p
        self.device = p.DEVICE
        self.model = self.load_model(p.PMODELIMC, p)

    # ------------------------------------------------------------------------------------------------------------------
    def prediction_masks(self, id: int, pos: dict):
        """ Retrieves patches, then preprocessing is applied and the self.build_maps method reassembles them. """

        # --- Adapt patches to fed network
        patchImg = []
        for i in range(len(self.patches)):
            patchImg.append(self.patches[i]["patch"])
        patches = np.asarray(patchImg)
        patches = self.patch_preprocessing(patches=patches)
        patches = np.expand_dims(patches, axis=1)
        patches = torch.from_numpy(patches).float()

        # --- Prediction
        inference = torch.zeros((1,) + (patches.shape[1],) + (patches.shape[2],) + (patches.shape[3],)).to(self.device)
        masks = torch.zeros(patches.shape)
        for id_batch in range(patches.shape[0]):
            inference[0, ] = patches[id_batch, ]
            inference = inference.to(self.device)
            masks[id_batch, ] = self.model(inference).detach().to('cpu')

        self.predicted_masks = masks.detach().numpy().copy()

        # --- reassemble patches
        self.build_maps(prediction=self.predicted_masks, id=id, pos=pos)

    # ------------------------------------------------------------------------------------------------------------------
    def build_maps(self, prediction: np.ndarray, id: int, pos: dict):
        """ Assembles the patches and predictions to create the overlay map and the prediction map. """

        pred_, overlay_ = np.zeros((pos['max'] - pos['min'], self.dim[2])), np.zeros((pos['max'] - pos['min'], self.dim[2]))
        for i in range(len(self.patches)):
            patch_ = self.patches[i]
            pos_ = patch_["(x, y)"]
            pred_[pos_[1] - pos['min']:(pos_[1] - pos['min'] + self.patch_height), pos_[0]:pos_[0] + self.patch_width] = pred_[pos_[1] - pos['min']:pos_[1] - pos['min'] + self.patch_height, pos_[0]:pos_[0] + self.patch_width] + prediction[i, 0, :, :]
            overlay_[pos_[1] - pos['min']:pos_[1] - pos['min'] + self.patch_height, pos_[0]:pos_[0] + self.patch_width] = overlay_[pos_[1] - pos['min']:pos_[1] - pos['min'] + self.patch_height, pos_[0]: pos_[0] + self.patch_width] + np.ones((self.patch_height, self.patch_width))

        overlay_[overlay_ == 0] = 1
        pred_ = pred_ / overlay_

        self.map_overlay[str(id)] = {"prediction": overlay_.copy(), "offset": pos['min']}
        self.map_prediction[str(id)] = {"prediction": pred_.copy(), "offset": pos['min']}

        # --- for display only
        self.final_mask_org = np.zeros(self.dim[1:])
        mask_tmp_height = pred_.shape[0]
        self.final_mask_org[pos['min']:(pos['min'] + mask_tmp_height), :] = pred_

    # ------------------------------------------------------------------------------------------------------------------
    def load_model(self, path_model: str, p):
        """ Loads the trained architecture. """

        netSeg = dilatedUnet(
            input_nc=1,
            output_nc=1,
            n_layers=p.NB_LAYERS,
            ngf=p.NGF,
            kernel_size=p.KERNEL_SIZE,
            padding=p.PADDING,
            use_bias=p.USE_BIAS,
            dropout=0
        )
        netSeg.load_state_dict(torch.load(path_model))
        netSeg.to(self.device)
        netSeg.eval()

        return netSeg

    # ------------------------------------------------------------------------------------------------------------------
    def patch_preprocessing(self, patches: np.ndarray):
        """ Patch preprocessing -> linear histogram between 0 and 1. """

        for k in range(patches.shape[0]):
            tmp = patches[k, ]
            min = tmp.min()
            tmp = tmp - min
            max = tmp.max()
            if max == 0:
                max = 0.1
            tmp = tmp / max
            patches[k, ] = tmp

        return patches