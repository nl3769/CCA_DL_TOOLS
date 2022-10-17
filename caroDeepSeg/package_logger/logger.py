'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import torch
import matplotlib.pyplot            as plt
import numpy                        as np
from torch.utils.tensorboard        import SummaryWriter
import package_utils.fold_handler   as pufh

# ----------------------------------------------------------------------------------------------------------------------
class loggerClassSeg():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, keys_metrics_seg):

        self.p = p
        self.total_steps = 0
        self.loss_seg = {'training': [],
                         'validation': []}
        self.metrics_seg = {'training': {},
                            'validation': {}}

        for key in self.metrics_seg.keys():
            for key_metrics in keys_metrics_seg:
                self.metrics_seg[key][key_metrics] = []

        self.validation_loss = {}
        self.history_model = {'training_based': [],
                              'validation_based': []}
        self.writer = None

    # ------------------------------------------------------------------------------------------------------------------
    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    # ------------------------------------------------------------------------------------------------------------------
    def add_loss_seg(self, loss: float, set: str):
        """ Update the loss at the end of an epoch. """

        self.loss_seg[set].append(loss)

    # ------------------------------------------------------------------------------------------------------------------
    def add_metrics_seg(self, metrics: dict, set: str):
        """ Update the metrics at the end of an epoch. """

        for key in metrics.keys():
            self.metrics_seg[set][key].append(metrics[key])

    # ------------------------------------------------------------------------------------------------------------------
    def plot_loss(self):
        """ Plot the loss for training and evaluation during training. """

        #############################
        # --- LOSS SEGMENTATION --- #
        #############################

        # --- validation
        if len(self.loss_seg['validation']) != 0:
            epoch = list(range(0, len(self.loss_seg['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_seg['validation'], color='b')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_validation_BCEDice_seg.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_seg['training']) != 0:
            epoch = list(range(0, len(self.loss_seg['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_seg['training'], color='r')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_training_BCEDice_seg.png'), dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_metrics(self):
        """ Plot the metrics for training and evaluation during training. """

        #######################
        # --- SEG METRICS --- #
        #######################

        # --- validation
        if len(self.loss_seg['validation']) != 0:
            epoch = list(range(0, len(self.loss_seg['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics_seg['validation'].keys():
                plt.plot(epoch, self.metrics_seg['validation'][key])

            plt.title('Metrics during training (evaluation)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            legend_ = self.metrics_seg['validation'].keys()
            legend_ = [key.replace('_', '') for key in legend_]
            plt.legend(legend_)
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'metrics_validation_seg.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_seg['training']) != 0:
            epoch = list(range(0, len(self.loss_seg['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics_seg['training'].keys():
                plt.plot(epoch, self.metrics_seg['training'][key])

            plt.title('Metrics during training (training)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            legend_ = self.metrics_seg['training'].keys()
            legend_ = [key.replace('_', '') for key in legend_]
            plt.legend(legend_)
            plt.legend(legend_)
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'metrics_training_seg.png'), dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self.writer.close()

    # ------------------------------------------------------------------------------------------------------------------
    def model_history(self, set: str, string: str):
        """ Add information for which the model was updated. """
        self.history_model[set].append(string)

    # ------------------------------------------------------------------------------------------------------------------
    def save_model_history(self):
        """ Save the model history in a .txt file. It records the epoch and associated loss for which the model was updated. """

        keys = self.history_model.keys()

        # --- save model history
        for key in keys:
            textfile = open(os.path.join(self.p.PATH_MODEL_HISTORY, key + '.txt'), "w")
            for el in self.history_model[key]:
                textfile.write(el + "\n")
            textfile.close()

        # --- save loss
        keys = self.loss_seg.keys()
        for key in keys:
            textfile = open(os.path.join(self.p.PATH_MODEL_HISTORY, 'loss_' + key + '.txt'), "w")
            for id, val in enumerate(self.loss_seg[key]):
                textfile.write(str(id) + " " + str(val) + "\n")

        # --- save metrics
        key_set = self.metrics_seg.keys()
        for set in key_set:
            key_metrics = self.metrics_seg[set].keys()
            for metric in key_metrics:
                textfile = open(os.path.join(self.p.PATH_MODEL_HISTORY, 'metrics_' + metric + "_" + set + '.txt'), "w")
                for id, val in enumerate(self.metrics_seg[set][metric]):
                    textfile.write(str(id) + " " + str(val) + "\n")


    # ------------------------------------------------------------------------------------------------------------------
    def save_best_model(self, epoch, models):
        """ Save the best model.
        Args:
            TODO
        Returns:
            TODO
        """

        loss_train = self.loss_seg['training']

        # --- save the model which minimize the validation loss
        if self.p.VALIDATION:
            loss_val = self.loss_seg['validation']

            if epoch > 0 and loss_val[-1] < np.min(loss_val[:-1]):
                disp = f'Epoch: {epoch} |  training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_VALIDATION SAVED.'
                print(disp)
                for key in models.keys():
                    psave = os.path.join(self.p.PATH_SAVE_MODEL)
                    torch.save(models[key].state_dict(), os.path.join(psave, key + '_val.pth'))
                self.model_history(set='validation_based', string=disp)

        # --- save the model which minimize the training loss
        if epoch > 0 and loss_train[-1] < np.min(loss_train[:-1]):

            if self.p.VALIDATION:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_TRAINING SAVED.'
            else:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | MODEL_TRAINING SAVED.'
            print(disp)
            for key in models.keys():
                psave = os.path.join(self.p.PATH_SAVE_MODEL)
                torch.save(models[key].state_dict(), os.path.join(psave, key + '_train.pth'))
            self.model_history(set='training_based', string=disp)

    # ------------------------------------------------------------------------------------------------------------------
    def plot1Seg(self, I1, M1, M1_pred, psave, fname):

        plt.figure()
        ftsize = 6

        # --- image
        plt.subplot2grid((2, 2), (0, 0), colspan=1)
        plt.imshow(I1, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title(r'I1', fontsize=ftsize)
        # --- real mask
        plt.subplot2grid((2, 2), (0, 1), colspan=1)
        plt.imshow(M1, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('MA GT', fontsize=ftsize)
        # --- predicted mask
        plt.subplot2grid((2, 2), (1, 0), colspan=1)
        plt.imshow(M1_pred, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.title('MA pred', fontsize=ftsize)
        # --- superposition
        superimpose = np.zeros(I1.shape + (3,))
        superimpose[..., 0] = I1 * 255
        superimpose[..., 1] = np.multiply((1-M1), I1) * 255
        M1_pred[M1_pred>0.5] = 1
        M1_pred[M1_pred <= 0.5] = 0
        superimpose[..., 2] = np.multiply((1-M1_pred), I1) * 255
        # superimpose[..., 2] = I1
        plt.subplot2grid((2, 2), (1, 1), colspan=1)
        plt.imshow(superimpose.astype(np.int))
        plt.colorbar()
        plt.axis('off')
        plt.title('MA pred', fontsize=ftsize)

        # --- save fig and close

        plt.savefig(os.path.join(psave, fname.replace('/', '_').replace('pkl', 'png')), bbox_inches='tight', dpi=1000)
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------
