'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import torch
import matplotlib
import matplotlib.pyplot            as plt
import numpy                        as np
from torch.utils.tensorboard        import SummaryWriter

matplotlib.use('Agg')

class loggerClass():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, keys_metrics_flow):

        self.p = p

        self.total_steps = 0

        if keys_metrics_flow is not None:
            self.loss_flow = {'training': [], 'validation': []}
            self.metrics_flow = {'training': {}, 'validation': {}}

            for key in self.metrics_flow.keys():
                for key_metrics in keys_metrics_flow:
                    self.metrics_flow[key][key_metrics] = []

        self.validation_loss = {}
        self.history_model = {'trn_based': [], 'val_based': []}
        self.history_loss = {'trn_based': [], 'val_based': []}

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
    def add_loss_flow(self, loss: float, set: str):
        """ Update the loss at the end of an epoch. """

        self.loss_flow[set].append(loss)

    # ------------------------------------------------------------------------------------------------------------------
    def add_metrics_flow(self, metrics: dict, set: str):
        """ Update the metrics at the end of an epoch. """

        for key in metrics.keys():
            self.metrics_flow[set][key].append(metrics[key])

    # ------------------------------------------------------------------------------------------------------------------
    def plot_loss(self):
        """ Plot the loss for training and evaluation during training. """

        magnitude = np.max(self.loss_flow['validation'] + self.loss_flow['training'])

        # --- validation
        if len(self.loss_flow['validation']) != 0:

            epoch = list(range(0, len(self.loss_flow['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_flow['validation'], color='r')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim((0, magnitude))
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'loss_val_EPE.png'), dpi=150)
            plt.close()

            epoch = list(range(0, len(self.loss_flow['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_flow['validation'], color='r', label='val')
            plt.plot(epoch, self.loss_flow['training'], color='g', label='trn')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim((0, magnitude))
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'loss_trn_val_EPE.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_flow['training']) != 0:
            epoch = list(range(0, len(self.loss_flow['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_flow['training'], color='g')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim((0, magnitude))
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'loss_trn_EPE.png'), dpi=150)
            plt.close()


    # ------------------------------------------------------------------------------------------------------------------
    def plot_metrics(self):
        """ Plot the metrics for training and evaluation during training. """

        ########################
        # --- FLOW METRICS --- #
        ########################

        # --- validation
        if len(self.loss_flow['validation']) != 0:
            epoch = list(range(0, len(self.loss_flow['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics_flow['validation'].keys():
                plt.plot(epoch, self.metrics_flow['validation'][key])

            plt.title('Metrics during training (evaluation)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend(self.metrics_flow['validation'].keys())
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'metrics_validation_flow.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_flow['training']) != 0:
            epoch = list(range(0, len(self.loss_flow['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics_flow['training'].keys():
                plt.plot(epoch, self.metrics_flow['training'][key])

            plt.title('Metrics during training (training)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend(self.metrics_flow['training'].keys())
            plt.savefig(os.path.join(self.p.PATH_SAVE_FIGURE, 'metrics_training_flow.png'), dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def close(self):
        self.writer.close()

    # ------------------------------------------------------------------------------------------------------------------
    def model_history(self, set: str, string: str):
        """ Add information for which the model was updated. """
        self.history_model[set].append(string)

    # ------------------------------------------------------------------------------------------------------------------
    def update_history_loss(self):
        """ save loss a the end of each epoch. """

        if self.p.VALIDATION:
            self.history_loss['val_based'].append(str(self.loss_flow['validation'][-1]))
            self.history_loss['trn_based'].append(str(self.loss_flow['training'][-1]))
        else:
            self.history_loss['trn_based'].append(str(self.loss_flow['training'][-1]))

    # ------------------------------------------------------------------------------------------------------------------
    def save_model_history(self):
        """ Save the model history in a .txt file. It records the epoch and associated loss for which the model was updated. """

        # --- indicate when model was saved during training
        keys = self.history_model.keys()

        for key in keys:
            textfile = open(os.path.join(self.p.PATH_MODEL_HISTORY, key + '_epochs_saved.txt'), "w")
            for el in self.history_model[key]:
                textfile.write(el + "\n")
            textfile.close()

        # --- indicate when model was saved during training
        keys = self.history_loss.keys()
        for key in keys:
            textfile = open(os.path.join(self.p.PATH_MODEL_HISTORY, key + '_loss_motion.txt'), "w")
            for el in self.history_loss[key]:
                textfile.write(el + "\n")
            textfile.close()




    # ------------------------------------------------------------------------------------------------------------------
    def save_best_model(self, epoch, models):
        """ Save the best model. """

        loss_train = self.loss_flow['training']

        # --- save the model which minimize the validation loss
        if self.p.VALIDATION:
            loss_val = self.loss_flow['validation']

            if epoch > 0 and loss_val[-1] < np.min(loss_val[:-1]):
                disp = f'Epoch: {epoch} |  training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_VALIDATION SAVED.'
                print(disp)
                for key in models.keys():
                    torch.save(models[key].state_dict(), os.path.join(self.p.PATH_SAVE_MODEL, key + '_val.pth'))
                self.model_history(set='val_based', string=disp)

        # --- save the model which minimize the training loss
        if epoch > 0 and loss_train[-1] < np.min(loss_train[:-1]):

            if self.p.VALIDATION:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_TRAINING SAVED.'
            else:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | MODEL_TRAINING SAVED.'
            print(disp)
            for key in models.keys():
                torch.save(models[key].state_dict(), os.path.join(self.p.PATH_SAVE_MODEL, key + '_train.pth'))
            self.model_history(set='trn_based', string=disp)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_pred(self, I1, I2, M1_gt, M2_gt, M1_pred, M2_pred, OF_gt, OF_pred, epoch_id, psave, set, fname):

        ftsize = 6
        plt.rcParams['text.usetex'] = True
        plt.figure()

        # ------ IMAGE
        # -- I1
        plt.subplot2grid((4, 4), (0, 0), colspan=2)
        plt.imshow(I1, cmap='gray')
        plt.axis('off')
        plt.title(r'I1', fontsize=ftsize)
        # -- I1
        plt.subplot2grid((4, 4), (0, 2), colspan=2)
        plt.imshow(I2, cmap='gray')
        plt.axis('off')
        plt.title(r'I2', fontsize=ftsize)

        # ------ MASK
        # -- M1 GT
        plt.subplot2grid((4, 4), (1, 0), colspan=1)
        plt.imshow(M1_gt, cmap='gray')
        plt.axis('off')
        plt.title(r'M1 GT', fontsize=ftsize)
        # -- M1 PRED
        plt.subplot2grid((4, 4), (1, 1), colspan=1)
        plt.imshow(M1_pred, cmap='gray')
        plt.axis('off')
        plt.title(r'M1 PRED', fontsize=ftsize)
        # -- M2 GT
        plt.subplot2grid((4, 4), (1, 2), colspan=1)
        plt.imshow(M2_gt, cmap='gray')
        plt.axis('off')
        plt.title(r'M2 GT', fontsize=ftsize)
        # -- M2 PRED
        plt.subplot2grid((4, 4), (1, 3), colspan=1)
        plt.imshow(M2_pred, cmap='gray')
        plt.axis('off')
        plt.title(r'M2 PRED', fontsize=ftsize)

        # ------ FLOW
        OF_gt_norm = np.sqrt(np.power(OF_gt[0,], 2) + np.power(OF_gt[1,], 2))
        OF_pred_norm = np.sqrt(np.power(OF_pred[0,], 2) + np.power(OF_pred[1,], 2))
        vmin = min([np.min(OF_gt_norm), np.min(OF_pred_norm)])
        vmax = min([np.max(OF_gt_norm), np.max(OF_pred_norm)])
        # -- OF GT
        plt.subplot2grid((4, 4), (2, 0), colspan=2,)
        plt.imshow(OF_gt_norm, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.title(r'OF GT', fontsize=ftsize)

        # -- OF PRED
        plt.subplot2grid((4, 4), (2, 2), colspan=2)
        plt.imshow(OF_pred_norm, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.title(r'OF PRED', fontsize=ftsize)

        plt.tight_layout()

        fname = fname.replace('/', '_')
        fname = fname.replace('.png', '')
        fname = set + "_epoch_" + str(epoch_id) + "_" + fname + '.png'

        # --- save fig and close
        plt.savefig(os.path.join(psave, fname), bbox_inches='tight', dpi=1000)
        plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_pred_flow(self, I1, I2, OF_gt, OF_pred, epoch_id, psave, set, fname):
        ftsize = 6
        plt.rcParams['text.usetex'] = True
        plt.figure()

        # ------ IMAGE
        # -- I1
        plt.subplot2grid((2, 2), (0, 0), colspan=1)
        plt.imshow(I1, cmap='gray')
        plt.axis('off')
        plt.title('I1', fontsize=ftsize)
        # -- I1
        plt.subplot2grid((2, 2), (0, 1), colspan=1)
        plt.imshow(I2, cmap='gray')
        plt.axis('off')
        plt.title('I2', fontsize=ftsize)

        # ------ FLOW
        OF_gt_norm = np.sqrt(np.power(OF_gt[0,], 2) + np.power(OF_gt[1,], 2))
        OF_pred_norm = np.sqrt(np.power(OF_pred[0,], 2) + np.power(OF_pred[1,], 2))
        vmin = min([np.min(OF_gt_norm), np.min(OF_pred_norm)])
        vmax = min([np.max(OF_gt_norm), np.max(OF_pred_norm)])
        # -- OF GT
        plt.subplot2grid((2, 2), (1, 0), colspan=1, )
        plt.imshow(OF_gt_norm, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.title(r'OF GT', fontsize=ftsize)

        # -- OF PRED
        plt.subplot2grid((2, 2), (1, 1), colspan=1)
        plt.imshow(OF_pred_norm, cmap='hot')
        plt.axis('off')
        plt.colorbar()
        plt.clim(vmin, vmax)
        plt.title(r'OF PRED', fontsize=ftsize)

        plt.tight_layout()

        fname = fname.replace('/', '_')
        fname = fname.replace('.png', '')
        fname = set + "_epoch_" + str(epoch_id) + "_" + fname + '.png'

        # --- save fig and close
        plt.savefig(os.path.join(psave, fname), bbox_inches='tight', dpi=1000)
        plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_pred_flow_error(self, OF_gt, OF_pred, epoch_id, psave, set, fname):

        ftsize = 5

        fname = fname.replace('/', '_')
        fname = fname.replace('.png', '')
        fname = 'error_' + set + "_epoch_" + str(epoch_id) + "_" + fname + '.png'

        err = OF_pred - OF_gt

        plt.figure()

        plt.subplot2grid((2, 1), (0, 0), colspan=1)
        plt.imshow(err[0, ], cmap='hot')
        plt.title('pred - GT (x)', fontsize=ftsize)
        plt.colorbar()
        plt.axis('off')

        plt.subplot2grid((2, 1), (1, 0), colspan=1)
        plt.imshow(err[1, ], cmap='hot')
        plt.title('pred - GT (Z)', fontsize=ftsize)
        plt.colorbar()
        plt.axis('off')

        plt.savefig(os.path.join(psave, fname), bbox_inches='tight', dpi=1000)
        plt.close()