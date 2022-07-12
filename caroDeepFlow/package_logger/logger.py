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

class loggerClass():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, p, keys_metrics_seg, keys_metrics_flow):

        self.p = p

        self.total_steps = 0

        self.loss_full = {'training': [],
                          'validation': []}

        self.loss_seg = {'training': [],
                         'validation': []}

        self.loss_flow = {'training': [],
                          'validation': []}

        self.metrics_seg = {'training': {},
                            'validation': {}}

        self.metrics_flow = {'training': {},
                             'validation': {}}

        for key in self.metrics_seg.keys():
            for key_metrics in keys_metrics_seg:
                self.metrics_seg[key][key_metrics] = []

        for key in self.metrics_flow.keys():
            for key_metrics in keys_metrics_flow:
                self.metrics_flow[key][key_metrics] = []

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
    def add_loss_full(self, loss: float, set: str):
        """ Update the loss at the end of an epoch. """

        self.loss_full[set].append(loss)

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

        #####################
        # --- LOSS FLOW --- #
        #####################

        # --- validation
        if len(self.loss_flow['validation']) != 0:
            epoch = list(range(0, len(self.loss_flow['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_flow['validation'], color='b')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_validation_EPE_flow.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_flow['training']) != 0:
            epoch = list(range(0, len(self.loss_flow['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_flow['training'], color='r')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_training_EPE_flow.png'), dpi=150)
            plt.close()

        #####################
        # --- LOSS FULL --- #
        #####################

        # --- validation
        if len(self.loss_full['validation']) != 0:
            epoch = list(range(0, len(self.loss_flow['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_full['validation'], color='b')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_validation_full.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_full['training']) != 0:
            epoch = list(range(0, len(self.loss_full['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_full['training'], color='r')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'loss_training_full.png'), dpi=150)
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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_validation_flow.png'), dpi=150)
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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_training_flow.png'), dpi=150)
            plt.close()

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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_validation_seg.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_seg['training']) != 0:
            epoch = list(range(0, len(self.loss_flow['training'])))
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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_training_seg.png'), dpi=150)
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

        for key in keys:
            textfile = open(os.path.join(self.p.PRES, self.p.EXPNAME, key + '.txt'), "w")
            for el in self.history_model[key]:
                textfile.write(el + "\n")
            textfile.close()

    # ------------------------------------------------------------------------------------------------------------------
    def save_best_model(self, epoch, models):
        """ Save the best model. """

        loss_train = self.loss_full['training']

        # --- save the model which minimize the validation loss
        if self.p.VALIDATION:
            loss_val = self.loss_full['validation']

            if epoch > 0 and loss_val[-1] < np.min(loss_val[:-1]):
                disp = f'Epoch: {epoch} |  training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_VALIDATION SAVED.'
                print(disp)
                for key in models.keys():
                    psave = os.path.join(self.p.PRES, self.p.EXPNAME, "saved_models")
                    pufh.create_dir(psave)
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
                psave = os.path.join(self.p.PRES, self.p.EXPNAME, "saved_models")
                pufh.create_dir(psave)
                torch.save(models[key].state_dict(), os.path.join(psave, key + '_train.pth'))
            self.model_history(set='training_based', string=disp)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_pred(self, I1, I2, M1_gt, M2_gt, M1_pred, M2_pred, OF_gt, OF_pred, epoch_id, fname):

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

        psave = os.path.join(self.p.PRES, self.p.EXPNAME, "random_backup_prediction")
        pufh.create_dir(psave)
        fname = fname.replace('/', '_')
        fname = fname.replace('.png', '')
        fname = "epoch_" + str(epoch_id) + "_" + fname + '.png'

        # --- save fig and close
        plt.savefig(os.path.join(psave, fname), bbox_inches='tight', dpi=1000)
        plt.close()

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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_validation_seg.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_seg['training']) != 0:
            epoch = list(range(0, len(self.loss_flow['training'])))
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
            psave = os.path.join(self.p.PRES, self.p.EXPNAME, "figure")
            pufh.create_dir(psave)
            plt.savefig(os.path.join(psave, 'metrics_training_seg.png'), dpi=150)
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

        for key in keys:
            textfile = open(os.path.join(self.p.PRES, self.p.EXPNAME, key + '.txt'), "w")
            for el in self.history_model[key]:
                textfile.write(el + "\n")
            textfile.close()

    # ------------------------------------------------------------------------------------------------------------------
    def save_best_model(self, epoch, models):
        """ Save the best model. """

        loss_train = self.loss_full['training']

        # --- save the model which minimize the validation loss
        if self.p.VALIDATION:
            loss_val = self.loss_full['validation']

            if epoch > 0 and loss_val[-1] < np.min(loss_val[:-1]):
                disp = f'Epoch: {epoch} |  training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_VALIDATION SAVED.'
                print(disp)
                for key in models.keys():
                    psave = os.path.join(self.p.PRES, self.p.EXPNAME, "saved_models")
                    pufh.create_dir(psave)
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
                psave = os.path.join(self.p.PRES, self.p.EXPNAME, "saved_models")
                pufh.create_dir(psave)
                torch.save(models[key].state_dict(), os.path.join(psave, key + '_train.pth'))
            self.model_history(set='training_based', string=disp)

    # ------------------------------------------------------------------------------------------------------------------
    def plot_pred(self, I1, I2, M1_gt, M2_gt, M1_pred, M2_pred, OF_gt, OF_pred, epoch_id, fname):

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

        psave = os.path.join(self.p.PRES, self.p.EXPNAME, "random_backup_prediction")
        pufh.create_dir(psave)
        fname = fname.replace('/', '_')
        fname = fname.replace('.png', '')
        fname = "epoch_" + str(epoch_id) + "_" + fname + '.png'

        # --- save fig and close
        plt.savefig(os.path.join(psave, fname), bbox_inches='tight', dpi=1000)
        plt.close()

# ----------------------------------------------------------------------------------------------------------------------
