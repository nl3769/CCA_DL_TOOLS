'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os.path
import torch
import numpy                    as np
import matplotlib.pyplot        as plt
from torch.utils.tensorboard    import SummaryWriter
 
class loggerClass():

    def __init__(self, model, scheduler, p):

        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.early_stop = p.EARLY_STOP
        self.early_stop_criterium = 0
        self.pres = p.PATH_RES
        self.validation = p.VALIDATION
        self.path_save_model = p.PATH_SAVE_MODEL
        self.path_save_model_history = p.PATH_MODEL_HISTORY
        self.path_save_figure = p.PATH_SAVE_FIGURE

        self.loss_generator = \
            {'training': [],
            'validation': []}

        self.loss_discriminator = \
            {'training': [],
            'validation': []}

        self.loss_org = \
            {'training': {'loss_GAN': [],   'loss_pixel': []},
            'validation' : {'loss_GAN': [], 'loss_pixel': []}}

        self.validation_loss = {}

        self.history_model = \
            {'training_based': [],
            'validation_based': []}

        self.metrics = \
            {'training': {'l1': [],   'l2': []},
             'validation': {'l1': [], 'l2': []}}

        self.writer = None

    # ------------------------------------------------------------------------------------------------------------------
    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    # ------------------------------------------------------------------------------------------------------------------
    def add_loss(self, loss_gen: float, loss_disc: float, loss_org: dict, set: str):
        """ Update the loss at the end of an epoch. """

        self.loss_generator[set].append(loss_gen)
        self.loss_discriminator[set].append(loss_disc)
        self.loss_org[set]['loss_GAN'].append(loss_org['loss_GAN'])
        self.loss_org[set]['loss_pixel'].append(loss_org['loss_pixel'])

    # ------------------------------------------------------------------------------------------------------------------
    def add_metrics(self, metrics: dict, set: str):
        """ Update the metrics at the end of an epoch. """

        for key in metrics.keys():
            self.metrics[set][key].append(metrics[key])

    # ------------------------------------------------------------------------------------------------------------------
    def plot_loss(self):
        """ Plot the loss for training and evaluation during training. """
        
        # --- validation
        if len(self.loss_generator['validation']) != 0:
            epoch = list(range(0, len(self.loss_generator['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_generator['validation'], color='b')
            plt.title('Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(self.path_save_figure, 'loss_validation.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_generator['training']) != 0:
            epoch = list(range(0, len(self.loss_generator['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_generator['training'], color='r')
            plt.title('Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.savefig(os.path.join(self.path_save_figure, 'loss_training.png'), dpi=150)
            plt.close()

        # --- training
        cond1 = len(self.loss_generator['training']) != 0
        cond2 = len(self.loss_generator['validation']) != 0
        cond3 = len(self.loss_generator['training']) == len(self.loss_generator['validation'])

        if cond1 and cond2 and cond3:
            epoch = list(range(0, len(self.loss_generator['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            plt.plot(epoch, self.loss_generator['training'], color='r', label='training')
            plt.plot(epoch, self.loss_generator['validation'], color='b', label='validation')
            plt.title('Loss - training and validation')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['training', 'validation'])
            plt.savefig(os.path.join(self.path_save_figure, 'loss_training_validation.png'), dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------------------------------------
    def plot_metrics(self):
        """ Plot the metrics for training and evaluation during training. """

        # --- validation
        if len(self.loss_generator['validation']) != 0:
            epoch = list(range(0, len(self.loss_generator['validation'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics['validation'].keys():
                plt.plot(epoch, self.metrics['validation'][key])

            plt.title('Metrics during training (evaluation)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend(self.metrics['validation'].keys())
            plt.savefig(os.path.join(self.path_save_figure, 'metrics_validation.png'), dpi=150)
            plt.close()

        # --- training
        if len(self.loss_generator['training']) != 0:
            epoch = list(range(0, len(self.loss_generator['training'])))
            plt.figure()
            fig = plt.gcf()
            fig.set_size_inches(8, 6)
            for key in self.metrics['training'].keys():
                plt.plot(epoch, self.metrics['training'][key])

            plt.title('Metrics during training (training)')
            plt.xlabel('Epoch')
            plt.ylabel('Metrics')
            plt.legend(self.metrics['training'].keys())
            plt.savefig(os.path.join(self.path_save_figure, 'metrics_training.png'), dpi=150)
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
            textfile = open(os.path.join(self.path_save_model_history, key + '.txt'), "w")
            for el in self.history_model[key]:
                textfile.write(el + "\n")
            textfile.close()

    # ------------------------------------------------------------------------------------------------------------------
    def save_best_model(self, epoch, model):
        """ Save the best model. """


        self.early_stop_criterium += 1
        loss_val = self.loss_generator['validation']
        loss_train = self.loss_generator['training']

        # --- save the model which minimize the validation loss
        if self.validation:
            if epoch > 0 and loss_val[-1] < np.min(loss_val[:-1]):
                disp = f'Epoch: {epoch} |  training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_VALIDATION SAVED.'
                print(disp)
                torch.save(model.state_dict(), os.path.join(self.path_save_model, 'model_validation.pth'))
                self.model_history(set='validation_based', string=disp)
                self.early_stop_criterium = 0

        # --- save the model which minimize the training loss
        if epoch > 0 and loss_train[-1] < np.min(loss_train[:-1]):
            if self.validation:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | validation loss: {loss_val[-1]} | MODEL_TRAINING SAVED.'
            else:
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | MODEL_TRAINING SAVED.'
            print(disp)
            torch.save(model.state_dict(), os.path.join(self.path_save_model, 'model_training.pth'))
            self.model_history(set='training_based', string=disp)

        if epoch == 0:
            if self.validation:
                torch.save(model.state_dict(), os.path.join(self.path_save_model, 'model_training.pth'))
                torch.save(model.state_dict(), os.path.join(self.path_save_model, 'model_validation.pth'))
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | MODEL_TRAINING SAVED.'
                self.model_history(set='validation_based', string=disp)
                self.model_history(set='training_based', string=disp)
            else:
                torch.save(model.state_dict(), os.path.join(self.path_save_model, 'model_training.pth'))
                disp = f'Epoch: {epoch} | training loss: {loss_train[-1]} | MODEL_TRAINING SAVED.'
                self.model_history(set='training_based', string=disp)

    # ------------------------------------------------------------------------------------------------------------------
    def display_loss(self, epoch: int):
        """ Display loss at the end of each epoch. """

        # --- display loss
        if self.validation:
            loss_GAN_train = self.loss_org['training']['loss_GAN'][-1]
            loss_pixel_train = self.loss_org['training']['loss_pixel'][-1]
            loss_GAN_val = self.loss_org['validation']['loss_GAN'][-1]
            loss_pixel_val = self.loss_org['validation']['loss_pixel'][-1]

            training_loss = self.loss_generator['training'][-1]
            validation_loss = self.loss_generator['validation'][-1]


            print(f'EPOCH {epoch} --- training loss: {training_loss}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')
            print(f'EPOCH {epoch} --- validation loss: {validation_loss}  / loss_GAN: {loss_GAN_val} / loss_pixel: {loss_pixel_val}')
        else:
            loss_GAN_train = self.loss_org['training']['loss_GAN'][-1]
            loss_pixel_train = self.loss_org['training']['loss_pixel'][-1]
            training_loss = self.loss_generator['training'][-1]
            print(f'EPOCH {epoch} --- training loss: {training_loss}  / loss_GAN: {loss_GAN_train} / loss_pixel: {loss_pixel_train}')

    # ------------------------------------------------------------------------------------------------------------------
    def get_criterium_early_stop(self):

        criterium = False

        if self.early_stop_criterium > self.early_stop:
            criterium = True

        print(f'self.early_stop_criterium = {self.early_stop_criterium}')
        return criterium