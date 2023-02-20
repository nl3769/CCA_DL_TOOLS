'''
@Author  :   <Nolann Lainé>
@Contact :   <nolann.laine@outlook.fr>
'''

import os
import math

# ----------------------------------------------------------------------------------------------------------------------
class splitData():

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, info):

        # size in % of the three subset
        self.training_size = info['training_size']/100
        self.validation_size = info['validation_size']/100

        # name of the subdataset
        self.subset = info['subdataset']

        # path to the data and save data
        self.pdata = info['pres']
        self.psave = info['psave']

        # dictionnary to store path
        self.split = {'training': [],
                      'validation': [],
                      'testing': []}

    # ------------------------------------------------------------------------------------------------------------------
    def split_data(self):
        """ Split data according to size in %. """

        for subset in self.subset:
            files = os.listdir(os.path.join(self.pdata, subset))
            nb_files = len(files)
            # --- get id
            id_training = math.floor(nb_files * self.training_size)
            id_validation = math.floor(nb_files * self.validation_size)
            # --- split data
            training = files[0:id_training]
            validation = files[id_training:id_training+id_validation]
            testing = files[id_training+id_validation:]

            self.split['training'].append([os.path.join(subset, key) for key in training])
            self.split['validation'].append([os.path.join(subset, key) for key in validation])
            self.split['testing'].append([os.path.join(subset, key) for key in testing])

    # ------------------------------------------------------------------------------------------------------------------
    def save_res(self):
        """ Save result in .txt file. """

        for set in self.split.keys():
            with open(os.path.join(self.psave, set + '.txt'), 'w') as f:
                for id in range(len(self.split[set])):
                    for fname in self.split[set][id]:
                        f.write(fname + '\n')

# ----------------------------------------------------------------------------------------------------------------------