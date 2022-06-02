import torch
import os

# ----------------------------------------------------------------------------------------------------------------------
def save_print(model, path):
    ''' Save model architecture and number of parameters. '''
    with open(os.path.join(path, 'print_model.txt'), 'w') as f:
        print(model, file=f)
        print('', file=f)
        for i in range(3):
            print('###########################################################################################', file=f)
        print('', file=f)
        print("Parameter Count: %d" % count_parameters(model), file=f)

# ----------------------------------------------------------------------------------------------------------------------
def count_parameters(model):
    ''' Count the number of parameters in the model. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)