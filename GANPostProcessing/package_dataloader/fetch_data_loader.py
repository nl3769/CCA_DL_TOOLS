from package_dataloader.DataHandler import DataHandler
from torch.utils.data import DataLoader

def fetch_dataloader(p, set: str, shuffle, evaluation=False, data_aug = False):
    """ Create the data loader for the corresponding set """

    dataset = DataHandler(p, data_aug=data_aug, set=set)

    if evaluation == True:
        p.BATCH_SIZE = 1
        p.WORKERS = 0

    if set == 'validation' or set == 'testing':
        batch_size = 1
    else:
        batch_size = p.BATCH_SIZE

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        pin_memory=False,
                        shuffle=shuffle,
                        num_workers=p.WORKERS,
                        drop_last=True)

    print(set + ' with %d image ' % len(dataset))

    len_dataset = len(dataset.org_list)

    return loader, len_dataset
