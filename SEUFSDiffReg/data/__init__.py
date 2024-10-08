'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(dataset,
                                           batch_size = dataset_opt['batch_size'],
                                           shuffle = dataset_opt['use_shuffle'],
                                           num_workers = dataset_opt['num_workers'],
                                           pin_memory = True)
    elif phase == 'test':
        return torch.utils.data.DataLoader(dataset, 
                                           batch_size=1, 
                                           shuffle=False, 
                                           num_workers=1, 
                                           pin_memory=True)
    else:
        raise NotImplementedError('Dataloader [{:s}] is not found.'.format(phase))


def create_dataset_3D(dataset_opt, phase):
    '''create dataset'''
    from data.ACDC_dataset import ACDCDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                split=phase
                )
    return dataset


def create_dataset_ACDC(dataset_opt, phase):
    '''create dataset'''
    from data.ACDC_dataset import ACDCDataset_resize_name as D
    dataset = D(dataroot = dataset_opt['dataroot'],
                split = phase)
    return dataset

