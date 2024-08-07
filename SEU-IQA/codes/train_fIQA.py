import os
import math
import torch
import random
import logging
import argparse
import pandas as pd
# /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py
from utils import util
from models import create_model
import torch.distributed as dist
# /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py
import options.options as option
import torch.multiprocessing as mp
from data import create_dataloader, create_dataset
from sklearn.model_selection import KFold


def obtain_MOS_Score(score_root):
    score_list = {}
    fnames = [fname for fname in os.listdir(score_root) if '.txt' in fname]
    for fname in sorted(fnames):
        ELO_path = os.path.join(score_root, fname)
        with open(ELO_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_name, img_MOS = line.split(',')[0], float(line.split(',')[1][:-1])
                score_list[img_name] = img_MOS
    # print(f"obtain_MOS_Score score_listz: {score_list}")
    # obtain_MOS_Score score_listz: {'A0185_00_00.bmp': 1474.3754, 'A0185_00_01.bmp': 1284.9828}
    return score_list


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    os.environ['RANK'] = '0'
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'], map_location=lambda storage, loc: storage.cuda(device_id))
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def check_resume(opt, resume_iter)
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None
    torch.backends.cudnn.benckmark = True
    print(f"*** Option: {opt}")

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model_G' not in key and 'pretrain_model_R' not in key and 'resume' not in key and 'strict_load' not in key))

        # config loggers. Before it, the log will not work
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False)
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def dict2str(opt, indent_l=1)
        logger.info(option.dict2str(opt))
        
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            # /home/liuhongzhi/Method/IQA/SEU-IQA/tb_logger
            # tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
            tb_logger = SummaryWriter(log_dir='/home/liuhongzhi/Method/IQA/SEU-IQA/experiments/tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/options/options.py def dict_to_nonedict(opt)
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/utils/util.py def set_random_seed(seed)
    util.set_random_seed(seed)
    
    #### create model
    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/__init__.py def create_model(opt)
    model = create_model(opt) 

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        
    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 'datasets', OrderedDict([('train', OrderedDict([('choice', 'PIPAL')
            dataset_choice = dataset_opt['choice']
            train_sampler = None
            if dataset_choice == 'PIPAL':
                # ('choice', 'PIPAL')
                # print(f"Training dataset_choice: {dataset_choice}")
                logger.info('Training dataset_choice: {}'.format(dataset_choice))
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataset(dataset_opt, mode)
                train_set = create_dataset(dataset_opt['train_PIPAL'], mode='train')
                train_size = int(math.ceil(len(train_set)) / (dataset_opt['train_PIPAL']['batch_size']))
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataloader(dataset, dataset_opt, opt=None, sampler=None)
                # train_PIPAL_loader = create_dataloader(train_set, dataset_opt['train_PIPAL'], opt, train_sampler)
                # assert train_PIPAL_loader is not None
            elif dataset_choice == 'fIQA':
                # ('choice', 'PIPAL')
                # print(f"Training dataset_choice: {dataset_choice}")
                logger.info('Training dataset_choice: {}'.format(dataset_choice))
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataset(dataset_opt, mode)  
                # from data.PairedTrain_dataset import fIQADataset as D
                train_set = create_dataset(dataset_opt['train_fIQA'], mode='train')
                print(f"train_set {train_set} {len(train_set)}")
                print(f"{dataset_opt['train_fIQA']['batch_size']}")
                train_size = int(math.ceil(len(train_set)) / (dataset_opt['train_fIQA']['batch_size']))
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/data/__init__.py def create_dataloader(dataset, dataset_opt, opt=None, sampler=None)
                # train_PIPAL_loader = create_dataloader(train_set, dataset_opt['fIQA'], opt, train_sampler)
                # assert train_PIPAL_loader is not None
            else:
                raise NotImplementedError('Chosen Training Dataset is not recognized. Only support PIPAL or BAPPS')

            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))

            if rank <= 0:
                if dataset_choice == 'PIPAL':
                    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                elif dataset_choice == 'fIQA':
                    logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(total_epochs, total_iters))
            else:
                raise NotImplementedError(' choice of datasets is not recofnized. ')

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    curr_k = 0
    
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for train_index, valid_index in kf.split(train_set):
            curr_k = (curr_k + 1) % 5
            # print(f"Training train_index {train_index} val_index {val_index}")
            train_fold = torch.utils.data.dataset.Subset(train_set, train_index)
            valid_fold = torch.utils.data.dataset.Subset(train_set, valid_index)
            dataset_opt = opt['datasets']['train']
            train_loader = create_dataloader(train_fold, dataset_opt['train_fIQA'], opt, train_sampler)
            valid_loader = create_dataloader(valid_fold, dataset_opt)
            train_size = len(train_loader)
            valid_size = len(valid_loader)
            # print(f"Epoch: {epoch} train_size: {train_size} valid_size: {valid_size}")
    
            for train_data in train_loader:
                current_step += 1
                if current_step > total_iters:
                    break
    
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def feed_data(self, data, Train=True)
                model.feed_data(train_data)
                # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def optimize_parameters(self, step)
                model.optimize_parameters(current_step)

                #### log
                if current_step % opt['logger']['print_freq'] == 0:
                    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def get_current_log(self)
                    logs = model.get_current_log()
                    # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/base_model.py def get_current_learning_rate(self)
                    message = '< epoch:{:3d}, iter:{:8,d}, lr:{:.3e} Kfold {}> '.format(epoch, current_step, model.get_current_learning_rate(), curr_k)
                    for k, v in logs.items():
                        # print(f"k: {k} v: {v}")
                        message += ' {:s}: {:.4e} '.format(k, v)
                        # tensorboard logger
                        if opt['use_tb_logger'] and 'debug' not in opt['name']:
                            if rank <= 0:
                                tb_logger.add_scalar(k, v, current_step)
                    if rank <= 0:
                        logger.info(message)

                #### save models and training states
                if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                    if rank <= 0:
                        logger.info('Saving models and training states.')
                        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py def save(self, iter_label)
                        model.save(current_step)
                        # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/base_model.py def save_training_state(self, epoch, iter_step)
                        model.save_training_state(epoch, current_step) 
                
                # Validation
                if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                    index = 0
                    IQA_noise_List, IQA_blur_List, IQA_color_List, IQA_contrast_List = [], [], [], []
                    MOS_noise_List, MOS_blur_List, MOS_color_List, MOS_contrast_List = [], [], [], []
                    txt_Fname = os.path.join(opt['path']['val_images'], "{}_{}_iter{}.txt".format(opt['name'], 'test_fIQA_Valid', current_step))
                    logger.info('Validation Testing, Please Wait')
                    print(f"Writing txt_Fname: {txt_Fname}")
                    with open(txt_Fname, 'a') as f:
                        f.write('image name' + ',' + 'Comprehensive' + ',' + 'Noise' + ',' + 'Blur' + ',' + 'Color' + ',' + 'Contrast' + '\n')
                        for val_data in valid_loader:
                            index += 1
                            if index % 100 == 0:
                                logger.info("Processing No.{} Image in PIPAL_Full".format(index))
                            model.feed_data(val_data, Train=False)                            
                            model.test()
                            noise_pre, blur_pre, color_pre, contrast_pre = model.get_current_score()
                            noise_score = float(noise_pre.numpy())
                            blur_score = float(blur_pre.numpy())
                            color_score = float(color_pre.numpy())
                            contrast_score = float(contrast_pre.numpy())
                            Comprehensive_score = (noise_score + blur_score + color_score + contrast_score) / 4
                            Dist_name = val_data['name'][0].split('/')[-1]
                            # print(f"Validation Dist_name: {Dist_name} score: {score}")
                            f.write(Dist_name + ',' + str(Comprehensive_score) + ',' + str(noise_score) + ',' + str(blur_score)+ ',' + str(color_score)+ ',' + str(contrast_score) + '\n')
                            IQA_noise_List.append(noise_score)
                            IQA_blur_List.append(blur_score)
                            IQA_color_List.append(color_score)
                            IQA_contrast_List.append(contrast_score)
                            MOS_noise_List.append(val_data["Dist_scores"][0])
                            MOS_blur_List.append(val_data["Dist_scores"][1])
                            MOS_color_List.append(val_data["Dist_scores"][2])
                            MOS_contrast_List.append(val_data["Dist_scores"][3])
                            
                        # Calculate Correlation between MOS and IQA scores
                        # https://github.com/pandas-dev/pandas/blob/v2.2.2/pandas/core/series.py#L2899-L2985
                        IQA_list_pd = pd.Series(IQA_noise_List)
                        MOS_list_pd = pd.Series(MOS_noise_List)
                        # Spearman rank-order correlation coefficient - Spearman rank correlation
                        SROCC_noise =  abs(MOS_list_pd.corr(IQA_list_pd, method='spearman'))
                        # Pearson linear correlation coefficient - Standard correlation coefficient
                        PLCC_noise = abs(MOS_list_pd.corr(IQA_list_pd, method='pearson'))
                        
                        IQA_list_pd = pd.Series(IQA_blur_List)
                        MOS_list_pd = pd.Series(MOS_blur_List)
                        # Spearman rank-order correlation coefficient - Spearman rank correlation
                        SROCC_blur =  abs(MOS_list_pd.corr(IQA_list_pd, method='spearman'))
                        # Pearson linear correlation coefficient - Standard correlation coefficient
                        PLCC_blur = abs(MOS_list_pd.corr(IQA_list_pd, method='pearson'))

                        IQA_list_pd = pd.Series(IQA_color_List)
                        MOS_list_pd = pd.Series(MOS_color_List)
                        # Spearman rank-order correlation coefficient - Spearman rank correlation
                        SROCC_color = abs(MOS_list_pd.corr(IQA_list_pd, method='spearman'))
                        # Pearson linear correlation coefficient - Standard correlation coefficient
                        PLCC_color = abs(MOS_list_pd.corr(IQA_list_pd, method='pearson'))
                
                        IQA_list_pd = pd.Series(IQA_contrast_List)
                        MOS_list_pd = pd.Series(MOS_contrast_List)
                        # Spearman rank-order correlation coefficient - Spearman rank correlation
                        SROCC_contrast = abs(MOS_list_pd.corr(IQA_list_pd, method='spearman'))
                        # Pearson linear correlation coefficient - Standard correlation coefficient
                        PLCC_contrast = abs(MOS_list_pd.corr(IQA_list_pd, method='pearson'))
                        
                        # Record corr on Tensorboard
                        # print(f"Validation # {valid_name}: PLCC: {PLCC:.4e}_SROCC: {SROCC:.4e}")
                        # logger.info('# Validation # {}_PLCC: {:.4e}_SROCC: {:.4e}'.format(valid_name, PLCC, SROCC))
                        print(f"Validation # {'test_fIQA_Valid'}: Noise PLCC: {PLCC_noise:.4f} Noise SROCC: {SROCC_noise:.4f}")
                        print(f"Validation # {'test_fIQA_Valid'}: Blur PLCC: {PLCC_blur:.4f} Blur SROCC: {SROCC_blur:.4f}")
                        print(f"Validation # {'test_fIQA_Valid'}: Color PLCC: {PLCC_color:.4f} Color SROCC: {SROCC_color:.4f}")
                        print(f"Validation # {'test_fIQA_Valid'}: Contrast PLCC: {PLCC_contrast:.4f} Contrast SROCC: {SROCC_contrast:.4f}")
                        logger.info('# Validation # {} Noise PLCC: {:.4f} Noise SROCC: {:.4f}'.format('test_fIQA_Valid', PLCC_noise, SROCC_noise))
                        logger.info('# Validation # {} Blur PLCC: {:.4f} Blur SROCC: {:.4f}'.format('test_fIQA_Valid', PLCC_blur, SROCC_blur))
                        logger.info('# Validation # {} Color PLCC: {:.4f} Color SROCC: {:.4f}'.format('test_fIQA_Valid', PLCC_color, SROCC_color))
                        logger.info('# Validation # {} Contrast PLCC: {:.4f} Contrast SROCC: {:.4f}'.format('test_fIQA_Valid', PLCC_contrast, SROCC_contrast))

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')

if __name__ == '__main__':
    main()


# *** Option: OrderedDict([('name', 'LPIPS-Alex_TonPIPAL'), ('use_tb_logger', True), ('gpu_ids', [1]), 
# ('datasets', OrderedDict([('train', OrderedDict([('choice', 'PIPAL'),
# ('train_PIPAL', OrderedDict([('phase', 'train'), ('name', 'PIPAL'), 
# ('mos_root', '/home/liuhongzhi/Data/PIPAL/training_part/MOS_Scores_train'), 
# ('ref_root', '/home/liuhongzhi/Data/PIPAL/training_part/Reference_train'), 
# ('dist_root', '/home/liuhongzhi/Data/PIPAL/training_part/Distortion'), 
# ('n_workers', 8), ('batch_size', 16), ('crop_flag', True), ('crop_size', 248), ('norm_flag', True)])), 
# ('train_BAPPS', OrderedDict([('phase', 'train'), ('name', 'BAPPS'), ('train_valid', 'both'), 
# ('train_root', '/home/liuhongzhi/Data/BAPPS/2afc/train'), 
# ('valid_root', '/home/liuhongzhi/Data/BAPPS/2afc/val'), 
# ('n_workers', 8), ('batch_size', 16), ('crop_flag', True), ('crop_size', 248), ('norm_flag', True)]))])), 
# ('val', OrderedDict([('test_PIPAL_Full', OrderedDict([('name', 'PIPAL'), 
# ('mos_root', '/home/liuhongzhi/Data/PIPAL/training_valid_full/MOS_Scores'), 
# ('ref_root', '/home/liuhongzhi/Data/PIPAL/training_valid_full/Reference'), 
# ('dist_root', '/home/liuhongzhi/Data/PIPAL/training_valid_full/Distortion')])), 
# ('test_PIPAL_Valid', OrderedDict([('name', 'PIPAL'), 
# ('mos_root', '/home/liuhongzhi/Data/PIPAL/validation_part/MOS_Scores_valid'), 
# ('ref_root', '/home/liuhongzhi/Data/PIPAL/validation_part/Reference_valid'), 
# ('dist_root', '/home/liuhongzhi/Data/PIPAL/validation_part/Distortion_valid')]))]))])), 
# ('network_G', OrderedDict([('FENet', 'Alex'), ('tune_flag', False), ('bias_flag', True), ('lpips_flag', False)])), 
# ('path', OrderedDict([('pretrain_model_G', None), ('pretrain_model_R', None), ('strict_load', True), ('resume_state', None), 
# ('root', '/public/liuhongzhi/Method/IQA/SEU-IQA'), 
# ('experiments_root', '/public/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL'), 
# ('models', '/public/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL/models'), 
# ('training_state', '/public/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL/training_state'), 
# ('log', '/public/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL'), 
# ('val_images', '/public/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL/val_images')])),
# ('train', OrderedDict([('lr_IQA', 0.0001), ('weight_decay_IQA', 0), ('niter', 400000), ('warmup_iter', -1), ('loss_weight', 1.0), ('manual_seed', 10), ('val_freq', 2000.0)])), 
# ('logger', OrderedDict([('print_freq', 50), ('save_checkpoint_freq', 2000.0)])), 
# ('is_train', True), ('dist', False)])
