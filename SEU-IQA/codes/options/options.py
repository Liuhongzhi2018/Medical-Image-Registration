import os
import os.path as osp
import logging
import yaml
import time
from utils.util import OrderedYaml
Loader, Dumper = OrderedYaml()


def parse(opt_path, is_train=True):
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    
    # # export CUDA_VISIBLE_DEVICES
    # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    # print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    # mode
    opt['is_train'] = is_train

    # obtain datasets
    # print(opt['datasets'])
    # for phase, dataset in opt['datasets'].items():
    #     phase = phase.split('_')[0]
    #     print(phase)
    #     dataset['phase'] = phase

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path'] and key != 'strict_load':
            opt['path'][key] = osp.expanduser(path)
    opt['path']['root'] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if is_train:
        # Naming in the original code
        # experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name'])
        experiments_root = osp.join(opt['path']['root'], 'experiments', opt['name']+'_'+curr_time)
        opt['path']['experiments_root'] = experiments_root
        opt['path']['models'] = osp.join(experiments_root, 'models')
        opt['path']['training_state'] = osp.join(experiments_root, 'training_state')
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = osp.join(experiments_root, 'val_images')

        # change some options for debug mode
        if 'debug' in opt['name']:
            opt['train']['val_freq'] = 8
            opt['logger']['print_freq'] = 1
            opt['logger']['save_checkpoint_freq'] = 8
    else:  # test
        results_root = osp.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        
    # print(f"option: {opt}")

    return opt

# option: OrderedDict([('name', 'LPIPS-Alex_TonPIPAL'), ('use_tb_logger', True), ('gpu_ids', [1]), 
# ('datasets', OrderedDict([('train', OrderedDict([('choice', 'PIPAL'), 
# ('train_PIPAL', OrderedDict([('phase', 'train'), ('name', 'PIPAL'), 
# ('mos_root', '/home/liuhongzhi/Data/PIPAL/training_part/MOS_Scores_train'), 
# ('ref_root', '/home/liuhongzhi/Data/PIPAL/training_part/Reference_train'), 
# ('dist_root', '/home/liuhongzhi/Data/PIPAL/training_part/Distortion'), 
# ('n_workers', 8), ('batch_size', 16), ('crop_flag', True), ('crop_size', 248), ('norm_flag', True)])), 
# ('train_BAPPS', OrderedDict([('phase', 'train'), ('name', 'BAPPS'), 
# ('train_valid', 'both'), ('train_root', '/home/liuhongzhi/Data/BAPPS/2afc/train'), 
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
# ('logger', OrderedDict([('print_freq', 50), ('save_checkpoint_freq', 2000.0)])), ('is_train', True)])

# option: OrderedDict([('name', 'LPIPS-Alex_TonPIPAL'), ('use_tb_logger', True), ('gpu_ids', [0]), 
# ('datasets', OrderedDict([('test', OrderedDict([('test_PIPAL_Full', OrderedDict([('name', 'PIPAL'), 
# ('ref_root', '/home/liuhongzhi/Data/PIPAL/validation_part/Reference_valid'), 
# ('dist_root', '/home/liuhongzhi/Data/PIPAL/validation_part/Distortion_valid')]))]))])), 
# ('network_G', OrderedDict([('FENet', 'Alex'), ('tune_flag', False), ('bias_flag', True), ('lpips_flag', False)])), 
# ('path', OrderedDict([('pretrain_model_G', '/home/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL_2024-07-14-20-56-38/models/2000_G.pth'), 
# ('strict_load', True),
# ('root', '/public/liuhongzhi/Method/IQA/SEU-IQA'), 
# ('results_root', '/public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL'), 
# ('log', '/public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL')])), ('is_train', False)])


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def check_resume(opt, resume_iter):
    '''Check resume states and pretrain_model paths'''
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is not None or opt['path'].get('pretrain_model_D', None) is not None:
            logger.warning('pretrain_model path will be ignored when resuming training.')

        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'], '{}_G.pth'.format(resume_iter))
        logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])
