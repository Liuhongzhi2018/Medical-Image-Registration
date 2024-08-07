import logging
logger = logging.getLogger('base')
from .fIQA_model import fIQA_Model as M

def create_model(opt):
    model = opt['model']
    # if model == 'MCM_IQA' or model == 'DISTS':
    #     # /home/liuhongzhi/Method/IQA/SEU-IQA/codes/models/IQA_model.py
    #     from .IQA_model import IQA_Model as M
    # else:
    #     raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    # Model [IQA_Model] is created
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

# ****** model: None 
# opt: {'name': 'LPIPS-Alex_TonPIPAL', 'use_tb_logger': True, 'gpu_ids': 'None', 
# 'datasets': {'test': {'test_PIPAL_Part': {'name': 'PIPAL', 
# 'ref_root': '/home/liuhongzhi/Data/PIPAL/validation_part/Reference_valid', 
# 'dist_root': '/home/liuhongzhi/Data/PIPAL/validation_part/Distortion_valid'}}}, 
# 'network_G': {'FENet': 'Alex', 'tune_flag': False, 'bias_flag': True, 'lpips_flag': False}, 
# 'path': {'pretrain_model_G': '/home/liuhongzhi/Method/IQA/SEU-IQA/experiments/LPIPS-Alex_TonPIPAL_2024-07-22-18-00-27/models/latest_G.pth', 
# 'strict_load': True, 'root': '/public/liuhongzhi/Method/IQA/SEU-IQA', 
# 'results_root': '/public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL', 
# 'log': '/public/liuhongzhi/Method/IQA/SEU-IQA/results/LPIPS-Alex_TonPIPAL'}, 'is_train': False}
