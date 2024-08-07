import logging
logger = logging.getLogger('base')
from .fIQA_model import fIQA_Model as M

def create_model(opt):
    model = opt['model']
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    print('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m

