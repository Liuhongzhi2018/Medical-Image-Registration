def create_model(opt):
    from .model import DDPM as M
    m = M(opt)
    return m


def create_DiffuseMorph_model(opt):
    from .model_DiffuseMorph import DiffuseMorph_DDPM as M
    m = M(opt)
    return m
