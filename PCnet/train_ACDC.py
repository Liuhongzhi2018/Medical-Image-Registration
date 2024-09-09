import glob
# from torch.utils.tensorboard import SummaryWriter
import os, losses, utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from natsort import natsorted
from models import PCNet
import random
import time
import logging
import SimpleITK as sitk
import scipy
import statistics
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, \
                                     binary_erosion,\
                                     generate_binary_structure


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


same_seeds(24)
class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def jacobian_determinant(disp):
    disp = np.expand_dims(disp.transpose(3, 2, 1, 0), 0)
    # print(f"jacobian_determinant disp: {disp.shape}")
    # jacobian_determinant disp: (1, 3, 160, 192, 160)
    _, _, H, W, D = disp.shape
    disp_one = disp[0][0]
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    # return jacdet
    nonpjacdet = np.sum(jacdet <= 0)/np.prod(disp_one.shape)
    return nonpjacdet

# https://docs.monai.io/en/stable/metrics.html
# https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
def Dice(pred, gt):
    # intersection = np.logical_and(gt, pred)
    # union = np.logical_or(gt, pred)
    # dice = 2 * (intersection + smooth)/(mask_sum + smooth)
    smooth = 0.001
    intersection = np.logical_and(pred, gt)
    mask_sum =  np.sum(np.abs(pred)) + np.sum(np.abs(gt))
    # dice = 2.0 * torch.sum(torch.masked_select(y, y_pred))) / (y_o + torch.sum(y_pred))
    dice = 2.0 * (np.sum(intersection) + smooth) / (mask_sum + smooth)
    return dice

# https://github.com/OldaKodym/evaluation_metrics/blob/master/metrics.py
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

# https://ilmonteux.github.io/2019/05/10/segmentation-metrics.html
def IOU(pred, gt, classes=1):
    '''
    Intersection over Union (IoU) and Jaccard coefficients (or indices)
    The Jaccard index is also known as Intersection over Union (IoU)
    '''
    smooth = 0.001
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = (np.sum(intersection) + smooth) / (np.sum(union) + smooth)
    return iou

# https://github.com/JHU-MedImage-Reg/LUMIR_L2R/blob/ecfd6f368e9c8e64417120ddbe06562054757a89/L2R_LUMIR_Eval/utils.py
def calc_TRE(dfm_lms, fx_lms, spacing_mov=1):
    '''
    Target Registration Error (TRE)
    '''
    x = np.linspace(0, fx_lms.shape[0] - 1, fx_lms.shape[0])
    y = np.linspace(0, fx_lms.shape[1] - 1, fx_lms.shape[1])
    z = np.linspace(0, fx_lms.shape[2] - 1, fx_lms.shape[2])
    yv, xv, zv = np.meshgrid(y, x, z)
    unique = np.unique(fx_lms)
    smooth = 0.001
    dfm_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (dfm_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + smooth)
        yc = np.sum(label * yv) / (np.sum(label) + smooth)
        zc = np.sum(label * zv) / (np.sum(label) + smooth)
        dfm_pos[i - 1, 0] = xc
        dfm_pos[i - 1, 1] = yc
        dfm_pos[i - 1, 2] = zc

    fx_pos = np.zeros((len(unique) - 1, 3))
    for i in range(1, len(unique)):
        label = (fx_lms == unique[i]).astype('float32')
        xc = np.sum(label * xv) / (np.sum(label) + smooth)
        yc = np.sum(label * yv) / (np.sum(label) + smooth)
        zc = np.sum(label * zv) / (np.sum(label) + smooth)
        fx_pos[i - 1, 0] = xc
        fx_pos[i - 1, 1] = yc
        fx_pos[i - 1, 2] = zc

    dfm_fx_error = np.mean(np.sqrt(np.sum(np.power((dfm_pos - fx_pos)*spacing_mov, 2), 1)))
    # print(('landmark error (vox): after {}'.format(dfm_fx_error)))
    return dfm_fx_error

def translabel(img):
    seg_table = [0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 
                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62, 63, 64, 65, 
                    66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
                    101, 102, 121, 122, 161, 162, 163, 164, 165, 166, 181, 182]
    img_out = np.zeros_like(img)
    for i in range(len(seg_table)):
        img_out[img == i] = seg_table[i]
    return img_out
        
def compute_per_class_Dice_HD95_IOU_TRE_NDV(pre, gt, gtspacing):
    n_dice_list = []
    n_hd95_list = []
    n_iou_list = []
    class_num = np.unique(gt)
    pred_num = np.unique(pre)
    # print(f"compute_per_class_Dice_HD95_IOU_TRE_NDV: {class_num} {pred_num}")

    for c in class_num:
        if c == 0: continue
        # print(f"class_num {c}")
        ngt_data = np.zeros_like(gt)
        ngt_data[gt == c] = 1
        npred_data = np.zeros_like(pre)
        npred_data[pre == c] = 1
        # n_dice = 2*np.sum(ngt_data*npred_data)/(np.sum(1*ngt_data+npred_data) + 0.0001)
        n_dice = Dice(npred_data, ngt_data)
        n_hd95 = hd95(ngt_data, npred_data, voxelspacing = gtspacing[::-1])
        n_iou = IOU(npred_data, ngt_data)
        n_dice_list.append(n_dice)
        n_hd95_list.append(n_hd95)
        n_iou_list.append(n_iou)
    mean_Dice = statistics.mean(n_dice_list)
    mean_HD95 = statistics.mean(n_hd95_list)
    mean_iou = statistics.mean(n_iou_list)
    
    tre = calc_TRE(ngt_data, npred_data)
    return tre, mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list

def register(epoch, name, output, def_out, y_seg, sample_dir):
    
    # test_txt_path = "/mnt/lhz/Github/Image_registration/RDP/images/LPBA/test_img_seg_list.txt"
    # pairlist = [f.split(' ') for f in read_files_txt(test_txt_path)]
    
    warp_img, warp_flow = output[0], output[1]
    warp_seg = def_out
    # print(f"compute shape: {warp_img.shape} {warp_seg.shape} {warp_flow.shape} {y_seg.shape}")
    # compute shape: torch.Size([1, 1, 64, 64, 64]) torch.Size([1, 1, 64, 64, 64]) torch.Size([1, 3, 64, 64, 64]) torch.Size([1, 1, 64, 64, 64])

    image_path = "/mnt/lhz/Datasets/Learn2reg/LPBA40/test"
    mov_path = os.path.join(image_path, name)
    data_in = sitk.ReadImage(mov_path)
    shape_img = data_in.GetSize()
    ED_origin = data_in.GetOrigin()
    ED_direction = data_in.GetDirection()
    ED_spacing = data_in.GetSpacing()
    
    warp_img = F.interpolate(warp_img, size=shape_img)
    warp_img_array = warp_img.detach().cpu().numpy().squeeze().transpose(2, 1, 0)
    
    warp_flow = F.interpolate(warp_flow, size=shape_img)
    deform = warp_flow.detach().cpu().numpy().squeeze().transpose(3, 2, 1, 0)
    jd = jacobian_determinant(deform)
    
    warp_seg = F.interpolate(warp_seg.float(), size=shape_img)
    warp_seg_array = warp_seg.squeeze().detach().cpu().numpy().transpose(2, 1, 0).astype(np.uint8)
    
    seg_gt = F.interpolate(y_seg.float(), size=shape_img)
    gt_seg_array = seg_gt.squeeze().detach().cpu().numpy().transpose(2, 1, 0).astype(np.uint8)
    
    # print(f"Transpose {shape_img} to: warp_img {warp_img_array.shape} deform: {deform.shape} warp_seg: {warp_seg_array.shape} gt_seg: {gt_seg_array.shape}")
    # Transpose (160, 192, 160) to: warp_img (160, 192, 160) deform: (160, 192, 160, 3) warp_seg: (160, 192, 160) gt_seg: (160, 192, 160)
    
    # print(f"before translabel: {np.unique(warp_seg_array)} {np.unique(gt_seg_array)}")
    # warp_seg_array = translabel(warp_seg_array)
    # gt_seg_array = translabel(gt_seg_array)
    # print(f"after translabel: {np.unique(warp_seg_array)} {np.unique(gt_seg_array)}")
    
    tre, mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list = compute_per_class_Dice_HD95_IOU_TRE_NDV(warp_seg_array, gt_seg_array, ED_spacing)
    
    savedSample_warped = sitk.GetImageFromArray(warp_img_array)
    savedSample_seg = sitk.GetImageFromArray(warp_seg_array)
    savedSample_defm = sitk.GetImageFromArray(deform)
    
    savedSample_warped.SetOrigin(ED_origin)
    savedSample_seg.SetOrigin(ED_origin)
    savedSample_defm.SetOrigin(ED_origin)
    
    savedSample_warped.SetDirection(ED_direction)
    savedSample_seg.SetDirection(ED_direction)
    savedSample_defm.SetDirection(ED_direction)
    
    savedSample_warped.SetSpacing(ED_spacing)
    savedSample_seg.SetSpacing(ED_spacing)
    savedSample_defm.SetSpacing(ED_spacing)
    
    warped_img_path = os.path.join(sample_dir, name.split('.')[0] + '_ep' + str(epoch) + '_warped_img.nii.gz')
    warped_seg_path = os.path.join(sample_dir, name.split('.')[0] + '_ep' + str(epoch) + '_warped_seg.nii.gz')
    warped_flow_path = os.path.join(sample_dir, name.split('.')[0] + '_ep' + str(epoch) + '_warped_deformflow.nii.gz')
    
    sitk.WriteImage(savedSample_warped, warped_img_path)
    sitk.WriteImage(savedSample_seg, warped_seg_path)
    sitk.WriteImage(savedSample_defm, warped_flow_path)
    
    print(f"Saving warped img: {warped_img_path}")
    print(f"Saving warped seg: {warped_seg_path}")
    print(f"Saving warped imgflow: {warped_flow_path}")
    
    return tre, jd, mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list


def main():
    batch_size = 1
    train_file = '/mnt/lhz/Github/Image_registration/PCnet/images/ACDC/train_img_seg_list.txt'
    val_file = '/mnt/lhz/Github/Image_registration/PCnet/images/ACDC/test_img_seg_list.txt'
    checkpoint_dir = '/mnt/lhz/Github/Image_registration/PCnet_checkpoints/ACDC/'
    weights = [1, 1]  # loss weights
    lr = 0.0001
    curr_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    save_dir = curr_time +'_PCnet_ncc_{}_reg_{}_lr_{}/'.format(weights[0], weights[1], lr)
    sample_dir = checkpoint_dir + 'experiments/' + save_dir + "samples"
    if not os.path.exists(checkpoint_dir + 'experiments/' + save_dir):
        os.makedirs(checkpoint_dir + 'experiments/' + save_dir)
    if not os.path.exists(checkpoint_dir + 'logs/' + save_dir):
        os.makedirs(checkpoint_dir + 'logs/' + save_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    sys.stdout = Logger(checkpoint_dir + 'logs/' + save_dir)
    # f = open(os.path.join('logs/'+save_dir, 'losses and dice' + ".txt"), "a")

    # Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(checkpoint_dir + 'logs/' + save_dir, "train.log"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    # logger.info(f"Config: {args}")
    
    epoch_start = 0
    max_epoch = 1000 # 30
    # img_size = (216, 256, 8)
    img_size = (128, 128, 32)
    cont_training = False

    '''
    Initialize model
    '''
    # /mnt/lhz/Github/Image_registration/PCnet/models.py
    # class PCNet(nn.Module)
    model = PCNet(img_size)
    logger.info(f"RDP model: {model}")
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')
    logger.info(f"RDP reg_model: {reg_model}")
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])
    # val_composed = transforms.Compose([trans.Seg_norm(),
    #                                    trans.NumpyType((np.float32, np.int16))])
    val_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16))])
    
    # train_set = datasets.LPBABrainDatasetS2S(glob.glob(train_dir + '*.pkl'), transforms=train_composed)
    train_set = datasets.ACDCDataset(fn=train_file, transforms=train_composed)
    # val_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=val_composed)
    val_set = datasets.ACDCDatasetVal(fn=val_file, transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    best_dsc = 0
    # writer = SummaryWriter(log_dir='logs/'+save_dir)
    best_epoch, best_avg_Dice, best_avg_HD95, best_avg_iou, best_avg_tre = 0, 0, 10000, 0, 10000

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_in = torch.cat((x,y),dim=1)
            x = F.interpolate(x, size=img_size, mode='trilinear')
            y = F.interpolate(y, size=img_size, mode='trilinear')

            output = model(x,y)

            loss = 0
            loss_vals = []
            for n, loss_function in enumerate(criterions):
                curr_loss = loss_function(output[n], y) * weights[n]
                loss_vals.append(curr_loss)
                loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))
            logger.info('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, len(train_loader), loss.item(), loss_vals[0].item(), loss_vals[1].item()))

        # print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        # print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), file=f, end=' ')
        logger.info('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        mdice_list, mhd95_list, mIOU_list, tre_list, jd_list = [], [], [], [], []
        if epoch % 500 == 0:
            with torch.no_grad():
                for data in val_loader:
                    model.eval()
                    # data = [t.cuda() for t in data]
                    data_list = [t.cuda() for t in data if not isinstance(t, list)]
                    # x = data[0]
                    # y = data[1]
                    # x_seg = data[2]
                    # y_seg = data[3]
                    name = data[0][0]
                    x = data_list[0]
                    y = data_list[1]
                    x_seg = data_list[2]
                    y_seg = data_list[3]
                    x = F.interpolate(x, size=img_size, mode='trilinear')
                    y = F.interpolate(y, size=img_size, mode='trilinear')
                    x_seg = F.interpolate(x_seg.half(), size=img_size, mode='trilinear')
                    y_seg = F.interpolate(y_seg.half(), size=img_size, mode='trilinear')
 
                    output = model(x,y)
                    def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                    dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                    eval_dsc.update(dsc.item(), x.size(0))
                    # print(epoch, ':',eval_dsc.avg)
                    logger.info(f"epoch {epoch} eval_dsc: {eval_dsc.avg}")
                    
                    tre, jd, mdice, mhd95, mIOU, dice_list, hd95_list, IOU_list = register(epoch, name, output, def_out, y_seg, sample_dir)
                    
                    logger.info(f"Epoch: {epoch} {name} mean Dice {mdice} - {', '.join(['%.4e' % f for f in dice_list])}")
                    logger.info(f"Epoch: {epoch} {name} mean HD95 {mhd95} - {', '.join(['%.4e' % f for f in hd95_list])}")
                    logger.info(f"Epoch: {epoch} {name} mean IOU {mIOU} - {', '.join(['%.4e' % f for f in IOU_list])}")
                    logger.info(f"Epoch: {epoch} {name} jacobian_determinant - {jd}")
                    
                    mdice_list.append(mdice)
                    mhd95_list.append(mhd95)
                    mIOU_list.append(mIOU)
                    tre_list.append(tre)
                    jd_list.append(jd)
                    
            best_dsc = max(eval_dsc.avg, best_dsc)
            # print(eval_dsc.avg, file=f)
            logger.info(eval_dsc.avg)

            cur_avg_dice, cur_avg_hd95, cur_avg_iou = np.mean(mdice_list), np.mean(mhd95_list), np.mean(mIOU_list)
            cur_meanTre = np.mean(tre_list)
            cur_meanjd = np.mean(jd_list)
            
            logger.info(f"Epoch: {epoch} - avgDice: {cur_avg_dice} avgHD95: {cur_avg_hd95} avgIOU: {cur_avg_iou} avgTRE: {cur_meanTre} avgJD: {cur_meanjd}")    
            # Epoch: 0 - avgDice: 0.3953122517969288 avgHD95: 10.5551533471546 avgIOU: 0.2547768406802452 avgTRE: 5.765068821433447 avgJD: 0.0

            if cur_avg_dice > best_avg_Dice and cur_avg_hd95 < best_avg_HD95 and cur_avg_iou > best_avg_iou:
                best_epoch = epoch
                best_avg_Dice = cur_avg_dice
                best_avg_HD95 = cur_avg_hd95
                best_avg_iou = cur_avg_iou
                # best_avg_tre = cur_meanTre
                save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_dsc': best_dsc,
                                'optimizer': optimizer.state_dict(),}, 
                                save_dir = checkpoint_dir + 'experiments/' + save_dir, 
                                filename ='best.pth.tar')
                # print(f"Saving best model to: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
                logger.info(f"Saving best model to: {os.path.join(checkpoint_dir + 'experiments/' + save_dir, 'best.pth.tar')}")
                # Saving best model to: /mnt/lhz/Github/Image_registration/RDP_checkpoints/LPBA/experiments/2024-09-03-00-56-43_RDP_ncc_1_reg_1_lr_0.0001_54r/best.pth.tar

            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_dsc': best_dsc,
                             'optimizer': optimizer.state_dict(),}, 
                            save_dir=checkpoint_dir + 'experiments/' + save_dir, 
                            filename='ep{}_dsc{:.3f}.pth.tar'.format(epoch, cur_avg_dice))
               
        loss_all.reset()
        
    save_checkpoint({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),}, 
                save_dir=checkpoint_dir + 'experiments/' + save_dir, 
                filename='final.pth.tar')


if __name__ == '__main__':
    '''
    
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()