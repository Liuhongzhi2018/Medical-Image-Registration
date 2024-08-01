import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
import shutil
from math import *
import time
from torch.utils import tensorboard
import torch.nn.functional as F
import core.metrics as Metrics
import SimpleITK as sitk
import numpy as np

# third-party modules
import os
import numpy
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr
import csv
import SimpleITK as sitk
import statistics

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

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of 
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    return hd95

def iou(pred, gt, classes=1):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_Dice_HD95(pre, gt, gtspacing):
    n_dice_list = []
    n_hd95_list = []
    class_num = np.unique(gt)
    # print(f"compute_Dice_HD95 class_num: {class_num}")
    # compute_Dice_HD95 class_num: [0. 1. 2. 3.]
    for c in class_num:
        if c == 0: continue
        ngt_data = np.zeros_like(gt)
        ngt_data[gt == c] = 1
        npred_data = np.zeros_like(pre)
        npred_data[pre == c] = 1
        # print(f"compute_Dice_HD95 ngt_data: {np.unique(ngt_data)} npred_data: {np.unique(npred_data)}")
        n_dice = 2*np.sum(ngt_data*npred_data)/(np.sum(1*ngt_data+npred_data) + 0.0001)
        # n_hd95 = hd95(ngt_data, npred_data, voxelspacing = gt_sitk.GetSpacing()[::-1])
        n_hd95 = hd95(ngt_data, npred_data, voxelspacing = gtspacing[::-1])
        n_dice_list.append(n_dice)
        n_hd95_list.append(n_hd95)
    mean_Dice = statistics.mean(n_dice_list)
    mean_HD95 = statistics.mean(n_hd95_list)
    return mean_Dice, mean_HD95, n_dice_list, n_hd95_list

def compute_Dice_HD95_IOU(pre, gt, gtspacing):
    n_dice_list = []
    n_hd95_list = []
    n_iou_list = []
    class_num = np.unique(gt)
    # print(f"compute_Dice_HD95 class_num: {class_num}")
    # compute_Dice_HD95 class_num: [0. 1. 2. 3.]
    for c in class_num:
        if c == 0: continue
        ngt_data = np.zeros_like(gt)
        ngt_data[gt == c] = 1
        npred_data = np.zeros_like(pre)
        npred_data[pre == c] = 1
        # print(f"compute_Dice_HD95 ngt_data: {np.unique(ngt_data)} npred_data: {np.unique(npred_data)}")
        n_dice = 2*np.sum(ngt_data*npred_data)/(np.sum(1*ngt_data+npred_data) + 0.0001)
        # n_hd95 = hd95(ngt_data, npred_data, voxelspacing = gt_sitk.GetSpacing()[::-1])
        n_hd95 = hd95(ngt_data, npred_data, voxelspacing = gtspacing[::-1])
        n_iou = iou(npred_data, ngt_data)
        n_dice_list.append(n_dice)
        n_hd95_list.append(n_hd95)
        n_iou_list.append(n_iou)
    mean_Dice = statistics.mean(n_dice_list)
    mean_HD95 = statistics.mean(n_hd95_list)
    mean_iou = statistics.mean(n_iou_list)
    return mean_Dice, mean_HD95, mean_iou, n_dice_list, n_hd95_list, n_iou_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    print(f"args: {args}")
    # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/core/logger.py
    opt = Logger.parse(args)
    print(f"opt: {opt}")
    
    # logger = logging.getLogger("Unsupervised Deformable Image Registration with Diffusion Model")
    logger = logging.getLogger("Diffusion Image Registration")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    handler = logging.FileHandler(os.path.join(opt['path']['experiments_root'], 'log.txt'))
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(opt)

    # print(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    # print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # num_gpus = torch.cuda.device_count()
    # for i in range(num_gpus):
    #     device = torch.device(f"cuda:{i}")
    #     properties = torch.cuda.get_device_properties(device)
    #     print(f"GPU {i} 的详细信息：")
    #     print("名称：", properties.name)
    #     print("显存大小：", properties.total_memory)
    #     print("可使用显存大小：", torch.cuda.memory_allocated(i) / (1024 ** 3))
    # torch.cuda.set_device(0)

    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    # print(opt['path']["tb_logger"])
    # writer = tensorboard.SummaryWriter(opt['path']["tb_logger"]+'/..')
    writer = tensorboard.SummaryWriter(opt['path']['experiments_root'])
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    phase = 'train'
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/data/__init__.py  def create_dataset_3D(dataset_opt, phase)
    # train_set = Data.create_dataset_3D_name(dataset_opt, phase)
    train_set = Data.create_dataset_ACDC(dataset_opt, phase)
    # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/data/__init__.py  def create_dataloader(dataset, dataset_opt, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')
    logger.info('Dataset Initialized')

    # model
    # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/__init__.py   def create_model(opt)
    # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py    class DDPM(BaseModel)
    diffusion = Model.create_model(opt)
    print("Model Initialized")
    logger.info("Model Initialized")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    best_epoch, best_avg_Dice, best_avg_HD95, best_avg_iou = 0, 0, 10000, 0
    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py  def feed_data(self, data)
            diffusion.feed_data(train_data)
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py  def optimize_parameters(self)
            diffusion.optimize_parameters()
            t = (time.time() - iter_start_time) / batchSize
            # log
            message = '(epoch: %d | iters: %d / %d | time: %.3f) ' % (current_epoch, (istep+1), training_iters, t)
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py  def get_current_log(self)
            errors = diffusion.get_current_log()
            for k, v in errors.items():
                message += '%s: %.6f ' % (k, v)
            print(message)
            logger.info(message)
            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                writer.add_scalar("train/l_pix", logs['l_pix'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_sim", logs['l_sim'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_smt", logs['l_smt'], (istep+1)*current_epoch)
                writer.add_scalar("train/l_tot", logs['l_tot'], (istep+1)*current_epoch)
            
        # save samples
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            md_list, mhd95_list, miou_list = [], [], []
            test_set = Data.create_dataset_ACDC(dataset_opt=opt['datasets']['test'], phase='test')
            test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
            for istep, test_data in enumerate(test_loader):
                # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py  def feed_data(self, data)
                diffusion.feed_data(test_data)
                # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py
                diffusion.test_registration()
                # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py 
                # def get_current_registration
                visuals = diffusion.get_current_registration()
                # print(f"visuals['contD'].shape {visuals['contD'].shape} visuals['contF'].shape {visuals['contF'].shape}")
                # visuals['contD'].shape torch.Size([1, 1, 32, 128, 128]) visuals['contF'].shape torch.Size([1, 3, 32, 128, 128])
                defm_frames_visual = visuals['contD'].squeeze(0).numpy().transpose(0, 2, 3, 1)
                # defm_frames_visual = visuals['contD'].numpy().transpose(0, 3, 4, 2, 1)[-1]
                flow_frames = visuals['contF'].numpy().transpose(0, 3, 4, 2, 1)
                flow_frames_ES = flow_frames[-1]
                # print(f"defm_frames_visual shape: {defm_frames_visual.shape} flow_frames shape: {flow_frames.shape} flow_frames_ES shape: {flow_frames_ES.shape}")
                # defm_frames_visual shape: (1, 128, 128, 32) flow_frames_ES shape: (128, 128, 32, 3)
                sflow = torch.from_numpy(flow_frames_ES.transpose(3, 2, 0, 1).copy()).unsqueeze(0)
                # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/core/metrics.py def transform_grid(dDepth, dHeight, dWidth)
                sflow = Metrics.transform_grid(sflow[:, 0], sflow[:, 1], sflow[:, 2])
                nb, nc, nd, nh, nw = sflow.shape
                segflow = torch.FloatTensor(sflow.shape).zero_()
                segflow[:, 2] = (sflow[:, 0] / (nd - 1) - 0.5) * 2.0  # D[0 -> 2]
                segflow[:, 1] = (sflow[:, 1] / (nh - 1) - 0.5) * 2.0  # H[1 -> 1]
                segflow[:, 0] = (sflow[:, 2] / (nw - 1) - 0.5) * 2.0  # W[2 -> 0]
                origin_seg = test_data['MS'].squeeze()
                origin_seg = origin_seg.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
                regist_seg = F.grid_sample(origin_seg.cuda().float(), 
                                           (segflow.cuda().float().permute(0, 2, 3, 4, 1)),
                                           mode='nearest')
                regist_seg_ = regist_seg.permute(0, 1, 3, 4, 2)
                regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
                label_seg = test_data['FS'][0].cpu().numpy()
                origin_seg = test_data['MS'][0].cpu().numpy()
                origin_ED = test_data['imageED'][0].cpu().numpy()
                origin_ES = test_data['imageES'][0].cpu().numpy()
                defm_frames_data = defm_frames_visual[0]
                # defm_frames_data = defm_frames_visual[...,-1]
                # defm_frames_data = np.rot90(defm_frames_data, k=-2, axes=(0, 1))
                origin_ED1 = origin_ED.transpose(2, 1, 0)
                origin_seg1 = origin_seg.transpose(2, 1, 0)
                origin_ES1 = origin_ES.transpose(2, 1, 0)
                label_seg1 = label_seg.transpose(2, 1, 0)
                defm_frames_data1 = defm_frames_data.transpose(2, 1, 0)
                regist_seg1 = regist_seg.transpose(2, 1, 0)
                
                regist_seg_int = np.rint(regist_seg1)
                label_seg_int = np.rint(label_seg1)
                # print(f"Segment class - label_seg: {np.unique(label_seg_int)} regist_seg: {np.unique(regist_seg_int)}")
                # Segment class - label_seg: [0. 1. 2. 3.] regist_seg: [0. 1. 2. 3.]

                savedSample_ED = sitk.GetImageFromArray(origin_ED1)
                savedSample_origin = sitk.GetImageFromArray(origin_seg1)
                savedSample_ES = sitk.GetImageFromArray(origin_ES1)
                savedSample_label = sitk.GetImageFromArray(label_seg1)
                savedSample_defm = sitk.GetImageFromArray(defm_frames_data1)
                savedSample_regist = sitk.GetImageFromArray(regist_seg1)
                
                data_in = sitk.ReadImage(test_data['Path'][0])
                ED_origin = data_in.GetOrigin()
                ED_direction = data_in.GetDirection()
                ED_spacing = data_in.GetSpacing()

                savedSample_ED.SetOrigin(ED_origin)
                savedSample_ED.SetDirection(ED_direction)
                savedSample_ED.SetSpacing(ED_spacing)
                
                savedSample_origin.SetOrigin(ED_origin)
                savedSample_origin.SetDirection(ED_direction)
                savedSample_origin.SetSpacing(ED_spacing)
                
                savedSample_ES.SetOrigin(ED_origin)
                savedSample_ES.SetDirection(ED_direction)
                savedSample_ES.SetSpacing(ED_spacing)
                
                savedSample_label.SetOrigin(ED_origin)
                savedSample_label.SetDirection(ED_direction)
                savedSample_label.SetSpacing(ED_spacing)
                
                savedSample_defm.SetOrigin(ED_origin)
                savedSample_defm.SetDirection(ED_direction)
                savedSample_defm.SetSpacing(ED_spacing)
                
                savedSample_regist.SetOrigin(ED_origin)
                savedSample_regist.SetDirection(ED_direction)
                savedSample_regist.SetSpacing(ED_spacing)
                
                sitk.WriteImage(savedSample_ED, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_ED_ep"+str(current_epoch)+".nii.gz"))
                sitk.WriteImage(savedSample_ES, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_ES_ep"+str(current_epoch)+".nii.gz"))
                sitk.WriteImage(savedSample_origin, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_origin_ep"+str(current_epoch)+".nii.gz"))
                sitk.WriteImage(savedSample_label, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_label_ep"+str(current_epoch)+".nii.gz"))
                sitk.WriteImage(savedSample_regist, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_regist_ep"+str(current_epoch)+".nii.gz"))
                sitk.WriteImage(savedSample_defm, os.path.join(opt['path']['samples'], test_data['Name'][0]+"_defm_ep"+str(current_epoch)+".nii.gz"))
                # print("------ Saving training samples ------")
                
                md, mhd95, mIOU, d_list, hd95_list, IOU_list = compute_Dice_HD95_IOU(regist_seg_int, label_seg_int, ED_spacing)
                md_list.append(md)
                mhd95_list.append(mhd95)
                miou_list.append(mIOU)
                
                print(f"------ Saving training samples {istep+1} {test_data['Name'][0]} ------")
                # print(f"{test_data['Name'][0]} mean Dice {md} - {d_list}")
                # print(f"{test_data['Name'][0]} mean HD95 {mhd95} - {hd95_list}")
                logger.info(f"{test_data['Name'][0]} mean Dice {md} - {d_list}")
                logger.info(f"{test_data['Name'][0]} mean HD95 {mhd95} - {hd95_list}")
                logger.info(f"{test_data['Name'][0]} mean IOU {mIOU} - {IOU_list}")
            
            cur_avg_dice = statistics.mean(md_list)
            cur_avg_hd95 = statistics.mean(mhd95_list)
            cur_avg_iou = statistics.mean(miou_list)
            if cur_avg_dice > best_avg_Dice and cur_avg_hd95 < best_avg_HD95 and cur_avg_iou > best_avg_iou:
                best_epoch = current_epoch
                best_avg_Dice = cur_avg_dice
                best_avg_HD95 = cur_avg_hd95
                best_avg_iou = cur_avg_iou
                
            for f in os.listdir(opt['path']['samples']):
                if "ep"+str(best_epoch) in f: continue
                else:
                    os.remove(os.path.join(opt['path']['samples'], f))
                    print(f"remove files without < ep {best_epoch} >")
                
            print(f"Epoch: {current_epoch} Current Dice {cur_avg_dice} HD95 {cur_avg_hd95} IOU {cur_avg_iou} Best_Dice {best_avg_Dice} Best_HD95 {best_avg_HD95} Best_IOU {best_avg_iou} at epoch {best_epoch}")
            logger.info(f"Epoch: {current_epoch} Current Dice {cur_avg_dice} HD95 {cur_avg_hd95} IOU {cur_avg_iou} Best_Dice {best_avg_Dice} Best_HD95 {best_avg_HD95} Best_IOU {best_avg_iou} at epoch {best_epoch}")

        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            # /mnt/no1/lhz/Github/Registration/SEUFSDiffReg/model/model.py  def save_network(self, epoch, iter_step)
            diffusion.save_network(current_epoch, current_step)
            print(f"------ Saving training checkpoint at epoch {current_epoch} ------")
            logger.info(f"------ Saving training checkpoint at epoch {current_epoch} ------")