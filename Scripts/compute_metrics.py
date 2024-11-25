# 计算三维下各种指标
from __future__ import absolute_import, print_function
import os
import logging
import GeodisTK
import numpy as np
import nibabel as nib
import SimpleITK as sitk 
import pandas as pd
from scipy import ndimage
import pystrum.pynd.ndutils as nd

# pixel accuracy
def binary_pa(s, g):
    """
        calculate the pixel accuracy of two N-d volumes.
        s: the segmentation volume of numpy array
        g: the ground truth volume of numpy array
        """
    pa = ((s==g).sum()) / g.size
    return pa


# Dice evaluation
def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape)==len(g.shape))
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return dice

# Dice evaluation
def binary_dice_class(s, g, classnum):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape)==len(g.shape))

    dicelist = []
    for i in range(1, class_num+1):
        s1 = np.where(s == i, 1, 0)
        g1 = np.where(g == i, 1, 0)
        prod = np.multiply(s1, g1)
        s0 = prod.sum()
        dice = (2.0 * s0 + 1e-10) / (s1.sum() + g1.sum() + 1e-10)
        dicelist.append(dice)
    mean_dice = sum(dicelist)/len(dicelist)
    return mean_dice, dicelist


# IOU evaluation
def binary_iou(s, g):
    assert (len(s.shape)==len(g.shape))
    # 两者相乘值为1的部分为交集
    intersecion = np.multiply(s, g)
    # 两者相加，值大于0的部分为交集
    union = np.asarray(s + g > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou


# Hausdorff and ASSD evaluation
def get_edge_points(img):
    """
    get edge points of a binary segmentation result
    """
    dim = len(img.shape)
    if (dim==2):
        strt = ndimage.generate_binary_structure(2, 1)
    else:
        strt = ndimage.generate_binary_structure(3, 1)  # 三维结构元素，与中心点相距1个像素点的都是邻域
    ero = ndimage.morphology.binary_erosion(img, strt)
    edge = np.asarray(img, np.uint8) - np.asarray(ero, np.uint8)
    return edge


def binary_hausdorff95(s, g, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    dist_list1 = s_dis[g_edge > 0]
    dist_list1 = sorted(dist_list1)
    dist1 = dist_list1[int(len(dist_list1) * 0.95)]
    dist_list2 = g_dis[s_edge > 0]
    dist_list2 = sorted(dist_list2)
    dist2 = dist_list2[int(len(dist_list2) * 0.95)]
    return max(dist1, dist2)


def binary_hausdorff95_class(s, g, clsn, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    hd_list = []
    for i in range(1, clsn+1):
        s1 = np.where(s == i, 1, 0)
        g1 = np.where(g == i, 1, 0)
        s_edge = get_edge_points(s1)
        g_edge = get_edge_points(g1)
        image_dim = len(s1.shape)
        assert (image_dim==len(g1.shape))
        if (spacing==None):
            spacing = [1.0] * image_dim
        else:
            assert (image_dim==len(spacing))
        img = np.zeros_like(s1)
        if (image_dim==2):
            s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
            g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
        elif (image_dim==3):
            s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
            g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

        dist_list1 = s_dis[g_edge > 0]
        dist_list1 = sorted(dist_list1)
        dist1 = dist_list1[int(len(dist_list1) * 0.95)]
        dist_list2 = g_dis[s_edge > 0]
        dist_list2 = sorted(dist_list2)
        dist2 = dist_list2[int(len(dist_list2) * 0.95)]
        hd_list.append(max(dist1, dist2))
    mean_hd = sum(hd_list)/len(hd_list)
    return mean_hd, hd_list


def binary_hausdorff95_class1(s, g, clsn, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    avg = binary_hausdorff95(s, g, spacing=None)
    hd_list = []
    for i in range(1, clsn+1):
        s1 = np.where(s == i, 1, 0)
        g1 = np.where(g == i, 1, 0)
        # print(np.count_nonzero(s1), np.count_nonzero(g1))
        y_true_contour = sitk.LabelContour(sitk.GetImageFromArray(g1.astype(np.uint8)), False)
        y_pred_contour = sitk.LabelContour(sitk.GetImageFromArray(s1.astype(np.uint8)), False)
        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_contour, squaredDistance=False, useImageSpacing=True)) # i.e. euclidean distance
        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_contour, squaredDistance=False, useImageSpacing=True))
        dist_y_pred = sitk.GetArrayViewFromImage(y_pred_distance_map)[sitk.GetArrayViewFromImage(y_true_distance_map)==0]  # pointless?
        dist_y_true = sitk.GetArrayViewFromImage(y_true_distance_map)[sitk.GetArrayViewFromImage(y_pred_distance_map)==0]
        # print (' - 95 hausdorff:', np.percentile(dist_y_true,95), np.percentile(dist_y_pred,95))
        try:
            max = np.percentile(dist_y_pred, 95) if np.percentile(dist_y_pred,95) > np.percentile(dist_y_true,95) else np.percentile(dist_y_true,95)
        except:
            # max = np.percentile(dist_y_true, 95)
            max = avg
        hd_list.append(max)
    mean_hd = sum(hd_list)/len(hd_list)
    return mean_hd, hd_list


# 平均表面距离
def binary_assd(s, g, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s_edge = get_edge_points(s)
    g_edge = get_edge_points(g)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img, s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img, g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# relative volume error evaluation
def binary_relative_volume_error(s_volume, g_volume):
    s_v = float(s_volume.sum())
    g_v = float(g_volume.sum())
    assert (g_v > 0)
    rve = abs(s_v - g_v) / g_v
    return rve


def compute_class_sens_spec(pred, label):
    """
    Compute sensitivity and specificity for a particular example
    for a given class for binary.
    Args:
        pred (np.array): binary arrary of predictions, shape is
                         (height, width, depth).
        label (np.array): binary array of labels, shape is
                          (height, width, depth).
    Returns:
        sensitivity (float): precision for given class_num.
        specificity (float): recall for given class_num
    """
    tp = np.sum((pred==1) & (label==1))
    tn = np.sum((pred==0) & (label==0))
    fp = np.sum((pred==1) & (label==0))
    fn = np.sum((pred==0) & (label==1))

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def get_evaluation_score(s_volume, g_volume, spacing, metric):
    if (len(s_volume.shape)==4):
        assert (s_volume.shape[0]==1 and g_volume.shape[0]==1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0]==1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower=="dice"):
        score = binary_dice(s_volume, g_volume)

    elif (metric_lower=="iou"):
        score = binary_iou(s_volume, g_volume)

    elif (metric_lower=='assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif (metric_lower=="hausdorff95"):
        score = binary_hausdorff95(s_volume, g_volume, spacing)

    elif (metric_lower=="rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower=="volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


def get_evaluation_class_score(s_volume, g_volume, classnum, spacing, metric):
    if (len(s_volume.shape)==4):
        assert (s_volume.shape[0]==1 and g_volume.shape[0]==1)
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    if (s_volume.shape[0]==1):
        s_volume = np.reshape(s_volume, s_volume.shape[1:])
        g_volume = np.reshape(g_volume, g_volume.shape[1:])
    metric_lower = metric.lower()

    if (metric_lower=="dice"):
        score, dicelist = binary_dice_class(s_volume, g_volume, classnum)
        return score, dicelist

    elif (metric_lower=="iou"):
        score = binary_iou(s_volume, g_volume)

    elif (metric_lower=='assd'):
        score = binary_assd(s_volume, g_volume, spacing)

    elif (metric_lower=="hausdorff95"):
        score, hdlist = binary_hausdorff95_class1(s_volume, g_volume, classnum, spacing)
        return score, hdlist

    elif (metric_lower=="rve"):
        score = binary_relative_volume_error(s_volume, g_volume)

    elif (metric_lower=="volume"):
        voxel_size = 1.0
        for dim in range(len(spacing)):
            voxel_size = voxel_size * spacing[dim]
        score = g_volume.sum() * voxel_size
    else:
        raise ValueError("unsupported evaluation metric: {0:}".format(metric))

    return score


def read_txt(file):
    with open(file, 'r') as f:
        content = f.readlines()
        # print(f"content: {content}")
    # filelist = [x.strip() for x in content if x.strip()]
    filelist = [x.strip() for x in content]
    # print(f"filelist: {filelist}")

    pairs = []
    for f in filelist:
        pairs.append([f.split(' ')[1], f.split(' ')[3]])

    return pairs

def read_nii(filename):        
    nii = sitk.ReadImage(filename)
    nii_array = sitk.GetArrayFromImage(nii)
    nii_size = nii.GetSize()
    nii_origin = nii.GetOrigin()
    nii_direction = nii.GetDirection()
    nii_spacing = nii.GetSpacing()
    # return nii_array, nii_size, nii_origin, nii_direction, nii_spacing
    return nii_array.astype(int), nii_size, nii_origin, nii_direction, nii_spacing


# Dice evaluation
def calculate_dice(s, g, i):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert (len(s.shape)==len(g.shape))

    s1 = np.where(s == i, 1, 0)
    g1 = np.where(g == i, 1, 0)
    prod = np.multiply(s1, g1)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s1.sum() + g1.sum() + 1e-10)
    return dice

# HD95 evaluation
def calculate_hausdorff95(s, g, clsn, spacing=None):
    """
    get the hausdorff distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    # avg = binary_hausdorff95(s, g, spacing=None)
    # TypeError: Cannot cast array data from dtype('int64') to dtype('float32') according to the rule 'safe'
    avg = binary_hausdorff95(s.astype('float32'), g.astype('float32'), spacing=None)
    s1 = np.where(s == clsn, 1, 0)
    g1 = np.where(g == clsn, 1, 0)
    # print(np.count_nonzero(s1), np.count_nonzero(g1))
    y_true_contour = sitk.LabelContour(sitk.GetImageFromArray(g1.astype(np.uint8)), False)
    y_pred_contour = sitk.LabelContour(sitk.GetImageFromArray(s1.astype(np.uint8)), False)
    y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_contour, squaredDistance=False, useImageSpacing=True)) # i.e. euclidean distance
    y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_contour, squaredDistance=False, useImageSpacing=True))
    dist_y_pred = sitk.GetArrayViewFromImage(y_pred_distance_map)[sitk.GetArrayViewFromImage(y_true_distance_map)==0]  # pointless?
    dist_y_true = sitk.GetArrayViewFromImage(y_true_distance_map)[sitk.GetArrayViewFromImage(y_pred_distance_map)==0]
    # print (' - 95 hausdorff:', np.percentile(dist_y_true,95), np.percentile(dist_y_pred,95))
    try:
        max = np.percentile(dist_y_pred, 95) if np.percentile(dist_y_pred, 95) > np.percentile(dist_y_true,95) else np.percentile(dist_y_true, 95)
    except:
        # max = np.percentile(dist_y_true, 95)
        max = avg
    # hd_list.append(max)
    # mean_hd = sum(hd_list)/len(hd_list)
    # return mean_hd, hd_list
    return max

# ASSD evaluation
def calculate_assd(s, g, i, spacing=None):
    """
    get the average symetric surface distance between a binary segmentation and the ground truth
    inputs:
        s: a 3D or 2D binary image for segmentation
        g: a 2D or 2D binary image for ground truth
        spacing: a list for image spacing, length should be 3 or 2
    """
    s1 = np.where(s == i, 1, 0)
    g1 = np.where(g == i, 1, 0)
    s_edge = get_edge_points(s1)
    g_edge = get_edge_points(g1)
    image_dim = len(s.shape)
    assert (image_dim==len(g.shape))
    if (spacing==None):
        spacing = [1.0] * image_dim
    else:
        assert (image_dim==len(spacing))
    img = np.zeros_like(s)
    if (image_dim==2):
        s_dis = GeodisTK.geodesic2d_raster_scan(img, s_edge, 0.0, 2)
        g_dis = GeodisTK.geodesic2d_raster_scan(img, g_edge, 0.0, 2)
    elif (image_dim==3):
        s_dis = GeodisTK.geodesic3d_raster_scan(img.astype('float32'), s_edge, spacing, 0.0, 2)
        g_dis = GeodisTK.geodesic3d_raster_scan(img.astype('float32'), g_edge, spacing, 0.0, 2)

    ns = s_edge.sum()
    ng = g_edge.sum()
    s_dis_g_edge = s_dis * g_edge
    g_dis_s_edge = g_dis * s_edge
    assd = (s_dis_g_edge.sum() + g_dis_s_edge.sum()) / (ns + ng)
    return assd


# IOU evaluation
def calculate_iou(s, g, i):
    assert (len(s.shape)==len(g.shape))
    s1 = np.where(s == i, 1, 0)
    g1 = np.where(g == i, 1, 0)
    intersecion = np.multiply(s1, g1)
    union = np.asarray(s1 + g1 > 0, np.float32)
    iou = intersecion.sum() / (union.sum() + 1e-10)
    return iou

# calculate non-positive Jacobian determinant
def jacobian_det(flowfile):
    nii = sitk.ReadImage(flowfile)
    flow = sitk.GetArrayFromImage(nii).transpose(3, 0, 1, 2)
    # flow = sitk.GetArrayFromImage(nii).transpose(3, 2, 1, 0)
    # print(f"jacobian_det flow shape: {flow.shape}")
    # jacobian_det flow shape: (3, 15, 288, 232)  CDHW
    # print(f"jacobian_det flow {flow.shape} max: {flow.max()} min: {flow.min()}")
    # jacobian_det flow (3, 10, 256, 216) max: 6.2563323974609375 min: -5.582195281982422

    bias_d = np.array([0, 0, 1])
    bias_h = np.array([0, 1, 0])
    bias_w = np.array([1, 0, 0])

    volume_d = np.transpose(flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_d
    volume_h = np.transpose(flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_h
    volume_w = np.transpose(flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1], (1, 2, 3, 0)) + bias_w

    jacobian_det_volume = np.linalg.det(np.stack([volume_w, volume_h, volume_d], -1))
    jd = np.sum(jacobian_det_volume <= 0)
    # jd_percent = jd / jacobian_det_volume.size
    jd_percent = jd / flow.size
    return jd_percent

# /mnt/lhz/Github/Image_registration/RDP/utils.py
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """
    nii = sitk.ReadImage(disp)
    flow = sitk.GetArrayFromImage(nii)
    # print(f"jacobian_determinant_vxm flow: {flow.shape}")
    # jacobian_determinant_vxm flow: (10, 256, 232, 3)  (D, H, W, C)
    # check inputs
    # disp = flow.transpose(1, 2, 3, 0)
    # disp = flow.transpose(0, 1, 2, 3)
    disp = flow
    volshape = disp.shape[:-1]
    # volshape = disp.shape[1:]
    # print(f"jacobian_determinant_vxm volshape: {volshape}")
    # jacobian_determinant_vxm volshape: (10, 256, 232)

    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'
    
    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))
    # grid = np.stack(grid_lst, len(volshape)).transpose(3, 0, 1, 2)

    # compute gradients
    # print(f"jacobian_determinant_vxm grid: {grid.shape}")
    J = np.gradient(disp + grid)
    # print(f"jacobian_determinant_vxm J: {len(J)}")
    # jacobian_determinant_vxm J: 4
    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]
        # print(f"dx {dx.shape} dy {dy.shape} dz {dz.shape}")
        # dx (10, 256, 232, 3) dy (10, 256, 232, 3) dz (10, 256, 232, 3)
        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        # print(f"Jdet0 {Jdet0.shape} Jdet1 {Jdet1.shape} Jdet2 {Jdet2.shape}")
        # Jdet0 (10, 256, 232) Jdet1 (10, 256, 232) Jdet2 (10, 256, 232)
        # return Jdet0 - Jdet1 + Jdet2
        NJD = Jdet0 - Jdet1 + Jdet2
    else:  # must be 2
        dfdx = J[0]
        dfdy = J[1]
        # return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
        NJD = dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
    
    return np.sum(NJD <= 0) / np.prod(volshape)

# grid = np.stack(grid_lst, len(volshape)).transpose(3, 0, 1, 2)
# NJD mean: 0.007091575087270413 std: 0.0010020894551796388
# grid = np.stack(grid_lst, len(volshape))
# NJD mean: 0.01087975939970453 std: 0.00588555683414767


def compute_metrics(pred_path, pred_gt_files, save_dir, logger):
    meanDice_list, eachDice_dict = [], {}
    meanHD95_list, eachHD95_dict = [], {}
    meanASSD_list, eachASSD_dict = [], {}
    meanIOU_list, eachIOU_dict = [], {}
    jacobian_det_list = []
    # for pairs in pred_gt_files:
    for i, pairs in enumerate(pred_gt_files):
        pair1, pair2 = pairs[0], pairs[1]
        # change name of pred_file to the output results
        pred_file = os.path.join(pred_path, pair1.split('/')[-1].split('_gt')[0] + '_ep25001_warped_seg.nii.gz')
        deform_file = os.path.join(pred_path, pair1.split('/')[-1].split('_gt')[0] + '_ep25001_warped_deform.nii.gz')
        gt_file = pair2
        print(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        logger.info(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        
        if not os.path.exists(pred_file) or not os.path.exists(gt_file) or not os.path.exists(deform_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
        # jacobian_det_value = jacobian_det(deform_file)
        jacobian_det_value = jacobian_determinant_vxm(deform_file)
        jacobian_det_list.append(jacobian_det_value)
        # print(f"{pred_file}: {pred_array.shape} {pred_size} {pred_origin} {pred_direction} {pred_spacing}")
        # /mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval/patient145_frame13_ep1_warped_seg.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"{gt_file}: {gt_array.shape} {gt_size} {gt_origin} {gt_direction} {gt_spacing}")
        # /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient145/patient145_frame01_gt.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"pred label: {np.unique(pred_array)} GT label: {np.unique(gt_array)}")
        # pred label: [0. 1. 2. 3.] GT label: [0 1 2 3]
        # pred label: [0 1 2 3] GT label: [0 1 2 3]

        pairs_Dice_class, pairs_HD95_class, pairs_ASSD_class, pairs_IOU_class = [], [], [], []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            iHD95_value = calculate_hausdorff95(pred_array, gt_array, i)
            iASSD_value = calculate_assd(pred_array, gt_array, i)
            iou_value = calculate_iou(pred_array, gt_array, i)
            
            # Dice Dict
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            
            # HD95 Dict
            if i not in eachHD95_dict.keys():
                eachHD95_dict[i] = []
                eachHD95_dict[i].append(iHD95_value)
            else:
                eachHD95_dict[i].append(iHD95_value)

            # ASSD Dict
            if i not in eachASSD_dict.keys():
                eachASSD_dict[i] = []
                eachASSD_dict[i].append(iASSD_value)
            else:
                eachASSD_dict[i].append(iASSD_value)

            # IOU Dict
            if i not in eachIOU_dict.keys():
                eachIOU_dict[i] = []
                eachIOU_dict[i].append(iou_value)
            else:
                eachIOU_dict[i].append(iou_value)
                     
            pairs_Dice_class.append(idice_value)
            pairs_HD95_class.append(iHD95_value)
            pairs_ASSD_class.append(iASSD_value)
            pairs_IOU_class.append(iou_value)
            
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

        eachpair_meanHD95 = sum(pairs_HD95_class)/len(pairs_HD95_class)
        meanHD95_list.append(eachpair_meanHD95)

        eachpair_meanASSD = sum(pairs_ASSD_class)/len(pairs_ASSD_class)
        meanASSD_list.append(eachpair_meanASSD)

        eachpair_meanIOU = sum(pairs_IOU_class)/len(pairs_IOU_class)
        meanIOU_list.append(eachpair_meanIOU)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    logger.info(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")
        logger.info(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")

    HD95_mean, HD95_std = np.mean(np.array(meanHD95_list)), np.std(np.array(meanHD95_list))
    print(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    logger.info(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    for c in eachHD95_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classHD95_mean, classHD95_std = np.mean(np.array(eachHD95_dict[c])), np.std(np.array(eachHD95_dict[c]))
        print(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")
        logger.info(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")

    ASSD_mean, ASSD_std = np.mean(np.array(meanASSD_list)), np.std(np.array(meanASSD_list))
    print(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    logger.info(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    for c in eachASSD_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classASSD_mean, classASSD_std = np.mean(np.array(eachASSD_dict[c])), np.std(np.array(eachASSD_dict[c]))
        print(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")
        logger.info(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")

    IOU_mean, IOU_std = np.mean(np.array(meanIOU_list)), np.std(np.array(meanIOU_list))
    print(f"IOU mean: {IOU_mean} std: {IOU_std}")
    logger.info(f"IOU mean: {IOU_mean} std: {IOU_std}")
    for c in eachIOU_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classIOU_mean, classIOU_std = np.mean(np.array(eachIOU_dict[c])), np.std(np.array(eachIOU_dict[c]))
        print(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")
        logger.info(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")

    NJD_mean, NJD_std = np.mean(np.array(jacobian_det_list)), np.std(np.array(jacobian_det_list))
    print(f"NJD mean: {NJD_mean} std: {NJD_std}")
    logger.info(f"NJD mean: {NJD_mean} std: {NJD_std}")


def compute_metrics_ACDC(pred_path, pred_gt_files, save_dir, logger):
    meanDice_list, eachDice_dict = [], {}
    meanHD95_list, eachHD95_dict = [], {}
    meanASSD_list, eachASSD_dict = [], {}
    meanIOU_list, eachIOU_dict = [], {}
    jacobian_det_list = []
    # for pairs in pred_gt_files:
    for i, pairs in enumerate(pred_gt_files):
        pair1, pair2 = pairs[0], pairs[1]
        # change name of pred_file to the output results
        pred_file = os.path.join(pred_path, pair1.split('/')[-1].split('_')[0] + '_ep990_warped_seg.nii.gz')
        deform_file = os.path.join(pred_path, pair1.split('/')[-1].split('_')[0] + '_ep990_warped_deformflow.nii.gz')
        gt_file = pair2
        print(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        logger.info(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        
        if not os.path.exists(pred_file) or not os.path.exists(gt_file) or not os.path.exists(deform_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
        # jacobian_det_value = jacobian_det(deform_file)
        jacobian_det_value = jacobian_determinant_vxm(deform_file)
        jacobian_det_list.append(jacobian_det_value)
        # print(f"{pred_file}: {pred_array.shape} {pred_size} {pred_origin} {pred_direction} {pred_spacing}")
        # /mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval/patient145_frame13_ep1_warped_seg.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"{gt_file}: {gt_array.shape} {gt_size} {gt_origin} {gt_direction} {gt_spacing}")
        # /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient145/patient145_frame01_gt.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"pred label: {np.unique(pred_array)} GT label: {np.unique(gt_array)}")
        # pred label: [0. 1. 2. 3.] GT label: [0 1 2 3]
        # pred label: [0 1 2 3] GT label: [0 1 2 3]

        pairs_Dice_class, pairs_HD95_class, pairs_ASSD_class, pairs_IOU_class = [], [], [], []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            iHD95_value = calculate_hausdorff95(pred_array, gt_array, i)
            iASSD_value = calculate_assd(pred_array, gt_array, i)
            iou_value = calculate_iou(pred_array, gt_array, i)
            
            # Dice Dict
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            
            # HD95 Dict
            if i not in eachHD95_dict.keys():
                eachHD95_dict[i] = []
                eachHD95_dict[i].append(iHD95_value)
            else:
                eachHD95_dict[i].append(iHD95_value)

            # ASSD Dict
            if i not in eachASSD_dict.keys():
                eachASSD_dict[i] = []
                eachASSD_dict[i].append(iASSD_value)
            else:
                eachASSD_dict[i].append(iASSD_value)

            # IOU Dict
            if i not in eachIOU_dict.keys():
                eachIOU_dict[i] = []
                eachIOU_dict[i].append(iou_value)
            else:
                eachIOU_dict[i].append(iou_value)
                     
            pairs_Dice_class.append(idice_value)
            pairs_HD95_class.append(iHD95_value)
            pairs_ASSD_class.append(iASSD_value)
            pairs_IOU_class.append(iou_value)
            
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

        eachpair_meanHD95 = sum(pairs_HD95_class)/len(pairs_HD95_class)
        meanHD95_list.append(eachpair_meanHD95)

        eachpair_meanASSD = sum(pairs_ASSD_class)/len(pairs_ASSD_class)
        meanASSD_list.append(eachpair_meanASSD)

        eachpair_meanIOU = sum(pairs_IOU_class)/len(pairs_IOU_class)
        meanIOU_list.append(eachpair_meanIOU)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    logger.info(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")
        logger.info(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")

    HD95_mean, HD95_std = np.mean(np.array(meanHD95_list)), np.std(np.array(meanHD95_list))
    print(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    logger.info(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    for c in eachHD95_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classHD95_mean, classHD95_std = np.mean(np.array(eachHD95_dict[c])), np.std(np.array(eachHD95_dict[c]))
        print(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")
        logger.info(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")

    ASSD_mean, ASSD_std = np.mean(np.array(meanASSD_list)), np.std(np.array(meanASSD_list))
    print(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    logger.info(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    for c in eachASSD_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classASSD_mean, classASSD_std = np.mean(np.array(eachASSD_dict[c])), np.std(np.array(eachASSD_dict[c]))
        print(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")
        logger.info(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")

    IOU_mean, IOU_std = np.mean(np.array(meanIOU_list)), np.std(np.array(meanIOU_list))
    print(f"IOU mean: {IOU_mean} std: {IOU_std}")
    logger.info(f"IOU mean: {IOU_mean} std: {IOU_std}")
    for c in eachIOU_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classIOU_mean, classIOU_std = np.mean(np.array(eachIOU_dict[c])), np.std(np.array(eachIOU_dict[c]))
        print(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")
        logger.info(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")

    NJD_mean, NJD_std = np.mean(np.array(jacobian_det_list)), np.std(np.array(jacobian_det_list))
    print(f"NJD mean: {NJD_mean} std: {NJD_std}")
    logger.info(f"NJD mean: {NJD_mean} std: {NJD_std}")


def compute_metrics_LPBA(pred_path, pred_gt_files, save_dir, logger):
    meanDice_list, eachDice_dict = [], {}
    meanHD95_list, eachHD95_dict = [], {}
    meanASSD_list, eachASSD_dict = [], {}
    meanIOU_list, eachIOU_dict = [], {}
    jacobian_det_list = []
    # for pairs in pred_gt_files:
    for i, pairs in enumerate(pred_gt_files):
        pair1, pair2 = pairs[0], pairs[1]
        # change name of pred_file to the output results
        pred_file = os.path.join(pred_path, pair1.split('/')[-1].split('.')[0] + '_ep25001_warped_seg.nii.gz')
        deform_file = os.path.join(pred_path, pair1.split('/')[-1].split('.')[0] + '_ep25001_warped_deform.nii.gz')
        gt_file = pair2
        print(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        logger.info(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        
        if not os.path.exists(pred_file) or not os.path.exists(gt_file) or not os.path.exists(deform_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
        # jacobian_det_value = jacobian_det(deform_file)
        jacobian_det_value = jacobian_determinant_vxm(deform_file)
        jacobian_det_list.append(jacobian_det_value)
        # print(f"{pred_file}: {pred_array.shape} {pred_size} {pred_origin} {pred_direction} {pred_spacing}")
        # /mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval/patient145_frame13_ep1_warped_seg.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"{gt_file}: {gt_array.shape} {gt_size} {gt_origin} {gt_direction} {gt_spacing}")
        # /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient145/patient145_frame01_gt.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"pred label: {np.unique(pred_array)} GT label: {np.unique(gt_array)}")
        # pred label: [0. 1. 2. 3.] GT label: [0 1 2 3]
        # pred label: [0 1 2 3] GT label: [0 1 2 3]

        pairs_Dice_class, pairs_HD95_class, pairs_ASSD_class, pairs_IOU_class = [], [], [], []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            iHD95_value = calculate_hausdorff95(pred_array, gt_array, i)
            iASSD_value = calculate_assd(pred_array, gt_array, i)
            iou_value = calculate_iou(pred_array, gt_array, i)
            
            # Dice Dict
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            
            # HD95 Dict
            if i not in eachHD95_dict.keys():
                eachHD95_dict[i] = []
                eachHD95_dict[i].append(iHD95_value)
            else:
                eachHD95_dict[i].append(iHD95_value)

            # ASSD Dict
            if i not in eachASSD_dict.keys():
                eachASSD_dict[i] = []
                eachASSD_dict[i].append(iASSD_value)
            else:
                eachASSD_dict[i].append(iASSD_value)

            # IOU Dict
            if i not in eachIOU_dict.keys():
                eachIOU_dict[i] = []
                eachIOU_dict[i].append(iou_value)
            else:
                eachIOU_dict[i].append(iou_value)
                     
            pairs_Dice_class.append(idice_value)
            pairs_HD95_class.append(iHD95_value)
            pairs_ASSD_class.append(iASSD_value)
            pairs_IOU_class.append(iou_value)
            
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

        eachpair_meanHD95 = sum(pairs_HD95_class)/len(pairs_HD95_class)
        meanHD95_list.append(eachpair_meanHD95)

        eachpair_meanASSD = sum(pairs_ASSD_class)/len(pairs_ASSD_class)
        meanASSD_list.append(eachpair_meanASSD)

        eachpair_meanIOU = sum(pairs_IOU_class)/len(pairs_IOU_class)
        meanIOU_list.append(eachpair_meanIOU)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    logger.info(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")
        logger.info(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")

    HD95_mean, HD95_std = np.mean(np.array(meanHD95_list)), np.std(np.array(meanHD95_list))
    print(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    logger.info(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    for c in eachHD95_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classHD95_mean, classHD95_std = np.mean(np.array(eachHD95_dict[c])), np.std(np.array(eachHD95_dict[c]))
        print(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")
        logger.info(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")

    ASSD_mean, ASSD_std = np.mean(np.array(meanASSD_list)), np.std(np.array(meanASSD_list))
    print(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    logger.info(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    for c in eachASSD_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classASSD_mean, classASSD_std = np.mean(np.array(eachASSD_dict[c])), np.std(np.array(eachASSD_dict[c]))
        print(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")
        logger.info(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")

    IOU_mean, IOU_std = np.mean(np.array(meanIOU_list)), np.std(np.array(meanIOU_list))
    print(f"IOU mean: {IOU_mean} std: {IOU_std}")
    logger.info(f"IOU mean: {IOU_mean} std: {IOU_std}")
    for c in eachIOU_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classIOU_mean, classIOU_std = np.mean(np.array(eachIOU_dict[c])), np.std(np.array(eachIOU_dict[c]))
        print(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")
        logger.info(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")

    NJD_mean, NJD_std = np.mean(np.array(jacobian_det_list)), np.std(np.array(jacobian_det_list))
    print(f"NJD mean: {NJD_mean} std: {NJD_std}")
    logger.info(f"NJD mean: {NJD_mean} std: {NJD_std}")


def compute_metrics_OASIS(pred_path, pred_gt_files, save_dir, logger):
    meanDice_list, eachDice_dict = [], {}
    meanHD95_list, eachHD95_dict = [], {}
    meanASSD_list, eachASSD_dict = [], {}
    meanIOU_list, eachIOU_dict = [], {}
    jacobian_det_list = []
    # for pairs in pred_gt_files:
    for i, pairs in enumerate(pred_gt_files):
        pair1, pair2 = pairs[0], pairs[1]
        # change name of pred_file to the output results
        pred_file = os.path.join(pred_path, pair1.split('/')[-1][:10] + '_ep25001_warped_seg.nii.gz')
        deform_file = os.path.join(pred_path, pair1.split('/')[-1][:10] + '_ep25001_warped_deform.nii.gz')
        gt_file = pair2
        print(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        logger.info(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        
        if not os.path.exists(pred_file) or not os.path.exists(gt_file) or not os.path.exists(deform_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
        # jacobian_det_value = jacobian_det(deform_file)
        jacobian_det_value = jacobian_determinant_vxm(deform_file)
        jacobian_det_list.append(jacobian_det_value)
        # print(f"{pred_file}: {pred_array.shape} {pred_size} {pred_origin} {pred_direction} {pred_spacing}")
        # /mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval/patient145_frame13_ep1_warped_seg.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"{gt_file}: {gt_array.shape} {gt_size} {gt_origin} {gt_direction} {gt_spacing}")
        # /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient145/patient145_frame01_gt.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"pred label: {np.unique(pred_array)} GT label: {np.unique(gt_array)}")
        # pred label: [0. 1. 2. 3.] GT label: [0 1 2 3]
        # pred label: [0 1 2 3] GT label: [0 1 2 3]

        pairs_Dice_class, pairs_HD95_class, pairs_ASSD_class, pairs_IOU_class = [], [], [], []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            iHD95_value = calculate_hausdorff95(pred_array, gt_array, i)
            iASSD_value = calculate_assd(pred_array, gt_array, i)
            iou_value = calculate_iou(pred_array, gt_array, i)
            
            # Dice Dict
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            
            # HD95 Dict
            if i not in eachHD95_dict.keys():
                eachHD95_dict[i] = []
                eachHD95_dict[i].append(iHD95_value)
            else:
                eachHD95_dict[i].append(iHD95_value)

            # ASSD Dict
            if i not in eachASSD_dict.keys():
                eachASSD_dict[i] = []
                eachASSD_dict[i].append(iASSD_value)
            else:
                eachASSD_dict[i].append(iASSD_value)

            # IOU Dict
            if i not in eachIOU_dict.keys():
                eachIOU_dict[i] = []
                eachIOU_dict[i].append(iou_value)
            else:
                eachIOU_dict[i].append(iou_value)
                     
            pairs_Dice_class.append(idice_value)
            pairs_HD95_class.append(iHD95_value)
            pairs_ASSD_class.append(iASSD_value)
            pairs_IOU_class.append(iou_value)
            
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

        eachpair_meanHD95 = sum(pairs_HD95_class)/len(pairs_HD95_class)
        meanHD95_list.append(eachpair_meanHD95)

        eachpair_meanASSD = sum(pairs_ASSD_class)/len(pairs_ASSD_class)
        meanASSD_list.append(eachpair_meanASSD)

        eachpair_meanIOU = sum(pairs_IOU_class)/len(pairs_IOU_class)
        meanIOU_list.append(eachpair_meanIOU)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    logger.info(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")
        logger.info(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")

    HD95_mean, HD95_std = np.mean(np.array(meanHD95_list)), np.std(np.array(meanHD95_list))
    print(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    logger.info(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    for c in eachHD95_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classHD95_mean, classHD95_std = np.mean(np.array(eachHD95_dict[c])), np.std(np.array(eachHD95_dict[c]))
        print(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")
        logger.info(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")

    ASSD_mean, ASSD_std = np.mean(np.array(meanASSD_list)), np.std(np.array(meanASSD_list))
    print(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    logger.info(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    for c in eachASSD_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classASSD_mean, classASSD_std = np.mean(np.array(eachASSD_dict[c])), np.std(np.array(eachASSD_dict[c]))
        print(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")
        logger.info(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")

    IOU_mean, IOU_std = np.mean(np.array(meanIOU_list)), np.std(np.array(meanIOU_list))
    print(f"IOU mean: {IOU_mean} std: {IOU_std}")
    logger.info(f"IOU mean: {IOU_mean} std: {IOU_std}")
    for c in eachIOU_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classIOU_mean, classIOU_std = np.mean(np.array(eachIOU_dict[c])), np.std(np.array(eachIOU_dict[c]))
        print(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")
        logger.info(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")

    NJD_mean, NJD_std = np.mean(np.array(jacobian_det_list)), np.std(np.array(jacobian_det_list))
    print(f"NJD mean: {NJD_mean} std: {NJD_std}")
    logger.info(f"NJD mean: {NJD_mean} std: {NJD_std}")


def compute_metrics_OAIZIB(pred_path, pred_gt_files, save_dir, logger):
    meanDice_list, eachDice_dict = [], {}
    meanHD95_list, eachHD95_dict = [], {}
    meanASSD_list, eachASSD_dict = [], {}
    meanIOU_list, eachIOU_dict = [], {}
    jacobian_det_list = []
    # for pairs in pred_gt_files:
    for i, pairs in enumerate(pred_gt_files):
        pair1, pair2 = pairs[0], pairs[1]
        # change name of pred_file to the output results
        pred_file = os.path.join(pred_path, pair1.split('/')[-1].split('.')[0] + '_ep20001_warped_seg.nii.gz')
        deform_file = os.path.join(pred_path, pair1.split('/')[-1].split('.')[0] + '_ep20001_warped_deform.nii.gz')
        gt_file = pair2
        print(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        logger.info(f"{i + 1} pred: {pred_file} gt: {gt_file} deform: {deform_file}")
        
        if not os.path.exists(pred_file) or not os.path.exists(gt_file) or not os.path.exists(deform_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
        # jacobian_det_value = jacobian_det(deform_file)
        jacobian_det_value = jacobian_determinant_vxm(deform_file)
        jacobian_det_list.append(jacobian_det_value)
        # print(f"{pred_file}: {pred_array.shape} {pred_size} {pred_origin} {pred_direction} {pred_spacing}")
        # /mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval/patient145_frame13_ep1_warped_seg.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"{gt_file}: {gt_array.shape} {gt_size} {gt_origin} {gt_direction} {gt_spacing}")
        # /mnt/lhz/Datasets/Learn2reg/ACDC/testing/patient145/patient145_frame01_gt.nii.gz: 
        # (10, 256, 232) (232, 256, 10) (0.0, 0.0, 0.0) (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) 
        # (1.7578099966049194, 1.7578099966049194, 10.0)
        # print(f"pred label: {np.unique(pred_array)} GT label: {np.unique(gt_array)}")
        # pred label: [0. 1. 2. 3.] GT label: [0 1 2 3]
        # pred label: [0 1 2 3] GT label: [0 1 2 3]

        pairs_Dice_class, pairs_HD95_class, pairs_ASSD_class, pairs_IOU_class = [], [], [], []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            iHD95_value = calculate_hausdorff95(pred_array, gt_array, i)
            iASSD_value = calculate_assd(pred_array, gt_array, i)
            iou_value = calculate_iou(pred_array, gt_array, i)
            
            # Dice Dict
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            
            # HD95 Dict
            if i not in eachHD95_dict.keys():
                eachHD95_dict[i] = []
                eachHD95_dict[i].append(iHD95_value)
            else:
                eachHD95_dict[i].append(iHD95_value)

            # ASSD Dict
            if i not in eachASSD_dict.keys():
                eachASSD_dict[i] = []
                eachASSD_dict[i].append(iASSD_value)
            else:
                eachASSD_dict[i].append(iASSD_value)

            # IOU Dict
            if i not in eachIOU_dict.keys():
                eachIOU_dict[i] = []
                eachIOU_dict[i].append(iou_value)
            else:
                eachIOU_dict[i].append(iou_value)
                     
            pairs_Dice_class.append(idice_value)
            pairs_HD95_class.append(iHD95_value)
            pairs_ASSD_class.append(iASSD_value)
            pairs_IOU_class.append(iou_value)
            
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

        eachpair_meanHD95 = sum(pairs_HD95_class)/len(pairs_HD95_class)
        meanHD95_list.append(eachpair_meanHD95)

        eachpair_meanASSD = sum(pairs_ASSD_class)/len(pairs_ASSD_class)
        meanASSD_list.append(eachpair_meanASSD)

        eachpair_meanIOU = sum(pairs_IOU_class)/len(pairs_IOU_class)
        meanIOU_list.append(eachpair_meanIOU)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    logger.info(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")
        logger.info(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")

    HD95_mean, HD95_std = np.mean(np.array(meanHD95_list)), np.std(np.array(meanHD95_list))
    print(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    logger.info(f"HD95 mean: {HD95_mean} std: {HD95_std}")
    for c in eachHD95_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classHD95_mean, classHD95_std = np.mean(np.array(eachHD95_dict[c])), np.std(np.array(eachHD95_dict[c]))
        print(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")
        logger.info(f"class {c}: HD95 mean: {classHD95_mean} std: {classHD95_std}")

    ASSD_mean, ASSD_std = np.mean(np.array(meanASSD_list)), np.std(np.array(meanASSD_list))
    print(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    logger.info(f"ASSD mean: {ASSD_mean} std: {ASSD_std}")
    for c in eachASSD_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classASSD_mean, classASSD_std = np.mean(np.array(eachASSD_dict[c])), np.std(np.array(eachASSD_dict[c]))
        print(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")
        logger.info(f"class {c}: ASSD mean: {classASSD_mean} std: {classASSD_std}")

    IOU_mean, IOU_std = np.mean(np.array(meanIOU_list)), np.std(np.array(meanIOU_list))
    print(f"IOU mean: {IOU_mean} std: {IOU_std}")
    logger.info(f"IOU mean: {IOU_mean} std: {IOU_std}")
    for c in eachIOU_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classIOU_mean, classIOU_std = np.mean(np.array(eachIOU_dict[c])), np.std(np.array(eachIOU_dict[c]))
        print(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")
        logger.info(f"class {c}: IOU mean: {classIOU_mean} std: {classIOU_std}")

    NJD_mean, NJD_std = np.mean(np.array(jacobian_det_list)), np.std(np.array(jacobian_det_list))
    print(f"NJD mean: {NJD_mean} std: {NJD_std}")
    logger.info(f"NJD mean: {NJD_mean} std: {NJD_std}")


if __name__ == '__main__':

    # (py36pt16) liuhongzhi@user-SYS-7049GP-TRT:
    # /mnt/lhz/Github/Image_registration/SDHNet$ 
    # python /mnt/lhz/Github/Image_registration/code/compute_metrics.py

    ## ACDC
    # seg_txt = r'/mnt/lhz/Github/Image_registration/RDN/images/ACDC/test_img_seg_list.txt'
    # pred_path = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval'
    # save_dir = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32'
    seg_txt = r'/mnt/lhz/Github/Image_registration/RDP/images/ACDC/test_img_seg_list.txt'
    pred_path = r'/mnt/lhz/Github/Image_registration/RDP_checkpoints/ACDC/experiments/2024-09-06-15-15-37_RDP_ncc_0.1_reg_0.1_lr_1e-05/samples'
    save_dir = r'/mnt/lhz/Github/Image_registration/RDP_checkpoints/ACDC/experiments/2024-09-06-15-15-37_RDP_ncc_0.1_reg_0.1_lr_1e-05'

    # # LPBA
    # seg_txt = r'/mnt/lhz/Github/Image_registration/RDN/images/LPBA/test_img_seg_list.txt'
    # pred_path = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/LPBA_2024-11-04-19-53-53/eval'
    # save_dir = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/LPBA_2024-11-04-19-53-53/'

    # OASIS
    # seg_txt = r'/mnt/lhz/Github/Image_registration/RDN/images/OASIS/test_img_seg_list.txt'
    # pred_path = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/OASIS_2024-11-04-19-53-54/eval'
    # save_dir = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/OASIS_2024-11-04-19-53-54'

    ## OAIZIB
    # seg_txt = r'/mnt/lhz/Github/Image_registration/RDN/images/OAIZIB/test_img_seg_list.txt'
    # pred_path = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/OAIZIB_2024-11-04-19-53-57/eval'
    # save_dir = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/OAIZIB_2024-11-04-19-53-57'

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(save_dir, 'Quantitative_results.txt'))
    file_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    pred_gt_files = read_txt(seg_txt)
    # print(f"segment pairs: {pred_gt_files}")

    # compute_metrics(pred_path, pred_gt_files, save_dir, logger)
    compute_metrics_ACDC(pred_path, pred_gt_files, save_dir, logger)
    # compute_metrics_LPBA(pred_path, pred_gt_files, save_dir, logger)
    # compute_metrics_OASIS(pred_path, pred_gt_files, save_dir, logger)
    # compute_metrics_OAIZIB(pred_path, pred_gt_files, save_dir, logger)


    logger.removeHandler(file_handler)
    file_handler.close()
    
    # # seg = sorted(os.listdir(seg_path))
    # seg = sorted(os.listdir(gd_path))

    # dices = []
    # diceslist = []
    # hds = []
    # hdslist = []
    # rves = []
    # case_name = []
    # senss = []
    # specs = []
    # ious = []

    # # for name in seg:
    # for name in seg:
    #     print("name: ", name)
    #     # if not name.startswith('.') and name.endswith('nii.gz'):
    #     if not name.startswith('.') and name.endswith('nii.gz') and 'label' in name:
    #         # 加载label and segmentation image from MONAI
    #         pre_name = name[:9] + '_sample_' + str(600) + '_pred.nii.gz'
    #         # seg_ = nib.load(os.path.join(seg_path, name))
    #         seg_ = nib.load(os.path.join(seg_path, pre_name))
    #         seg_arr = seg_.get_fdata().astype('float32')
    #         gd_ = nib.load(os.path.join(gd_path, name))
    #         gd_arr = gd_.get_fdata().astype('float32')
    #         case_name.append(name[:9])
    #         print("seg_arr max:{} min:{}: ".format(seg_arr.max(), seg_arr.min()))
    #         print("gd_arr max:{} min:{}: ".format(gd_arr.max(), gd_arr.min()))
    #         class_num = int(gd_arr.max())

    #         # # 求IOU
    #         # iou_score = get_evaluation_score(seg_arr, gd_arr, spacing=None, metric='iou')
    #         # ious.append(iou_score)

    #         # 求dice get_evaluation_class_score(s_volume, g_volume, classnum, spacing, metric)
    #         dice, dicelist = get_evaluation_class_score(seg_.get_fdata(), gd_.get_fdata(), class_num, spacing=None, metric='dice')
    #         dices.append(dice)
    #         diceslist.append(dicelist)
            
    #         # 求hausdorff95距离  
    #         hd_score, hdlist = get_evaluation_class_score(seg_arr, gd_arr, class_num, spacing=None, metric='hausdorff95')
    #         hds.append(hd_score)
    #         hdslist.append(hdlist)

    #         # 求体积相关误差
    #         rve = get_evaluation_class_score(seg_arr, gd_arr, class_num, spacing=None, metric='rve')
    #         rves.append(rve)

    #         # 敏感度，特异性
    #         sens, spec = compute_class_sens_spec(seg_.get_fdata(), gd_.get_fdata())
    #         senss.append(sens)
    #         specs.append(spec)

    #         # print(dice, dicelist, hd_score, rve, sens, spec)
    #         print(dice, dicelist, hd_score, hdlist)
    
    # # 存入pandas
    # # data = {'dice': dices, 'dicelist': diceslist, 'HD95': hds, 'HD95list': hdslist, 'RVE': rves, 'Sens': senss, 'Spec': specs, 'IOU':ious}
    # data = {'dice': dices, 'dicelist': diceslist, 'HD95': hds, 'HD95list': hdslist, 'RVE': rves, 'Sens': senss, 'Spec': specs}
    # df = pd.DataFrame(data=data, columns=['dice', 'dicelist', 'HD95', 'HD95list', 'RVE', 'Sens', 'Spec'], index=case_name)
    # df.to_csv(os.path.join(save_dir, 'metrics.csv'))