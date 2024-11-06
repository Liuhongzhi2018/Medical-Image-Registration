# 计算三维下各种指标
from __future__ import absolute_import, print_function
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk 
import pandas as pd
import GeodisTK
from scipy import ndimage

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


def compute_metrics(pred_path, pred_gt_files, save_dir):
    meanDice_list, eachDice_dict = [], {}
    for pairs in pred_gt_files:
        pair1, pair2 = pairs[0], pairs[1]
        pred_file = os.path.join(pred_path, pair1.split('/')[-1].split('_gt')[0] + '_ep5001_warped_seg.nii.gz')
        gt_file = pair2
        print(f"pred: {pred_file} gt: {gt_file}")
        if not os.path.exists(pred_file) or not os.path.exists(gt_file):
            raise Exception("file not exist!")
        # print(f"pred: {pred_file} gt: {gt_file}")

        pred_array, pred_size, pred_origin, pred_direction, pred_spacing = read_nii(pred_file)
        gt_array, gt_size, gt_origin, gt_direction, gt_spacing = read_nii(gt_file)
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

        pairs_Dice_class = []
        for i in np.unique(gt_array):
            # print(f"i: {i}")
            if i == 0: continue
            idice_value = calculate_dice(pred_array, gt_array, i)
            if i not in eachDice_dict.keys():
                eachDice_dict[i] = []
                eachDice_dict[i].append(idice_value)
            else:
                eachDice_dict[i].append(idice_value)
            pairs_Dice_class.append(idice_value)
        eachpair_meandice = sum(pairs_Dice_class)/len(pairs_Dice_class)
        meanDice_list.append(eachpair_meandice)

    # meanDice = sum(meanDice_list)/len(meanDice_list)
    dice_mean, dice_std = np.mean(np.array(meanDice_list)), np.std(np.array(meanDice_list))
    print(f"Dice mean: {dice_mean} std: {dice_std}")
    for c in eachDice_dict.keys():
        # mean_oneclass = sum(eachDice_dict[c])/len(eachDice_dict[c])
        classdice_mean, classdice_std = np.mean(np.array(eachDice_dict[c])), np.std(np.array(eachDice_dict[c]))
        print(f"class {c}: Dice mean: {classdice_mean} std: {classdice_std}")


if __name__ == '__main__':

    # seg_path = '/home/liuhz/Github/MedicalDetection/BrainFracSeg/MONAI/checkpoint/2021-12-19_23-00-22_UNet3D_HeadNeck/samples'
    seg_txt = r'/mnt/lhz/Github/Image_registration/RDN/images/ACDC/test_img_seg_list.txt'
    pred_path = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32/eval'
    save_dir = r'/mnt/lhz/Github/Image_registration/RDN_checkpoints/ACDC_2024-11-04-23-31-32'
    
    pred_gt_files = read_txt(seg_txt)
    # print(f"segment pairs: {pred_gt_files}")
    
    compute_metrics(pred_path, pred_gt_files, save_dir)
    
    
    
    
    
    
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