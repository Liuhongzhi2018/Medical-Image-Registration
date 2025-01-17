# -*- coding: utf-8 -*-
from multiprocessing.sharedctypes import Value
import os
from glob import glob
from PIL import Image
import numpy as np
import csv
import nrrd               # pip install pynrrd, if pynrrd is not already installed
import nibabel as nib     # pip install nibabel, if nibabel is not already installed
import SimpleITK as sitk
import time
import cv2
# (py39pt230) conda activate py39pt230

def edit_label(label_path, new_label_path):
    
    img_itk = sitk.ReadImage(label_path)
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_npy[img_npy == 1] = 1
    img_npy[img_npy == 2] = 1
    img_npy[img_npy == 3] = 1
    img_npy[img_npy == 4] = 1
    img_npy[img_npy == 5] = 1
    img_npy[img_npy == 6] = 1
    img_npy[img_npy == 7] = 1    
    img_npy[img_npy == 8] = 1
    img_npy[img_npy == 9] = 1
    img_itk = sitk.GetImageFromArray(img_npy)
    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(img_npy[img_npy != 0])


def edit_brain_label(label_path, new_label_path):
    
    img_itk = sitk.ReadImage(label_path)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()

    img_npy = sitk.GetArrayFromImage(img_itk)
    img_npy[img_npy == 2] = 1
    img_npy[img_npy == 3] = 1
    img_npy[img_npy == 4] = 1
    img_npy[img_npy == 5] = 1
    img_npy[img_npy == 6] = 1
    img_npy[img_npy == 7] = 1
    img_npy[img_npy == 8] = 1
    img_npy[img_npy == 9] = 1
    img_npy[img_npy == 10] = 1

    img_itk = sitk.GetImageFromArray(img_npy)
    img_itk.SetOrigin(origin)
    img_itk.SetSpacing(spacing)
    img_itk.SetDirection(direction)

    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(np.unique(img_npy))


def keep_same_image_label(img_path, label_path, new_label_path):
    
    refer_itk = sitk.ReadImage(img_path)
    img_itk = sitk.ReadImage(label_path)
    origin = refer_itk.GetOrigin()
    spacing = refer_itk.GetSpacing()
    direction = refer_itk.GetDirection()

    img_npy = sitk.GetArrayFromImage(img_itk)
    '''
    img_npy[img_npy == 2] = 1
    img_npy[img_npy == 3] = 1
    img_npy[img_npy == 4] = 1
    img_npy[img_npy == 5] = 1
    img_npy[img_npy == 6] = 1
    img_npy[img_npy == 7] = 1
    img_npy[img_npy == 8] = 1
    img_npy[img_npy == 9] = 1
    img_npy[img_npy == 10] = 1
    img_npy[img_npy == 11] = 1
    img_npy[img_npy == 12] = 1
    '''
    
    img_npy[img_npy == 2] = 0
    img_npy[img_npy == 3] = 0
    img_npy[img_npy == 4] = 0


    img_itk = sitk.GetImageFromArray(img_npy)
    img_itk.SetOrigin(origin)
    img_itk.SetSpacing(spacing)
    img_itk.SetDirection(direction)

    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(np.unique(img_npy))
    

def read_nii(label_path):

    img_itk = sitk.ReadImage(label_path)
    # print(img_itk)
    # print(img_itk.GetSize())
    img_npy = sitk.GetArrayFromImage(img_itk)
    print("=======label shape=======")
    # print(label_path, img_npy.shape)
    print(label_path, img_itk.GetSize())
    print("=======label value=======")
    # print(img_npy[img_npy != 0])
    print(np.unique(img_npy))


def read_nii_label(label_path):

    img = nib.load(label_path).get_fdata()
    print("=======label shape=======")
    # print(label_path, img_npy.shape)
    print(label_path, img.shape)
    print("=======label value=======")
    # print(img_npy[img_npy != 0])
    img_array = np.asarray(img)
    print(np.unique(img_array))
    # =======label shape=======
    # /mnt/520/lhz/Datasets/RibFrac2020/ribfrac-train-labels/train_label/RibFrac144-label.nii.gz (512, 512, 357)
    # =======label value=======
    # [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]


def edit_nii_label_1st(img, id, code):
    
    # print(type(img), img.dtype, id, type(id), code, type(code))
    # v = float(id)
    v = np.float64(id)
    c = int(code)
    # print("v: ",v, " c: ", c)
    if c == 0:
        img[img == v] = np.float64(50)
    elif c == 1:
        img[img == v] = np.float64(51)
    elif c == 2:
        img[img == v] = np.float64(52)
    elif c == 3:
        img[img == v] = np.float64(53)
    elif c == 4:
        img[img == v] = np.float64(54)
    else:
        img[img == v] = np.float64(55)
    # print("edit_nii_label: ", np.unique(img))
    return img

    # RibFrac2,0,0
    # RibFrac2,1,-1
    # RibFrac2,2,-1
    # RibFrac2,3,-1
    # RibFrac2,4,2
    # RibFrac2,5,2
    # RibFrac2,6,2
    # RibFrac2,7,-1
    # RibFrac2,8,-1
    # RibFrac2,9,-1

def edit_nii_label_2nd(img):
    
    img[img == np.float64(50)] = np.float64(0)
    img[img == np.float64(51)] = np.float64(1)
    img[img == np.float64(52)] = np.float64(2)
    img[img == np.float64(53)] = np.float64(3)
    img[img == np.float64(54)] = np.float64(4)
    img[img == np.float64(55)] = np.float64(5)
    return img


def read_and_edit_nii_label(sname, tname, id, code):

    img = nib.load(sname)
    img_affine = img.affine
    img_data = img.get_fdata()
    img_array = np.asarray(img_data)
    # print("Label in: ", np.unique(img_array))
    # print(img_affine, img_data.dtype, np.unique(img_array))
    newlabel = edit_nii_label_1st(img_data, id, code)
    # print("Newlabel: ", np.unique(newlabel))
    nib.Nifti1Image(newlabel, img_affine).to_filename(tname)

# Label in:  [0. 1. 2.]Newlabel:  [ 1.  2. 50.]Saving:  /mnt/520/lhz/Datasets/RibFrac2020/ribfrac-train-labels/train_label_new/RibFrac1-label.nii.gz
# Label in:  [ 1.  2. 50.]Newlabel:  [ 2. 50. 52.]
# Saving:  /mnt/520/lhz/Datasets/RibFrac2020/ribfrac-train-labels/train_label_new/RibFrac1-label.nii.gzLabel in:  [ 2. 50. 52.]Newlabel:  [50. 52.]


def read_and_edit_new_label(sname, tname):

    img = nib.load(sname)
    img_affine = img.affine
    img_data = img.get_fdata()
    img_array = np.asarray(img_data)
    # print(img_affine, img_data.dtype, np.unique(img_array))
    newlabel = edit_nii_label_2nd(img_data)
    print("Stage2 newlabel: ", np.unique(newlabel))
    nib.Nifti1Image(newlabel, img_affine).to_filename(tname)
    
    
def keep_image_with_labels(img_path, label_paths, new_label_path):
    
    refer_itk = sitk.ReadImage(img_path)
    
    img_npy_tmp = 0
    # label_list = [1, 2, 3, 4]  # chenggu, gudao, ronggu, hunhe
    label_list = [1, 2]
    for l in range(len(label_paths)):
        img_itk = sitk.ReadImage(label_paths[l])
        origin = refer_itk.GetOrigin()
        spacing = refer_itk.GetSpacing()
        direction = refer_itk.GetDirection()

        img_npy = sitk.GetArrayFromImage(img_itk)
        img_npy[img_npy == 1] = label_list[l]
        img_npy_tmp += img_npy

    img_itk = sitk.GetImageFromArray(img_npy_tmp)
    img_itk.SetOrigin(origin)
    img_itk.SetSpacing(spacing)
    img_itk.SetDirection(direction)

    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(np.unique(img_npy_tmp))
    
    
def keep_image_with_one_label(img_path, label_paths, new_label_path):
    
    refer_itk = sitk.ReadImage(img_path)
    
    img_npy_tmp = 0
    for l in range(len(label_paths)):
        img_itk = sitk.ReadImage(label_paths[l])
        origin = refer_itk.GetOrigin()
        spacing = refer_itk.GetSpacing()
        direction = refer_itk.GetDirection()

        img_npy = sitk.GetArrayFromImage(img_itk)
        img_npy[img_npy == 1] = 1
        img_npy_tmp += img_npy

    img_itk = sitk.GetImageFromArray(img_npy_tmp)
    img_itk.SetOrigin(origin)
    img_itk.SetSpacing(spacing)
    img_itk.SetDirection(direction)

    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(np.unique(img_npy_tmp))
    
    
def keep_image_with_labels_wlist(img_path, label_paths, label_list, new_label_path):
    
    refer_itk = sitk.ReadImage(img_path)
    
    img_npy_tmp = 0
    # label_list = [1, 2, 3, 4]  # chenggu, gudao, ronggu, hunhe
    # label_list = [1, 2]
    for l in range(len(label_paths)):
        img_itk = sitk.ReadImage(label_paths[l])
        origin = refer_itk.GetOrigin()
        spacing = refer_itk.GetSpacing()
        direction = refer_itk.GetDirection()

        img_npy = sitk.GetArrayFromImage(img_itk)
        img_npy[img_npy == 1] = label_list[l]
        img_npy_tmp += img_npy

    img_itk = sitk.GetImageFromArray(img_npy_tmp)
    img_itk.SetOrigin(origin)
    img_itk.SetSpacing(spacing)
    img_itk.SetDirection(direction)

    sitk.WriteImage(img_itk, new_label_path)
    print("=======changed label value=======")
    print(np.unique(img_npy_tmp))
    
    

def compute_range(img_dir):
    print(f"compute_range img_path {img_dir}")
    image_files = os.listdir(img_dir)
    print(f"compute_range image_files {image_files}")
    value_min, value_max= 0, 0
    value_min_list, value_max_list = [], []
    for f in image_files:
        img_path = os.path.join(img_dir, f)
        img_itk = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_itk)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        print(f"{f} {img_array_min} {img_array_max}")
        value_min_list.append(img_array_min)
        value_max_list.append(img_array_max)
        # if img_array_min < value_min:
        #     value_min = img_array_min
        # if img_array_max > value_max:
        #     value_max = img_array_max
    value_min = min(value_min_list)
    value_max = max(value_max_list)
    # value_min = max(value_min_list)
    # value_max = min(value_max_list)
    # value_min = int(np.average(value_min_list))
    # value_max = int(np.average(value_max_list))
    print(f"compute_range: {value_min} {value_max}")
    return (value_min, value_max)

# 1702242404KongDeQi_C1.nii.gz -1024 1687
# 2201078137YangGuiRong_B2.nii.gz -1024 1750
# 1909164442ZhaoYuJiang_B2.nii.gz -1024 1810
# 2009086360YangXianLong_C2.nii.gz -1024.0 2022.0
# fengshuo_A2.nii.gz -1024 2048
# 1703215579MaXiuYun_A2.nii.gz -1024 1912
# doubleFoot2_N.nii.gz -1000 2269
# 2102098223ChengZhaoHu_B3.nii.gz -1024.0 1788.0
# 2102021529JiaZhengTing_A3.nii.gz -1024.0 1959.0
# 2112152649JinDe_C1.nii.gz -1024 1735
# zhangbangwei_C1.nii.gz -1024 2894
# 2109121680CaoYuShan_B2.nii.gz -1024 1742
# YuanShiZeng_C2.nii.gz -1024 2049
# 2104245273DuXiangZhong_B1.nii.gz -1024.0 2069.0
# 2106296091LiuHongJun_A1.nii.gz -1024 1751
# 2201263775ZhangHao_C2.nii.gz -1024 1679
# 2106091911XingChangBin_B1.nii.gz -1024.0 1802.0
# lifengqin_C2.nii.gz -1024 1783
# 2112309166FuZhiMin_B1.nii.gz -1024 1755
# 2001106502HeCongLin_B2.nii.gz -1024 1814
# 2111086946ZhengZheng_A1.nii.gz -1024 1765
# 2101097700GuanGuiJiang_B3.nii.gz -1024.0 1953.0
# 2101075993LvFengHu_B3.nii.gz -1024.0 2000.0
# 2104094088WangChuanNing_A1.nii.gz -1024.0 2068.0
# 2107288782WangZengJian_C1.nii.gz -1024 2397
# 2110114540LiuZheYue_B1.nii.gz -1024 1704
# 2111110023SongYuXiang_B2.nii.gz -1024.0 1767.0
# 2208202516XuJinPeng_C2.nii.gz -1024.0 1801.0
# 2104061892ZhangJianJun_C1.nii.gz -1024.0 1828.0
# 2112131186YangYue_A1.nii.gz -1024 1777
# 2107267094PanJingYi_A1.nii.gz -1024 1564
# 2110180129NiuWenXian_A1.nii.gz -1024 1819
# WangBoYa_B3.nii.gz -1024 2002
# TengBaoZhu_B1.nii.gz -1024 1836
# 2111066331DuBaoKui_B1.nii.gz -1024 1769
# CaiSongTao_A1.nii.gz -1024.0 2286.991943359375
# wangzihao_B2.nii.gz -1024 3071
# chenyueliang_A1.nii.gz -1024 1780
# 2005299708ZhangYiDan_C1.nii.gz -1024 2030
# 1803134721CuiTongWei_C2.nii.gz -1024 1787
# 2101295535ZhouJinHao_C1.nii.gz -1024 1673
# 2007275601LiuSiQing_C2.nii.gz -1024.0 1801.0
# doubleFoot1_N.nii.gz -1000 2031
# 2007102698XinYongMei_C1.nii.gz -1024.0 1747.0
# jiao_N.nii.gz -1024.0 1984.0
# 2104256084WenHongLan_A1.nii.gz -1024.0 2020.0
# ZhangBingYun_A1.nii.gz -1024 2013
# 2101317570LiuHaiYan_B3.nii.gz -1024.0 1992.0
# 2103114352LiChenYan_A1.nii.gz -1024.0 2604.0
# 1703141013LiDan_A2.nii.gz -1024 1768
# cuishejun_A3.nii.gz -1024 2032
# 2106166463ZhangHaiLi_A1.nii.gz -1024 1771
# 1704140002JiaJun_C1.nii.gz -1024 1801
# 2012071643YaoFangChao_B3.nii.gz -1024.0 1776.0
# 2007307851LiBo_C2.nii.gz -1024.0 1707.0
# 2102032894SuJianWen_A3.nii.gz -1024.0 1931.0


def save_norm(fimg_dir, timg_dir, value_range):
    fimage_files = os.listdir(fimg_dir)
    all_num = len(fimage_files)
    count = 0
    for f in fimage_files:
        count += 1
        img_path = os.path.join(fimg_dir, f)
        img_itk = sitk.ReadImage(img_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        
        img_array = sitk.GetArrayFromImage(img_itk)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        
        # norm
        # new_img_array = value_range[0] + ((img_array - img_array_min) / (img_array_max - img_array_min)) * (value_range[1] - value_range[0])
        # clip
        new_img_array = img_array
        new_img_array[new_img_array > value_range[1]] = value_range[1]
        new_img_array[new_img_array < value_range[0]] = value_range[0]
        
        new_img_array_min = np.min(new_img_array)
        new_img_array_max = np.max(new_img_array)
        
        new_img_itk = sitk.GetImageFromArray(new_img_array)
        new_img_itk.SetOrigin(origin)
        new_img_itk.SetSpacing(spacing)
        new_img_itk.SetDirection(direction)
        new_img_path = os.path.join(timg_dir, f)
        sitk.WriteImage(new_img_itk, new_img_path)
        print(f"{count} / {all_num} {f} ( {img_array_min} {img_array_max} ) norm to ( {new_img_array_min} {new_img_array_max}) ")
    
    
def save_MAR_norm(fimg_dir, timg_dir, value_range):
    fimage_files = os.listdir(fimg_dir)
    all_num = len(fimage_files)
    MAR_threshold = value_range[1]  # +1000
    # MAR_threshold = value_range[1] - 500 # +1000
    # MAR_threshold = np.percentile(a, 90)
    count = 0
    for f in fimage_files:
        count += 1
        img_path = os.path.join(fimg_dir, f)
        img_itk = sitk.ReadImage(img_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        
        img_array = sitk.GetArrayFromImage(img_itk)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        
        img_array_uint8 = (img_array - img_array_min) / (img_array_max - img_array_min) * 255
        MAR_threshold_uint8 = (MAR_threshold - img_array_min) / (img_array_max - img_array_min) * 255
        
        # MAR
        # print(f"img_array shape: {img_array.shape} type: {img_array.dtype}")  # (Dz, Hy, Wx)
        for i in range(img_array.shape[0]):
            # img_slice = img_array[i, :, :]
            img_uint8_slice = img_array_uint8[i, :, :]
            img_array_slice = img_array[i, :, :]
            # print(f"{f} {i} img_slice shape: {img_slice.shape} {img_slice.dtype} {MAR_threshold} {img_array_max}")
            _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY)
            # _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            # kernel = np.ones((5, 5), np.uint8)
            # kernel = np.ones((10, 10), np.uint8)                                                                                                                                                                                                       
            overmask = cv2.dilate(mask, kernel, iterations=1)
            # print(f"img_slice shape: {img_slice.shape} overmask shape: {overmask.shape}")  # (Dz, Hy, Wx)
            # img_slice shape: (394, 394) overmask shape: (394, 394)
            # img_array[i, :, :] = cv2.inpaint(np.uint16(img_slice), overmask, 5, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
            img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + value_range[0] * overmask
            
        # norm
        # new_img_array = value_range[0] + ((img_array - img_array_min) / (img_array_max - img_array_min)) * (value_range[1] - value_range[0])
        
        # clip
        new_img_array = img_array
        new_img_array[new_img_array > value_range[1]] = value_range[1]
        new_img_array[new_img_array < value_range[0]] = value_range[0]
        
        new_img_array_min = np.min(new_img_array)
        new_img_array_max = np.max(new_img_array)
        
        new_img_itk = sitk.GetImageFromArray(new_img_array)
        new_img_itk.SetOrigin(origin)
        new_img_itk.SetSpacing(spacing)
        new_img_itk.SetDirection(direction)
        new_img_path = os.path.join(timg_dir, f)
        sitk.WriteImage(new_img_itk, new_img_path)
        print(f"{count} / {all_num} {f} ( {img_array_min} {img_array_max} ) norm to ( {new_img_array_min} {new_img_array_max}) ")


def save_MAR_norm1(fimg_dir, timg_dir, value_range):
    fimage_files = os.listdir(fimg_dir)
    all_num = len(fimage_files)
    MAR_threshold = value_range[1]  # +1000
    # MAR_threshold = value_range[1] - 500 # +1000
    # MAR_threshold = np.percentile(a, 90)
    count = 0
    for f in fimage_files:
        count += 1
        img_path = os.path.join(fimg_dir, f)
        img_itk = sitk.ReadImage(img_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        
        img_array = sitk.GetArrayFromImage(img_itk)
        # print(f"{f} img size: {img_itk.GetSize()}")
        # print(f"{f} img size: {img_itk.GetSize()} array shape: {img_array.shape}")
        # 20220815.144447-2.nii.gz img size: (394, 394, 394) (WHD) array shape: (394, 394, 394) (DHW)
        # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
        # img_array before mask: -1000 27173)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        img_array_mean = np.mean(img_array)
        
        # deep copy from origin array
        img_array_over0 = img_array.copy() 
        img_array_over0[img_array_over0 < 0] = 0
        img_array_meanover0 = np.mean(img_array_over0)
        # print(f"img_array_min: {np.min(img_array)} img_array_max: {np.max(img_array)} img_array_meanover0: {img_array_meanover0}")
        # img_array_min: -1000 img_array_max: 27173 img_array_meanover0: 31.85126410771587
        
        # norm to (0, 255)
        img_array_uint8 = (img_array - img_array_min) / (img_array_max - img_array_min) * 255
        MAR_threshold_uint8 = (MAR_threshold - img_array_min) / (img_array_max - img_array_min) * 255
        
        # MAR
        # print(f"img_array shape: {img_array.shape} type: {img_array.dtype}")  # (Dz, Hy, Wx)
        for i in range(img_array.shape[0]):
            # img_slice = img_array[i, :, :]
            img_uint8_slice = img_array_uint8[i, :, :]
            # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
            img_array_slice = img_array[i, :, :]
            # print(f"{f} {i} img_slice shape: {img_slice.shape} {img_slice.dtype} {MAR_threshold} {img_array_max}")
            _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY)
            # _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY_INV)
            # kernel = np.ones((3, 3), np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            # kernel = np.ones((10, 10), np.uint8)                                                                                                                                                                                                       
            overmask = cv2.dilate(mask, kernel, iterations=2)
            # print(f"img_slice shape: {img_slice.shape} overmask shape: {overmask.shape}")  # (Dz, Hy, Wx)
            # img_slice shape: (394, 394) overmask shape: (394, 394)
            # img_array[i, :, :] = cv2.inpaint(np.uint16(img_slice), overmask, 5, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + value_range[0] * overmask
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + img_array_mean * overmask
            # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
            img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            # print(f"img_array with mask: {np.min(img_array)} {np.max(img_array)}) ")
            # img_array with mask: -1000 27173)
            
        # norm
        # new_img_array = value_range[0] + ((img_array - img_array_min) / (img_array_max - img_array_min)) * (value_range[1] - value_range[0])
        
        # clip
        new_img_array = img_array
        new_img_array[new_img_array > value_range[1]] = value_range[1]
        new_img_array[new_img_array < value_range[0]] = value_range[0]
        
        new_img_array_min = np.min(new_img_array)
        new_img_array_max = np.max(new_img_array)
        
        new_img_itk = sitk.GetImageFromArray(new_img_array)
        new_img_itk.SetOrigin(origin)
        new_img_itk.SetSpacing(spacing)
        new_img_itk.SetDirection(direction)
        new_img_path = os.path.join(timg_dir, f)
        sitk.WriteImage(new_img_itk, new_img_path)
        print(f"{count} / {all_num} {f} ( {img_array_min} {img_array_max} ) norm to ( {new_img_array_min} {new_img_array_max}) ")



def save_MAR_norm_inpaint(fimg_dir, timg_dir, value_range):
    fimage_files = os.listdir(fimg_dir)
    all_num = len(fimage_files)
    MAR_threshold = value_range[1]  # +1000
    # MAR_threshold = value_range[1] - 500 # +1000
    # MAR_threshold = np.percentile(a, 90)
    count = 0
    for f in fimage_files:
        count += 1
        img_path = os.path.join(fimg_dir, f)
        img_itk = sitk.ReadImage(img_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        
        img_array = sitk.GetArrayFromImage(img_itk)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        
        img_array_uint8 = (img_array - img_array_min) / (img_array_max - img_array_min) * 255
        MAR_threshold_uint8 = (MAR_threshold - img_array_min) / (img_array_max - img_array_min) * 255
        
        # MAR
        # print(f"img_array shape: {img_array.shape} type: {img_array.dtype}")  # (Dz, Hy, Wx)
        for i in range(img_array.shape[0]):
            # img_slice = img_array[i, :, :]
            img_uint8_slice = img_array_uint8[i, :, :]
            img_array_slice = img_array[i, :, :]
            # print(f"{f} {i} img_slice shape: {img_slice.shape} {img_slice.dtype} {MAR_threshold} {img_array_max}")
            _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY)
            # _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY_INV)
            kernel = np.ones((3, 3), np.uint8)
            # kernel = np.ones((5, 5), np.uint8)
            # kernel = np.ones((10, 10), np.uint8)                                                                                                                                                                                                       
            overmask = cv2.dilate(mask, kernel, iterations=1)
            # print(f"img_slice shape: {img_slice.shape} overmask shape: {overmask.shape}")  # (Dz, Hy, Wx)
            # img_slice shape: (394, 394) overmask shape: (394, 394)
            # img_array[i, :, :] = cv2.inpaint(np.uint16(img_slice), overmask, 5, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + value_range[0] * overmask
            img_array[i, :, :] = cv2.inpaint(np.uint16(img_array_slice), overmask, 3, cv2.INPAINT_TELEA)
            
        # norm
        new_img_array = value_range[0] + ((img_array - img_array_min) / (img_array_max - img_array_min)) * (value_range[1] - value_range[0])
        
        # clip
        # new_img_array = img_array
        new_img_array[new_img_array > value_range[1]] = value_range[1]
        new_img_array[new_img_array < value_range[0]] = value_range[0]
        
        new_img_array_min = np.min(new_img_array)
        new_img_array_max = np.max(new_img_array)
        
        new_img_itk = sitk.GetImageFromArray(new_img_array)
        new_img_itk.SetOrigin(origin)
        new_img_itk.SetSpacing(spacing)
        new_img_itk.SetDirection(direction)
        new_img_path = os.path.join(timg_dir, f)
        sitk.WriteImage(new_img_itk, new_img_path)
        print(f"{count} / {all_num} {f} ( {img_array_min} {img_array_max} ) norm to ( {new_img_array_min} {new_img_array_max}) ")


def filter_with_mask(imgarray, ker=3):
    
    # print(f"filter_with_mask: kernel {ker}")
    newarray = cv2.blur(imgarray, (ker, ker))
    # newarray = cv2.medianBlur(imgarray, ker) # Unsupported format
    # newarray = cv2.GaussianBlur(imgarray, (ker, ker), 0)
    # newarray = cv2.bilateralFilter(imgarray, 15, 150, 150)
    
    return newarray
    
def save_MAR_norm_filter(fimg_dir, timg_dir, value_range):
    fimage_files = os.listdir(fimg_dir)
    all_num = len(fimage_files)
    MAR_threshold = value_range[1]  # +1000
    # MAR_threshold = value_range[1] - 500 # +1000
    # MAR_threshold = np.percentile(a, 90)
    count = 0
    for f in fimage_files:
        count += 1
        img_path = os.path.join(fimg_dir, f)
        img_itk = sitk.ReadImage(img_path)
        origin = img_itk.GetOrigin()
        spacing = img_itk.GetSpacing()
        direction = img_itk.GetDirection()
        
        img_array = sitk.GetArrayFromImage(img_itk)
        # print(f"{f} img size: {img_itk.GetSize()}")
        # print(f"{f} img size: {img_itk.GetSize()} array shape: {img_array.shape}")
        # 20220815.144447-2.nii.gz img size: (394, 394, 394) (WHD) array shape: (394, 394, 394) (DHW)
        # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
        # img_array before mask: -1000 27173)
        img_array_min = np.min(img_array)
        img_array_max = np.max(img_array)
        img_array_mean = np.mean(img_array)
        
        # deep copy from origin array
        img_array_over0 = img_array.copy() 
        img_array_over0[img_array_over0 < 0] = 0
        img_array_meanover0 = np.mean(img_array_over0)
        # print(f"img_array_min: {np.min(img_array)} img_array_max: {np.max(img_array)} img_array_meanover0: {img_array_meanover0}")
        # img_array_min: -1000 img_array_max: 27173 img_array_meanover0: 31.85126410771587
        
        # norm to (0, 255)
        img_array_uint8 = (img_array - img_array_min) / (img_array_max - img_array_min) * 255
        MAR_threshold_uint8 = (MAR_threshold - img_array_min) / (img_array_max - img_array_min) * 255
        
        # MAR
        # print(f"img_array shape: {img_array.shape} type: {img_array.dtype}")  # (Dz, Hy, Wx)
        for i in range(img_array.shape[0]):
            # img_slice = img_array[i, :, :]
            img_uint8_slice = img_array_uint8[i, :, :]
            # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
            img_array_slice = img_array[i, :, :]
            # print(f"{f} {i} img_slice shape: {img_slice.shape} {img_slice.dtype} {MAR_threshold} {img_array_max}")
            _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY)
            # _, mask = cv2.threshold(np.uint8(img_uint8_slice), MAR_threshold_uint8, 1, cv2.THRESH_BINARY_INV)
            # kernel = np.ones((3, 3), np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            # kernel = np.ones((10, 10), np.uint8)                                                                                                                                                                                                       
            overmask = cv2.dilate(mask, kernel, iterations=1)
            # print(f"img_slice shape: {img_slice.shape} overmask shape: {overmask.shape}")  # (Dz, Hy, Wx)
            # img_slice shape: (394, 394) overmask shape: (394, 394)
            # img_array[i, :, :] = cv2.inpaint(np.uint16(img_slice), overmask, 5, cv2.INPAINT_TELEA)  # cv2.INPAINT_NS
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + value_range[0] * overmask
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + img_array_mean * overmask
            # print(f"img_array before mask: {np.min(img_array)} {np.max(img_array)}) ")
            # img_array[i, :, :] = img_array_slice * np.int32(1 - overmask)
            img_array_slice_mask = img_array_slice * np.int32(1 - overmask)
            # print(f"img_array with mask: {np.min(img_array)} {np.max(img_array)}) ")
            # img_array with mask: -1000 27173)
            
            # filter image with mask
            img_array_slice_mask = filter_with_mask(img_array_slice_mask, 5)
            img_array[i, :, :] = img_array_slice * np.int32(1 - overmask) + img_array_slice_mask * np.int32(overmask)
            
        # norm
        # new_img_array = value_range[0] + ((img_array - img_array_min) / (img_array_max - img_array_min)) * (value_range[1] - value_range[0])
        
        # clip
        new_img_array = img_array
        new_img_array[new_img_array > value_range[1]] = value_range[1]
        new_img_array[new_img_array < value_range[0]] = value_range[0]
        
        new_img_array_min = np.min(new_img_array)
        new_img_array_max = np.max(new_img_array)
        
        new_img_itk = sitk.GetImageFromArray(new_img_array)
        new_img_itk.SetOrigin(origin)
        new_img_itk.SetSpacing(spacing)
        new_img_itk.SetDirection(direction)
        new_img_path = os.path.join(timg_dir, f)
        sitk.WriteImage(new_img_itk, new_img_path)
        print(f"{count} / {all_num} {f} ( {img_array_min} {img_array_max} ) norm to ( {new_img_array_min} {new_img_array_max}) ")


if __name__ == '__main__':
    
    refimg_dir = r"/mnt/lhz/Github/SEU_Ankle_MONAI/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task02_12156deno-threelabel/imagesTs"
    # fimg_dir = r"/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v6_CBCT/NIFTI"
    fimg_dir = r"/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v8_No6/NIFTI"
    # timg_dir = r"/mnt/lhz/Github/SEU_Ankle_MONAI/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task04_ankletest/imagesTs_clip-10243071"
    timg_dir = r"/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v8_No6/NIFTI_demetal"
    os.makedirs(timg_dir, exist_ok = True)
    
    value_range = compute_range(refimg_dir)
    # value_range = (-1024, 3071)
    # value_range = (-1024, 3071)  min-max
    # compute_range: -1000 1564
    # compute_range: -1023 1928    average
    # save_norm(fimg_dir, timg_dir, value_range)
    # save_MAR_norm(fimg_dir, timg_dir, value_range)
    # save_MAR_norm1(fimg_dir, timg_dir, value_range)
    # save_MAR_norm_inpaint(fimg_dir, timg_dir, value_range)
    save_MAR_norm_filter(fimg_dir, timg_dir, value_range)
    
 