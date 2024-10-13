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
    
    
def change_to_label(img_path, new_label_path):
    
    img_itk = sitk.ReadImage(img_path)
    img_itk_origin = img_itk.GetOrigin()
    img_itk_spacing = img_itk.GetSpacing()
    img_itk_direction = img_itk.GetDirection()
    
    img_npy = sitk.GetArrayFromImage(img_itk)
    img_npy[img_npy == 0] = 0
    img_npy[img_npy == 1] = 0
    img_npy[img_npy == 2] = 0
    img_npy[img_npy == 3] = 1
    
    new_img_itk = sitk.GetImageFromArray(img_npy)
    new_img_itk.SetOrigin(img_itk_origin)
    new_img_itk.SetSpacing(img_itk_spacing)
    new_img_itk.SetDirection(img_itk_direction)
    
    sitk.WriteImage(new_img_itk, new_label_path)
    print(f"new label: {np.unique(img_npy)}")
    

if __name__ == '__main__':
    
    # label_from_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/labelsTr_3c"
    # label_from_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/labelsTs_3c"
    label_from_dir = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v6_CBCT/pair_1010"
    # label_to_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/Tibia_labelsTr"
    # label_to_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/Tibia_labelsTs"
    # label_to_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/Tibia_labelsTs_one"
    # label_to_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/Fibula_labelsTr"
    # label_to_dir = "/home/liuhongzhi/Data/Ankle/121_56/121_56_reorient/Fibula_labelsTs"
    label_to_dir = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v6_CBCT/pair_1010"
    
    os.makedirs(label_to_dir, exist_ok = True)

    img_path = os.path.join(label_from_dir, "jiao_N.nii.gz")
    new_label_path = os.path.join(label_to_dir, "jiao_N_3.nii.gz")
    
    change_to_label(img_path, new_label_path)
    print(f"Saving to: {new_label_path}")

    # label_files = os.listdir(label_from_dir)
    # for f in label_files:
    #     img_path = os.path.join(label_from_dir, f)
    #     new_label_path = os.path.join(label_to_dir, f)
        
    #     change_to_label(img_path, new_label_path)
    #     print(f"Saving to: {new_label_path}")
        
    
    '''
    val_img = r"D:\Project\20221026_gulou\20case_nifit\imagesTr\20210601004134.nii.gz"
    val_label1 = r"D:\Project\20221026_gulou\20case_nifit\20cases\20210601004134_cgrh.nii.gz"
    # val_label2 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210317000668_hh.nii.gz"
    
    # val_label2 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210530000540_gd_52.nii.gz"
    # val_label3 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210504000827_gd_44.nii.gz"
    # val_label4 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210504000827_gd_48.nii.gz"
    # val_label4 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210329001436_gd_52.nii.gz"
    # val_label5 = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210329001436_gd_64.nii.gz"
    # val_rlabel_files = [val_label2, val_label3, val_label4, val_label5]
    # val_rlabel_files = [val_label2]
    
    # val_rlabel_files = [
    #     # r"D:\Project\20221026_gulou\20case_nifit\20cases\20210530000540_gd_52.nii.gz",
    #     # r"D:\Project\20221026_gulou\20case_nifit\20cases\20210504000827_gd_44.nii.gz",
    #     # r"D:\Project\20221026_gulou\20case_nifit\20cases\20210504000827_gd_48.nii.gz",
    #     # r"D:\Project\20221026_gulou\20case_nifit\20cases\20210329001436_gd_64.nii.gz",
    #     # r"D:\Project\20221026_gulou\20case_nifit\20cases\20210317000668_hh.nii.gz",
    # ]
    # val_torlabel = r"D:\Project\20221026_gulou\20case_nifit\20cases\20210530000540_gdrh.nii.gz"
    
    val_tolabel = r"D:\Project\20221026_gulou\20case_nifit\labelsTr\20210601004134.nii.gz"
    
    # val_label_files = [val_label1, val_torlabel]
    val_label_files = [val_label1]
    
    # for f in val_rlabel_files:
    #     read_nii_label(f)
    # keep_image_with_one_label(val_img, val_rlabel_files, val_torlabel)
    
    for f in val_label_files:
        read_nii_label(f)
    keep_image_with_labels(val_img, val_label_files, val_tolabel)
    '''
    
    '''
    img_path = r"D:\Project\20221026_gulou\20case_nifit\imagesTs\20210426000851.nii.gz"
    label_path = r"D:\Project\20221026_gulou\20case_nifit\labelsTs\20210426000851.nii.gz"
    new_label_path = r"D:\Project\20221026_gulou\20case_nifit\labelsTs_one\20210426000851.nii.gz"
    keep_same_image_label(img_path, label_path, new_label_path)
    '''
    
    '''
    # img_dir = r"D:\Project\20221026_gulou\All_five\rg40\imagesAll"
    # label_dir = r"D:\Project\20221026_gulou\All_five\rg40\labelsAll_rg"
    # to_dir = r"D:\Project\20221026_gulou\All_five\rg40\labels"
    img_dir = r"D:\Project\20221026_gulou\20210611003517\120kVp-like 2.5mm C+__1.2.840.11__31745\imagesAll"
    label_dir = r"D:\Project\20221026_gulou\20210611003517\120kVp-like 2.5mm C+__1.2.840.11__31745\labelsAll_rg"
    to_dir = r"D:\Project\20221026_gulou\20210611003517\120kVp-like 2.5mm C+__1.2.840.11__31745"
    os.makedirs(to_dir, exist_ok = True)
    
    image_files = os.listdir(img_dir)
    label_files = os.listdir(label_dir)
    for f in image_files:
        image_id = f.split(".")[0]
        print("image: ", f, " id: ", image_id)
        label_paths = []
        label_list = []
        for l in label_files:
            if image_id in l:
                label_paths.append(os.path.join(label_dir, l))
                if 'cg' in l: label_list.append(1)
                elif 'gd' in l: label_list.append(2)
                elif 'rg' in l: label_list.append(3)
                elif 'hh' in l: label_list.append(4)
                    
        print("related labels: ", label_paths)
        print("label_list: ", label_list)
        
        img_path = os.path.join(img_dir, f)
        new_label_path = os.path.join(to_dir, f)
        print("Saving to: ", new_label_path)
        keep_image_with_labels_wlist(img_path, label_paths, label_list, new_label_path)
        time.sleep(5)
    '''