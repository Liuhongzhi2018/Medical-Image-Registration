import os
import numpy
import numpy as np
import json

def json2txt(json_file, txt_file):
    
    with open(json_file,'r',encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    
    # print(f"json: {load_dict} {load_dict[0]}")
    with open(txt_file,'w') as f:
        for item in load_dict:
            # line = item["image_ED"] + " " + item['image_ES'] + "\n"  
            line = item["image_ES"] + " " + item['image_ED'] + "\n" # Moving, fixed
            print(f"line: {line}")
            f.write(line)
            
def json2txt1(json_file, txt_file):
    
    with open(json_file,'r',encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    
    # print(f"json: {load_dict} {load_dict[0]}")
    with open(txt_file,'w') as f:
        for item in load_dict:
            # line = item["image_ED"] + " " + item['image_ES'] + "\n"  
            line = item["image_ES"] + " " + item['label_ES'] + " " + item['image_ED'] + " " + item['label_ED'] + "\n" # Moving, fixed
            print(f"line: {line}")
            f.write(line)
            
def img2txt(img_dir, txt_file):
    
    test_img_dir = os.path.join(img_dir, "test")
    test_imgs = os.listdir(test_img_dir)
    print(test_img_dir)
    print(test_imgs)
    
    # print(f"json: {load_dict} {load_dict[0]}")
    with open(txt_file,'w') as f:
        for img in test_imgs:
            print(f"img: {img}")
            moving_img = os.path.join(test_img_dir, img)
            moving_seg = os.path.join(test_img_dir, img.split('.')[0]+'.delineation.structure.label.nii.gz')
            fixed_img = '/mnt/lhz/Datasets/Learn2reg/LPBA40/fixed.nii.gz'
            fixed_seg = '/mnt/lhz/Datasets/Learn2reg/LPBA40/label/S01.delineation.structure.label.nii.gz'
            line = moving_img + " " + moving_seg + " " + fixed_img + " " + fixed_seg + "\n" # Moving, fixed
            print(f"line: {line}")
            f.write(line)
            
def OASIS_img2txt1(root_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file):
    
    img_dir = os.path.join(root_dir, "imagesTr")
    # seg_dir = os.path.join(root_dir, "masksTr")
    seg_dir = os.path.join(root_dir, "labelsTr")
    imgs_list = os.listdir(img_dir)
    imgs_list.sort()
    print(imgs_list)
    
    for i in range(len(imgs_list) - 1):
        img = imgs_list[i]
        n = int(img.split("_")[1])
        print(f"{i} img: {img} {n}")
        if n < 395:
            with open(train_img_txt_file,'a') as f:
                # "moving": "./imagesTr/OASIS_0396_0000.nii.gz" "fixed": "./imagesTr/OASIS_0395_0000.nii.gz" 
                img_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(img_dir, imgs_list[i]) + "\n"
                print(f"train images {img_line}")
                f.write(img_line)
            with open(train_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i]) + "\n"
                print(f"train seg {seg_line}")
                f.write(seg_line)
        else:
            with open(val_img_txt_file,'a') as f:
                img_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(img_dir, imgs_list[i]) + "\n"
                print(f"val images {img_line}")
                f.write(img_line)
            with open(val_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i]) + "\n"
                print(f"val seg {seg_line}")
                f.write(seg_line)
    
    # # print(f"json: {load_dict} {load_dict[0]}")
    # with open(txt_file,'w') as f:
    #     for img in test_imgs:
    #         print(f"img: {img}")
    #         moving_img = os.path.join(test_img_dir, img)
    #         moving_seg = os.path.join(test_img_dir, img.split('.')[0]+'.delineation.structure.label.nii.gz')
    #         fixed_img = '/mnt/lhz/Datasets/Learn2reg/LPBA40/fixed.nii.gz'
    #         fixed_seg = '/mnt/lhz/Datasets/Learn2reg/LPBA40/label/S01.delineation.structure.label.nii.gz'
    #         line = moving_img + " " + moving_seg + " " + fixed_img + " " + fixed_seg + "\n" # Moving, fixed
    #         print(f"line: {line}")
    #         f.write(line)
    
def OASIS_img2txt2(root_dir, test_txt_file):
    img_dir = os.path.join(root_dir, "imagesTs")
    seg_dir = os.path.join(root_dir, "masksTs")
    imgs_list = os.listdir(img_dir)
    imgs_list.sort()
    print(imgs_list)
    
    for i in range(len(imgs_list)-1):
        img = imgs_list[i]
        n = int(img.split("_")[1])
        print(f"{i} img: {img} {n}")
        with open(test_txt_file,'a') as f:
            mov_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i+1])
            fix_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i])
            line = mov_line + " " + fix_line + "\n"
            print(f"test line {line}")
            f.write(line)
            
            
def OASIS_img2txt3(root_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file):
    
    img_dir = os.path.join(root_dir, "imagesTr")
    # seg_dir = os.path.join(root_dir, "masksTr")
    seg_dir = os.path.join(root_dir, "labelsTr")
    imgs_list = os.listdir(img_dir)
    imgs_list.sort()
    print(imgs_list)
    
    for i in range(len(imgs_list) - 1):
        img = imgs_list[i]
        n = int(img.split("_")[1])
        print(f"{i} img: {img} {n}")
        if n < 395:
            with open(train_img_txt_file,'a') as f:
                # "moving": "./imagesTr/OASIS_0396_0000.nii.gz" "fixed": "./imagesTr/OASIS_0395_0000.nii.gz" 
                img_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(img_dir, imgs_list[i]) + "\n"
                print(f"train images {img_line}")
                f.write(img_line)
            with open(train_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i]) + "\n"
                print(f"train seg {seg_line}")
                f.write(seg_line)
        else:
            with open(val_img_txt_file,'a') as f:
                img_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(img_dir, imgs_list[i]) + "\n"
                print(f"val images {img_line}")
                f.write(img_line)
            with open(val_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i]) + "\n"
                print(f"val seg {seg_line}")
                f.write(seg_line)
            with open("/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/test_img_seg_list.txt",'a') as f:
                mov_line = os.path.join(img_dir, imgs_list[i+1]) + " " + os.path.join(seg_dir, imgs_list[i+1])
                fix_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i])
                line = mov_line + " " + fix_line + "\n"
                print(f"test line {line}")
                f.write(line)
                

if __name__ == "__main__":
    # json_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train.json"
    # txt_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train_list.txt"
    # img_dir = "/mnt/lhz/Datasets/Learn2reg/LPBA40"
    # txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/LPBA/test_img_seg_list.txt"
    img_dir = "/mnt/lhz/Datasets/Learn2reg/OASIS"
    train_img_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/train_img_list.txt"
    train_seg_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/train_seg_list.txt"
    val_img_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/val_img_list.txt"
    val_seg_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/val_seg_list.txt"
    test_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OASIS/test_img_seg_list.txt"
    # OASIS_img2txt1(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    # OASIS_img2txt2(img_dir, test_txt_file)
    OASIS_img2txt3(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)