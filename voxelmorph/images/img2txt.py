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

if __name__ == "__main__":
    # json_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train.json"
    # txt_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train_list.txt"
    img_dir = "/mnt/lhz/Datasets/Learn2reg/LPBA40"
    txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/LPBA/test_img_seg_list.txt"
    img2txt(img_dir, txt_file)