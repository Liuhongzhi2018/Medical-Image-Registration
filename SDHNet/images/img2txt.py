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
                
                
def ACTCT_img2txt(root_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file):
    img_dir = os.path.join(root_dir, "imagesTr")
    # seg_dir = os.path.join(root_dir, "masksTr")
    seg_dir = os.path.join(root_dir, "labelsTr")
    imgs_list = os.listdir(img_dir)
    imgs_list.sort()
    print(imgs_list)
    
    for i in range(len(imgs_list)):
        img = imgs_list[i]
        n = int(img.split("_")[1])
        print(f"{i} img: {img} {n}")
        if n == 1: continue
        # if n == 4 or n == 7 or n == 10 or n == 13 or n == 16 or n == 19 or n == 22 or n == 25 or n == 28:
        # for j in range(i + 1, len(imgs_list)):
        for j in range(len(imgs_list)):
            imgj = imgs_list[j]
            nj = int(imgj.split("_")[1])
            if n == nj: continue
            elif ((n == 4 and nj == 1) or (n == 7 and nj == 1) or (n == 10 and nj == 1) or (n == 13 and nj == 1) or (n == 16 and nj == 1) or 
                  (n == 19 and nj == 1) or (n == 22 and nj == 1) or (n == 25 and nj == 1) or (n == 28 and nj == 1) or
                  (n == 7 and nj == 4) or (n == 10 and nj == 4) or (n == 13 and nj == 4) or (n == 16 and nj == 4) or
                  (n == 19 and nj == 4) or (n == 22 and nj == 4) or (n == 25 and nj == 4) or (n == 28 and nj == 4) or
                  (n == 10 and nj == 7) or (n == 13 and nj == 7) or (n == 16 and nj == 7) or (n == 19 and nj == 7) or
                  (n == 22 and nj == 7) or (n == 25 and nj == 7) or (n == 28 and nj == 7) or (n == 13 and nj == 10) or
                  (n == 16 and nj == 10) or (n == 19 and nj == 10) or (n == 22 and nj == 10) or (n == 25 and nj == 10) or
                  (n == 28 and nj == 10) or (n == 16 and nj == 13) or (n == 19 and nj == 13) or (n == 22 and nj == 13) or
                  (n == 25 and nj == 13) or (n == 28 and nj == 13) or (n == 19 and nj == 16) or (n == 22 and nj == 16) or
                  (n == 25 and nj == 16) or (n == 28 and nj == 16) or (n == 22 and nj == 19) or (n == 25 and nj == 19) or
                  (n == 28 and nj == 19) or (n == 25 and nj == 22) or (n == 28 and nj == 22) or (n == 28 and nj == 25)):
                with open(val_img_txt_file,'a') as f:
                    img_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(img_dir, imgs_list[j]) + "\n"
                    print(f"val images {img_line}")
                    f.write(img_line)
                with open(val_seg_txt_file,'a') as f:
                    seg_line = os.path.join(seg_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[j]) + "\n"
                    print(f"val seg {seg_line}")
                    f.write(seg_line)
                with open("/mnt/lhz/Github/Image_registration/voxelmorph/images/AbdomenCTCT/test_img_seg_list.txt",'a') as f:
                    mov_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i])
                    fix_line = os.path.join(img_dir, imgs_list[j]) + " " + os.path.join(seg_dir, imgs_list[j])
                    line = mov_line + " " + fix_line + "\n"
                    print(f"test line {line}")
                    f.write(line)
            else:
                with open(train_img_txt_file,'a') as f:
                    img_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(img_dir, imgs_list[j]) + "\n"
                    print(f"train images {img_line}")
                    f.write(img_line)
                with open(train_seg_txt_file,'a') as f:
                    seg_line = os.path.join(seg_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[j]) + "\n"
                    print(f"train seg {seg_line}")
                    f.write(seg_line)
                    
def ACTMR_img2txt(root_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file):
    img_dir = os.path.join(root_dir, "imagesTr")
    # seg_dir = os.path.join(root_dir, "masksTr")
    seg_dir = os.path.join(root_dir, "labelsTr")
    imgs_list = os.listdir(img_dir)
    imgs_list.sort()
    print(imgs_list)
    
    for i in range(len(imgs_list)):
        img = imgs_list[i]
        n = int(img.split("_")[1])
        m = int(img.split("_")[2].split('.')[0])
        print(f"{i} img: {img} {n} {m}")
        
        if  n == 1 or n == 2 or n == 3 or n == 4 or n == 5:
            if m == 0: continue
            with open(train_img_txt_file,'a') as f:
                img_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(img_dir, imgs_list[i].replace("0001", "0000")) + "\n"
                print(f"train images {img_line}")
                f.write(img_line)
            with open(train_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i].replace("0001", "0000")) + "\n"
                print(f"train seg {seg_line}")
                f.write(seg_line)
      
        elif  n == 6 or n == 7 or n == 8:
            if m == 0: continue
            with open("/mnt/lhz/Github/Image_registration/voxelmorph/images/AbdomenMRCT/CT2MR/test_allimg_allseg_list.txt",'a') as f:
                mov_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i])
                fix_line = os.path.join(img_dir, imgs_list[i].replace("0001", "0000")) + " " + os.path.join(seg_dir, imgs_list[i].replace("0001", "0000"))
                line = mov_line + " " + fix_line + "\n"
                print(f"test line {line}")
                f.write(line)
            with open(val_img_txt_file,'a') as f:
                img_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(img_dir, imgs_list[i].replace("0001", "0000")) + "\n"
                print(f"val images {img_line}")
                f.write(img_line)
            with open(val_seg_txt_file,'a') as f:
                seg_line = os.path.join(seg_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[i].replace("0001", "0000")) + "\n"
                print(f"val seg {seg_line}")
                f.write(seg_line)
                
        else:
            for j in range(len(imgs_list)):
                imgj = imgs_list[j]
                nj = int(imgj.split("_")[1])
                mj = int(imgj.split("_")[2].split('.')[0])
                
                if (nj == 1 or nj == 2 or nj == 3 or nj == 4 or nj == 5 or nj == 6 or nj == 7 or nj == 8):
                    continue
                
                if m == 1 and mj == 0:
                    with open(train_img_txt_file,'a') as f:
                        img_line = os.path.join(img_dir, imgs_list[i]) + " " + os.path.join(img_dir, imgs_list[j]) + "\n"
                        print(f"train images {img_line}")
                        f.write(img_line)
                    with open(train_seg_txt_file,'a') as f:
                        seg_line = os.path.join(seg_dir, imgs_list[i]) + " " + os.path.join(seg_dir, imgs_list[j]) + "\n"
                        print(f"train seg {seg_line}")
                        f.write(seg_line)
                        
                        
def OAIZIB_img2txt(root_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file):
    trimg_dir = os.path.join(root_dir, "oai_zib_mri_train", "train")
    # seg_dir = os.path.join(root_dir, "masksTr")
    trseg_dir = os.path.join(root_dir, "oai_zib_labelmaps", "labelmaps", "train")
    trimgs_list = os.listdir(trimg_dir)
    # trimgs_list.sort()
    print(trimgs_list)
    
    for i in range(1, len(trimgs_list)):
        with open(train_img_txt_file,'a') as f:
            img_line = os.path.join(trimg_dir, trimgs_list[i]) + " " + os.path.join(trimg_dir, trimgs_list[i-1]) + "\n"
            print(f"train images {img_line}")
            f.write(img_line)
        with open(train_seg_txt_file,'a') as f:
            seg_line = os.path.join(trseg_dir, trimgs_list[i]) + " " + os.path.join(trseg_dir, trimgs_list[i-1]) + "\n"
            print(f"train seg {seg_line}")
            f.write(seg_line)
                
                
    tsimg_dir = os.path.join(root_dir, "oai_zib_mri_test", "test")
    # seg_dir = os.path.join(root_dir, "masksTr")
    tsseg_dir = os.path.join(root_dir, "oai_zib_labelmaps", "labelmaps", "test")
    tsimgs_list = os.listdir(tsimg_dir)
    # trimgs_list.sort()
    print(tsimgs_list)
    
    for i in range(1, len(tsimgs_list)):
        with open(val_img_txt_file,'a') as f:
            img_line = os.path.join(tsimg_dir, tsimgs_list[i]) + " " + os.path.join(tsimg_dir, tsimgs_list[i-1]) + "\n"
            print(f"test images {img_line}")
            f.write(img_line)
        with open(val_seg_txt_file,'a') as f:
            seg_line = os.path.join(tsseg_dir, tsimgs_list[i]) + " " + os.path.join(tsseg_dir, tsimgs_list[i-1]) + "\n"
            print(f"test seg {seg_line}")
            f.write(seg_line)
        with open("/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/test_img_seg_list.txt",'a') as f:
            mov_line = os.path.join(tsimg_dir, tsimgs_list[i]) + " " + os.path.join(tsseg_dir, tsimgs_list[i])
            fix_line = os.path.join(tsimg_dir, tsimgs_list[i-1]) + " " + os.path.join(tsseg_dir, tsimgs_list[i-1])
            line = mov_line + " " + fix_line + "\n"
            print(f"test line {line}")
            f.write(line)
            
            
def combinationtxt(train_img_txt_file, train_seg_txt_file, train_img_seg_txt_file):
    with open(train_img_txt_file, 'r') as file:
        imgs = file.readlines()
    imglist = [x.strip() for x in imgs if x.strip()]
    
    with open(train_seg_txt_file, 'r') as file:
        segs = file.readlines()
    seglist = [x.strip() for x in segs if x.strip()]
    
    for i in range(len(imglist)):
        # print(f"imgs: {imglist[i]} segs: {seglist[i]}")
        mov, fix = imglist[i].split(" ")
        mov_seg, fix_seg = seglist[i].split(" ")
        print(f"mov: {mov} mov_seg: {mov_seg} fix: {fix} fix_seg: {fix_seg}")
    
        with open(train_img_seg_txt_file,'a') as f:
            line = mov + " " + mov_seg + " " + fix + " " + fix_seg + "\n"
            print(f"train file {line}")
            f.write(line)

                    
if __name__ == "__main__":
    # json_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train.json"
    # txt_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train_list.txt"
    # img_dir = "/mnt/lhz/Datasets/Learn2reg/LPBA40"
    # txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/LPBA/test_img_seg_list.txt"
    
    # img_dir = "/mnt/lhz/Datasets/Learn2reg/OASIS"
    # img_dir = "/mnt/lhz/Datasets/Learn2reg/OAI-ZIB"
    # img_dir = "/mnt/lhz/Datasets/Learn2reg/AbdomenCTCT"
    
    # train_img_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/train_img_list.txt"
    # train_seg_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/train_seg_list.txt"
    # val_img_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/val_img_list.txt"
    # val_seg_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/val_seg_list.txt"
    # test_txt_file = "/mnt/lhz/Github/Image_registration/voxelmorph/images/OAIZIB/test_img_seg_list.txt"
    train_img_txt_file = "/mnt/lhz/Github/Image_registration/RDP/images/OAIZIB/train_img_list.txt"
    train_seg_txt_file = "/mnt/lhz/Github/Image_registration/RDP/images/OAIZIB/train_seg_list.txt"
    train_img_seg_txt_file = "/mnt/lhz/Github/Image_registration/RDP/images/OAIZIB/train_img_seg_list.txt"
    
    # OASIS_img2txt1(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    # OASIS_img2txt2(img_dir, test_txt_file)
    # OASIS_img2txt3(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    # ACTMR_img2txt(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    # ACTCT_img2txt(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    # OAIZIB_img2txt(img_dir, train_img_txt_file, train_seg_txt_file, val_img_txt_file, val_seg_txt_file)
    combinationtxt(train_img_txt_file, train_seg_txt_file, train_img_seg_txt_file)