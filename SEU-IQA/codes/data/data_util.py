import os
import random
import numpy as np
import cv2
from itertools import combinations
import json

####################
# Files & IO
####################

###################### get PIPAL image combinations ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def all_img_paths(img_root):
    assert os.path.isdir(img_root), '{:s} is not a valid directory'.format(img_root)
    images_paths = {}
    for dirRoot, _, fnames in os.walk(img_root):
        for fname in fnames:
            if is_image_file(fname):
                images_paths[fname] = os.path.join(dirRoot, fname)
    return images_paths


def image_combinations_fIQA(dist_root, mos_root, phase='train', dataset_name='fIQA'):
    if phase == 'train':
        img_list, img_score_dict = [], {}
        df_list = os.listdir(dist_root)
        for df in df_list:
            if os.path.isdir(os.path.join(dist_root, df)):
                # print(f"Directory: {os.path.join(dist_root, df)}")
                # Directory: /home/liuhongzhi/Data/IQA/face_IQA/0
                # Directory: /home/liuhongzhi/Data/IQA/face_IQA/1
                subdir_files = os.listdir(os.path.join(dist_root, df))
                img_list.extend([os.path.join(dist_root, df, s) for s in subdir_files])
            elif 'json' in df:
                # print(f"Json file: {os.path.join(dist_root, df)}")
                with open(os.path.join(dist_root, df), 'r') as f:
                    load_dict = json.load(f)
                    img_score_dict.update(load_dict)
                
        names_dist, dist_scores = [], []
        for i in img_list:
            names_dist.append(i)
            name = os.path.basename(i)
            img_scores = img_score_dict[name]
            # print(f"img {i} scores: {img_scores}")
            cur_img_list = []
            for k, v in img_scores.items():
                if "noise" in k:
                    cur_img_list.append(v)
                elif "blur" in k:
                    cur_img_list.append(v)
                elif "color" in k:
                    cur_img_list.append(v)
                elif "contrast" in k:
                    cur_img_list.append(v)
                else:
                    continue
            # print(f"names_dist: {names_dist} \ndist_scores: {dist_scores}")
            dist_scores.append(cur_img_list)
                    
        return names_dist, dist_scores
    
        # Name: ['/home/liuhongzhi/Data/IQA/face_IQA/0/0-0-0811.png', 
        #        '/home/liuhongzhi/Data/IQA/face_IQA/0/0-0-0469.png', 
        #        '/home/liuhongzhi/Data/IQA/face_IQA/0/0-0-0720.png', 
        #        '/home/liuhongzhi/Data/IQA/face_IQA/0/6-0-0941.png',]
        
        # Scores: [[3.7205151507439096, 4.198845604445939, 1.0299286842346191, 4.456061363220215], 
        # [4.567188048893066, 3.9981981809827816, 4.441505432128906, 1.2082160711288452], 
        # [4.24913136039793, 4.2635159164278855, 2.1098692417144775, 3.006633758544922]]
        
    elif phase == 'test':
        '''Prepare Testing or Validation Image Name List'''
        if dataset_name != 'fIQA':
            raise NotImplementedError('Nor fIQA. Please check dataset name in configuration file.')
        img_list = []
        count = 0
        df_list = os.listdir(dist_root)
        for df in df_list:
            if os.path.isdir(os.path.join(dist_root, df)):
                print(f"Directory: {os.path.join(dist_root, df)}")
                # Directory: /home/liuhongzhi/Data/IQA/face_IQA/0
                # Directory: /home/liuhongzhi/Data/IQA/face_IQA/1
                subdir_files = os.listdir(os.path.join(dist_root, df))
                img_list.extend([os.path.join(dist_root, df, s) for s in subdir_files])
                print(f"img_list: {img_list} {len(img_list)}")
        # for i in img_list:
        #     count += 1
        #     print(f"{count} {i}")
        #     names_dist.append(i)
            
        return img_list
    else:
        raise NotImplementedError("Error: Wrong Phase ! Onlhy Train and Test")


def image_combinations(ref_root, dist_root, mos_root, phase='train', dataset_name='PIPAL'):
    if phase == 'train':
        img_extension = '.bmp'
        ''' Form Combinations :  [ [ref_path, dist_A_path, dist_B_path, real_pro ], [..], ....] '''
        assert os.path.isdir(ref_root) is True, '{} is not a valid directory'.format(ref_root)
        assert os.path.isdir(dist_root) is True, '{} is not a valid directory'.format(dist_root)
        assert os.path.isdir(mos_root) is True, '{} is not a valid directory'.format(mos_root)

        '''obtain name of ref'''
        ref_fnames = list(map(lambda x: x if is_image_file(x) else print('Ignore {}'.format(x)), sorted(list(map(lambda x: x, os.walk(ref_root)))[0][2])))
        # print(f"Phase: {phase} ref_root: {ref_root} ref_fnames: {ref_fnames}")

        '''obtain paths of distortions. dict contains 200 list, every list contains different distorted image of one reference '''
        dist_class_Ref = {ref_name: [] for ref_name in ref_fnames}
        for root, _, fnames in os.walk(dist_root):
            fnames = [fname for fname in fnames if is_image_file(fname)]
            for fname in sorted(fnames):
                ref_name = fname.split("_")[0] + img_extension
                dist_class_Ref[ref_name].append(fname)
        # print(f"Phase: {phase} dist_class_Ref: {dist_class_Ref}")

        '''obtain MOS score of every distortion img'''
        mos_dict = {}
        mos_fnames = [fname for fname in os.listdir(mos_root) if ".txt" in fname]
        for fname in mos_fnames:
            mos_path = os.path.join(mos_root, fname)
            with open(mos_path, "r") as f_ELO:
                lines = f_ELO.readlines()
                splited_lines = [dist_score.split(',') for dist_score in lines]
                for DisName_ELO in splited_lines:
                    # print(f"DisName_ELO: {DisName_ELO}")
                    DisName, dist_ELO = DisName_ELO[0], float(DisName_ELO[1][:-1])
                    mos_dict[DisName] = dist_ELO
        # print(f"Phase: {phase} mos_dict: {mos_dict}")

        ''' obtain [dist_A, dist_B, ref, real_pro] '''
        pair_combinations = {ref_name: [] for ref_name in ref_fnames}
        for ref_fname, dist_paths in dist_class_Ref.items():
            # print(f"*** dist_class_Ref *** ref_fname: {ref_fname} dist_paths: {dist_paths}")
            distAB_combinations = [list(dist_AB) for dist_AB in combinations(dist_paths, 2)] # length=2
            for index, dist_AB in enumerate(distAB_combinations):
                # print(f"*** distAB_combinations *** index: {index} dist_AB: {dist_AB}")
                # *** distAB_combinations *** index: 3600 dist_AB: ['A0002_02_09.bmp', 'A0002_03_11.bmp']
                # Obtain [ dist_A, dist_B, ref ]
                dist_AB.append(dist_AB[0].split("_")[0] + img_extension)
                # print(f"dist_AB append: {dist_AB}")
                # dist_AB append: ['A0002_02_09.bmp', 'A0002_03_11.bmp', 'A0002.bmp']
                # Obtain [ dist_A, dist_B, ref, real_pro ]
                if dist_AB[0] in mos_dict and dist_AB[1] in mos_dict:
                    dist_A_score = mos_dict[dist_AB[0]]
                    dist_B_score = mos_dict[dist_AB[1]]
                    dist_AB.append(dist_A_score)
                    dist_AB.append(dist_B_score)
                else:
                    print(index, dist_AB)
                    raise NotImplementedError("There is Distorted Image that does not have MOS scores in Your MOS file!")
            pair_combinations[ref_fname] = distAB_combinations
            # print(f"Phase: {phase} pair_combinations ref_fname: {ref_fname} distAB_combinations: {distAB_combinations}")
        # print(f"Phase: {phase} pair_combinations: {pair_combinations}")

        names_ref, names_dist_A, names_dist_B, dist_A_scores, dist_B_scores = [], [], [], [], []
        for _, pairs in pair_combinations.items():
            # print(f"*** pairs: {pairs}")
            for pair in pairs:
                # print(f"** pair: {pair}")
                # ** pair: ['A0001_00_00.bmp', 'A0001_00_01.bmp', 'A0001.bmp', 1520.0648, 1437.0798]
                names_dist_A.append(pair[0])
                names_dist_B.append(pair[1])
                names_ref.append(pair[2])
                dist_A_scores.append(pair[3])
                dist_B_scores.append(pair[4])
        return names_ref, names_dist_A, names_dist_B, dist_A_scores, dist_B_scores

    elif phase == 'test':
        '''Prepare Testing or Validation Image Name List'''
        if dataset_name != 'PIPAL' and "TID2013":
            raise NotImplementedError('Nor PIPAL or TID2013. Please check dataset name in configuration file.')
        ref_fnames, dist_fnames = [], []
        for dirpath, _, fnames in os.walk(dist_root):
            for fname in sorted(fnames):
                # def is_image_file(filename)
                if is_image_file(fname):
                    img_extension = fname.split(".")[-1]
                    fname_ = fname.split("_")[0] + '.' + img_extension
                    # print(f"fname: {fname} fname_: {fname_}")
                    # fname: A0185_00_00.bmp fname_: A0185.bmp
                    # fname: A0185_00_01.bmp fname_: A0185.bmp
                    # ref_fnames.append(fname.split("_")[0] + '.' + img_extension)
                    ref_fnames.append(fname_)
                    dist_fnames.append(fname)
        # print(f"ref_fnames {ref_fnames} \ndist_fnames {dist_fnames}")
        return ref_fnames, dist_fnames
    else:
        raise NotImplementedError("Error: Wrong Phase ! Onlhy Train and Test")


# ref_root: /home/liuhongzhi/Data/PIPAL/training_part/Reference_train 
# ref_fnames: ['A0001.bmp', 'A0002.bmp', 'A0003.bmp', 'A0006.bmp', 'A0007.bmp', 'A0008.bmp', 'A0009.bmp', 'A0010.bmp',
# 'A0012.bmp', 'A0013.bmp', 'A0014.bmp', 'A0015.bmp', 'A0016.bmp', 'A0017.bmp', 'A0018.bmp', 'A0021.bmp', 'A0023.bmp',
# 'A0024.bmp', 'A0025.bmp', 'A0026.bmp', 'A0027.bmp', 'A0028.bmp', 'A0029.bmp', 'A0030.bmp', 'A0031.bmp', 'A0032.bmp',
# 'A0034.bmp', 'A0035.bmp', 'A0037.bmp', 'A0038.bmp', 'A0039.bmp', 'A0040.bmp', 'A0041.bmp', 'A0042.bmp', 'A0043.bmp',
# 'A0044.bmp', 'A0045.bmp', 'A0046.bmp', 'A0047.bmp', 'A0048.bmp', 'A0049.bmp', 'A0051.bmp', 'A0052.bmp', 'A0053.bmp',
# 'A0054.bmp', 'A0055.bmp', 'A0056.bmp', 'A0057.bmp', 'A0058.bmp', 'A0059.bmp', 'A0060.bmp', 'A0063.bmp', 'A0064.bmp',
# 'A0065.bmp', 'A0066.bmp', 'A0067.bmp', 'A0068.bmp', 'A0069.bmp', 'A0070.bmp', 'A0072.bmp', 'A0073.bmp', 'A0075.bmp',
# 'A0077.bmp', 'A0078.bmp', 'A0079.bmp', 'A0080.bmp', 'A0081.bmp', 'A0082.bmp', 'A0083.bmp', 'A0084.bmp', 'A0085.bmp',
# 'A0086.bmp', 'A0087.bmp', 'A0088.bmp', 'A0090.bmp', 'A0092.bmp', 'A0093.bmp', 'A0094.bmp', 'A0096.bmp', 'A0097.bmp',
# 'A0098.bmp', 'A0099.bmp', 'A0100.bmp', 'A0101.bmp', 'A0102.bmp', 'A0103.bmp', 'A0104.bmp', 'A0105.bmp', 'A0106.bmp',
# 'A0107.bmp', 'A0108.bmp', 'A0110.bmp', 'A0112.bmp', 'A0113.bmp', 'A0114.bmp', 'A0115.bmp', 'A0116.bmp', 'A0118.bmp',
# 'A0119.bmp', 'A0120.bmp', 'A0121.bmp', 'A0123.bmp', 'A0124.bmp', 'A0130.bmp', 'A0131.bmp', 'A0132.bmp', 'A0133.bmp', 
# 'A0134.bmp', 'A0135.bmp', 'A0137.bmp', 'A0138.bmp', 'A0139.bmp', 'A0140.bmp', 'A0141.bmp', 'A0142.bmp', 'A0143.bmp',
# 'A0145.bmp', 'A0146.bmp', 'A0148.bmp', 'A0149.bmp', 'A0150.bmp', 'A0151.bmp', 'A0152.bmp', 'A0153.bmp', 'A0155.bmp',
# 'A0156.bmp', 'A0157.bmp', 'A0158.bmp', 'A0159.bmp', 'A0160.bmp', 'A0162.bmp', 'A0163.bmp', 'A0164.bmp', 'A0165.bmp',
# 'A0166.bmp', 'A0167.bmp', 'A0170.bmp', 'A0171.bmp', 'A0172.bmp', 'A0174.bmp', 'A0175.bmp', 'A0176.bmp', 'A0177.bmp',
# 'A0178.bmp', 'A0179.bmp', 'A0180.bmp', 'A0181.bmp', 'A0182.bmp', 'A0183.bmp', 'A0184.bmp']

# Phase: train dist_class_Ref: {'A0001.bmp': ['A0001_00_00.bmp', 'A0001_00_01.bmp', 'A0001_00_02.bmp', 'A0001_00_03.bmp', 
# 'A0001_00_04.bmp', 'A0001_00_05.bmp', 'A0001_00_06.bmp', 'A0001_00_07.bmp', 'A0001_00_08.bmp', 'A0001_00_09.bmp', 
# 'A0001_00_10.bmp', 'A0001_00_11.bmp', 'A0001_01_00.bmp', 'A0001_01_01.bmp', 'A0001_01_02.bmp', 'A0001_01_03.bmp',
# 'A0001_01_04.bmp', 'A0001_01_05.bmp', 'A0001_01_06.bmp', 'A0001_01_07.bmp', 'A0001_01_08.bmp', 'A0001_01_09.bmp', 
# 'A0001_01_10.bmp', 'A0001_01_11.bmp', 'A0001_01_12.bmp', 'A0001_01_13.bmp', 'A0001_01_14.bmp', 'A0001_01_15.bmp', 
# 'A0001_02_00.bmp', 'A0001_02_01.bmp', 'A0001_02_02.bmp', 'A0001_02_03.bmp', 'A0001_02_04.bmp', 'A0001_02_05.bmp', 
# 'A0001_02_06.bmp', 'A0001_02_07.bmp', 'A0001_02_08.bmp', 'A0001_02_09.bmp', 'A0001_03_00.bmp', 'A0001_03_01.bmp', 
# 'A0001_03_02.bmp', 'A0001_03_03.bmp', 'A0001_03_04.bmp', 'A0001_03_05.bmp', 'A0001_03_06.bmp', 'A0001_03_07.bmp', 
# 'A0001_03_08.bmp', 'A0001_03_09.bmp', 'A0001_03_10.bmp', 'A0001_03_11.bmp', 'A0001_03_12.bmp', 'A0001_03_13.bmp', 
# 'A0001_03_14.bmp', 'A0001_03_15.bmp', 'A0001_03_16.bmp', 'A0001_03_17.bmp', 'A0001_03_18.bmp', 'A0001_03_19.bmp', 
# 'A0001_03_20.bmp', 'A0001_03_21.bmp', 'A0001_03_22.bmp', 'A0001_03_23.bmp', 'A0001_04_00.bmp', 'A0001_04_01.bmp', 
# 'A0001_04_02.bmp', 'A0001_04_03.bmp', 'A0001_04_04.bmp', 'A0001_04_05.bmp', 'A0001_04_06.bmp', 'A0001_04_07.bmp', 
# 'A0001_04_08.bmp', 'A0001_04_09.bmp', 'A0001_04_10.bmp', 'A0001_04_11.bmp', 'A0001_04_12.bmp', 'A0001_05_00.bmp', 
# 'A0001_05_01.bmp', 'A0001_05_02.bmp', 'A0001_05_03.bmp', 'A0001_05_04.bmp', 'A0001_05_05.bmp', 'A0001_05_06.bmp', 
# 'A0001_05_07.bmp', 'A0001_05_08.bmp', 'A0001_05_09.bmp', 'A0001_05_10.bmp', 'A0001_05_11.bmp', 'A0001_05_12.bmp', 
# 'A0001_05_13.bmp', 'A0001_06_00.bmp', 'A0001_06_01.bmp', 'A0001_06_02.bmp', 'A0001_06_03.bmp', 'A0001_06_04.bmp', 
# 'A0001_06_05.bmp', 'A0001_06_06.bmp', 'A0001_06_07.bmp', 'A0001_06_08.bmp', 'A0001_06_09.bmp', 'A0001_06_10.bmp', 
# 'A0001_06_11.bmp', 'A0001_06_12.bmp', 'A0001_06_13.bmp', 'A0001_06_14.bmp', 'A0001_06_15.bmp', 'A0001_06_16.bmp', 
# 'A0001_06_17.bmp', 'A0001_06_18.bmp', 'A0001_06_19.bmp', 'A0001_06_20.bmp', 'A0001_06_21.bmp', 'A0001_06_22.bmp', 
# 'A0001_06_23.bmp', 'A0001_06_24.bmp', 'A0001_06_25.bmp', 'A0001_06_26.bmp'], 
# 'A0002.bmp': ['A0002_00_00.bmp', 'A0002_00_01.bmp', 'A0002_00_02.bmp', 'A0002_00_03.bmp', 'A0002_00_04.bmp', 
# 'A0002_00_05.bmp', 'A0002_00_06.bmp', 'A0002_00_07.bmp', 'A0002_00_08.bmp', 'A0002_00_09.bmp', 'A0002_00_10.bmp', 
# 'A0002_00_11.bmp', 'A0002_01_00.bmp', 'A0002_01_01.bmp', 'A0002_01_02.bmp', 'A0002_01_03.bmp', 'A0002_01_04.bmp', 
# 'A0002_01_05.bmp', 'A0002_01_06.bmp', 'A0002_01_07.bmp', 'A0002_01_08.bmp', 'A0002_01_09.bmp', 'A0002_01_10.bmp', 
# 'A0002_01_11.bmp', 'A0002_01_12.bmp', 'A0002_01_13.bmp', 'A0002_01_14.bmp', 'A0002_01_15.bmp', 'A0002_02_00.bmp', 
# 'A0002_02_01.bmp', 'A0002_02_02.bmp', 'A0002_02_03.bmp', 'A0002_02_04.bmp', 'A0002_02_05.bmp', 'A0002_02_06.bmp', 
# 'A0002_02_07.bmp', 'A0002_02_08.bmp', 'A0002_02_09.bmp', 'A0002_03_00.bmp', 'A0002_03_01.bmp', 'A0002_03_02.bmp', 
# 'A0002_03_03.bmp', 'A0002_03_04.bmp', 'A0002_03_05.bmp', 'A0002_03_06.bmp', 'A0002_03_07.bmp', 'A0002_03_08.bmp', 
# 'A0002_03_09.bmp', 'A0002_03_10.bmp', 'A0002_03_11.bmp', 'A0002_03_12.bmp', 'A0002_03_13.bmp', 'A0002_03_14.bmp', 
# 'A0002_03_15.bmp', 'A0002_03_16.bmp', 'A0002_03_17.bmp', 'A0002_03_18.bmp', 'A0002_03_19.bmp', 'A0002_03_20.bmp', 
# 'A0002_03_21.bmp', 'A0002_03_22.bmp', 'A0002_03_23.bmp', 'A0002_04_00.bmp', 'A0002_04_01.bmp', 'A0002_04_02.bmp', 
# 'A0002_04_03.bmp', 'A0002_04_04.bmp', 'A0002_04_05.bmp', 'A0002_04_06.bmp', 'A0002_04_07.bmp', 'A0002_04_08.bmp', 
# 'A0002_04_09.bmp', 'A0002_04_10.bmp', 'A0002_04_11.bmp', 'A0002_04_12.bmp', 'A0002_05_00.bmp', 'A0002_05_01.bmp', 
# 'A0002_05_02.bmp', 'A0002_05_03.bmp', 'A0002_05_04.bmp', 'A0002_05_05.bmp', 'A0002_05_06.bmp', 'A0002_05_07.bmp', 
# 'A0002_05_08.bmp', 'A0002_05_09.bmp', 'A0002_05_10.bmp', 'A0002_05_11.bmp', 'A0002_05_12.bmp', 'A0002_05_13.bmp', 
# 'A0002_06_00.bmp', 'A0002_06_01.bmp', 'A0002_06_02.bmp', 'A0002_06_03.bmp', 'A0002_06_04.bmp', 'A0002_06_05.bmp', 
# 'A0002_06_06.bmp', 'A0002_06_07.bmp', 'A0002_06_08.bmp', 'A0002_06_09.bmp', 'A0002_06_10.bmp', 'A0002_06_11.bmp', 
# 'A0002_06_12.bmp', 'A0002_06_13.bmp', 'A0002_06_14.bmp', 'A0002_06_15.bmp', 'A0002_06_16.bmp', 'A0002_06_17.bmp', 
# 'A0002_06_18.bmp', 'A0002_06_19.bmp', 'A0002_06_20.bmp', 'A0002_06_21.bmp', 'A0002_06_22.bmp', 'A0002_06_23.bmp', 
# 'A0002_06_24.bmp', 'A0002_06_25.bmp', 'A0002_06_26.bmp'],

# Phase: train mos_dict: {'A0107_00_00.bmp': 1585.9077, 'A0107_00_01.bmp': 1400.3891, 'A0107_00_02.bmp': 1661.6359, 
# 'A0107_00_03.bmp': 1432.3772, 'A0107_00_04.bmp': 1309.807, 'A0107_00_05.bmp': 1632.4123, 'A0107_00_06.bmp': 1471.8182,
# 'A0107_00_07.bmp': 1372.5693, 'A0107_00_08.bmp': 1554.0999, 'A0107_00_09.bmp': 1553.4894, 'A0107_00_10.bmp': 1364.1014, 
# 'A0107_00_11.bmp': 1414.5141, 'A0107_01_00.bmp': 1635.8818, 'A0107_01_01.bmp': 1407.0329, 'A0107_01_02.bmp': 1245.414, 
# 'A0107_01_03.bmp': 1555.4967, 'A0107_01_04.bmp': 1479.2454, 'A0107_01_05.bmp': 1313.6918, 'A0107_01_06.bmp': 1610.2639, 
# 'A0107_01_07.bmp': 1412.7482, 'A0107_01_08.bmp': 1326.3566, 'A0107_01_09.bmp': 1699.95, 'A0107_01_10.bmp': 1636.939, 
# 'A0107_01_11.bmp': 1500.4539, 'A0107_01_12.bmp': 1620.4251, 'A0107_01_13.bmp': 1613.0669, 'A0107_01_14.bmp': 1503.2346, 
# 'A0107_01_15.bmp': 1087.7429, 'A0107_02_00.bmp': 1520.3875, 'A0107_02_01.bmp': 1556.6221, 'A0107_02_02.bmp': 1694.0672, 
# 'A0107_02_03.bmp': 1748.861, 'A0107_02_04.bmp': 1795.7083, 'A0107_02_05.bmp': 1343.3179, 'A0107_02_06.bmp': 1381.0134, 
# 'A0107_02_07.bmp': 1421.8885, 'A0107_02_08.bmp': 1413.7453, 'A0107_02_09.bmp': 1427.0773, 'A0107_03_00.bmp': 1383.5353, 
# 'A0107_03_01.bmp': 1391.3711, 'A0107_03_02.bmp': 1284.7899, 'A0107_03_03.bmp': 1499.6644, 'A0107_03_04.bmp': 1493.7694, 
# 'A0107_03_05.bmp': 1540.8533, 'A0107_03_06.bmp': 1596.7678, 'A0107_03_07.bmp': 1341.3678, 'A0107_03_08.bmp': 1316.6401, 
# 'A0107_03_09.bmp': 1377.8564, 'A0107_03_10.bmp': 1394.353, 'A0107_03_11.bmp': 1428.0783, 'A0107_03_12.bmp': 1330.1954, 
# 'A0107_03_13.bmp': 1492.9282, 'A0107_03_14.bmp': 1467.356, 'A0107_03_15.bmp': 1451.4164, 'A0107_03_16.bmp': 1385.0968, 
# 'A0107_03_17.bmp': 1457.2894, 'A0107_03_18.bmp': 1371.7681, 'A0107_03_19.bmp': 1372.9655, 'A0107_03_20.bmp': 1348.7364,
# 'A0107_03_21.bmp': 1210.5582, 'A0107_03_22.bmp': 1226.295, 'A0107_03_23.bmp': 1221.6987, 'A0107_04_00.bmp': 1563.9225, 
# 'A0107_04_01.bmp': 1451.4339, 'A0107_04_02.bmp': 1674.2518, 'A0107_04_03.bmp': 1486.0391, 'A0107_04_04.bmp': 1328.2661, 
# 'A0107_04_05.bmp': 1617.937, 'A0107_04_06.bmp': 1426.3544, 'A0107_04_07.bmp': 1191.4287, 'A0107_04_08.bmp': 1628.1255, 
# 'A0107_04_09.bmp': 1406.1167, 'A0107_04_10.bmp': 1158.5803, 'A0107_04_11.bmp': 1494.8648, 'A0107_04_12.bmp': 1587.5735, 
# 'A0107_05_00.bmp': 1400.6473, 'A0107_05_01.bmp': 1241.4506, 'A0107_05_02.bmp': 1171.4595, 'A0107_05_03.bmp': 1002.8377, 
# 'A0107_05_04.bmp': 1590.9584, 'A0107_05_05.bmp': 1300.8142, 'A0107_05_06.bmp': 1183.425, 'A0107_05_07.bmp': 1195.2387, 
# 'A0107_05_08.bmp': 1104.2305, 'A0107_05_09.bmp': 1340.1489, 'A0107_05_10.bmp': 1527.7426, 'A0107_05_11.bmp': 1427.8779, 
# 'A0107_05_12.bmp': 1536.2396, 'A0107_05_13.bmp': 1337.7467, 'A0107_06_00.bmp': 1480.8802, 'A0107_06_01.bmp': 1478.8462, 
# 'A0107_06_02.bmp': 1350.9153, 'A0107_06_03.bmp': 1445.7128, 'A0107_06_04.bmp': 1442.6787, 'A0107_06_05.bmp': 1545.233, 
# 'A0107_06_06.bmp': 1613.2866, 'A0107_06_07.bmp': 1619.2095, 'A0107_06_08.bmp': 1667.2979, 'A0107_06_09.bmp': 1579.3367, 
# 'A0107_06_10.bmp': 1568.6644, 'A0107_06_11.bmp': 1401.9193, 'A0107_06_12.bmp': 1336.1175, 'A0107_06_13.bmp': 1305.701, 
# 'A0107_06_14.bmp': 1368.4856, 'A0107_06_15.bmp': 1185.9441, 'A0107_06_16.bmp': 1542.7033, 'A0107_06_17.bmp': 1652.1301, 
# 'A0107_06_18.bmp': 1640.3612, 'A0107_06_19.bmp': 1798.4687, 'A0107_06_20.bmp': 1475.3171, 'A0107_06_21.bmp': 1443.5048, 
# 'A0107_06_22.bmp': 1411.4883, 'A0107_06_23.bmp': 1525.6371, 'A0107_06_24.bmp': 1510.398, 'A0107_06_25.bmp': 1495.5525, 
# 'A0107_06_26.bmp': 1469.0858, 'A0100_00_00.bmp': 1541.7301, 'A0100_00_01.bmp': 1310.186,

# Phase: train pair_combinations: A0002.bmp distAB_combinations: [
    # ['A0002_00_00.bmp', 'A0002_00_01.bmp', 'A0002.bmp', 1466.4372, 1292.0443], 
    # ['A0002_00_00.bmp', 'A0002_00_02.bmp', 'A0002.bmp', 1466.4372, 1597.4977], 
    # ['A0002_00_00.bmp', 'A0002_00_03.bmp', 'A0002.bmp', 1466.4372, 1464.543], 
    # ['A0002_00_00.bmp', 'A0002_00_04.bmp', 'A0002.bmp', 1466.4372, 1292.3154], 
    # ['A0002_00_00.bmp', 'A0002_00_05.bmp', 'A0002.bmp', 1466.4372, 1639.5436], 
    # ['A0002_00_00.bmp', 'A0002_00_06.bmp', 'A0002.bmp', 1466.4372, 1539.0781], 
    # ['A0002_00_00.bmp', 'A0002_00_07.bmp', 'A0002.bmp', 1466.4372, 1366.1633],
    
# ref_fnames ['A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 
#             'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0185.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 
#             'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', 'A0186.bmp', ]

# dist_fnames ['A0185_00_00.bmp', 'A0185_00_01.bmp', 'A0185_00_02.bmp', 'A0185_00_03.bmp', 'A0185_00_04.bmp', 'A0185_00_05.bmp', 
# 'A0185_00_06.bmp', 'A0185_00_07.bmp', 'A0185_00_08.bmp', 'A0185_00_09.bmp']

###################### get BAPPS 2AFC image combinations ######################
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy', ]


def is_image_file(filename, mode='img'):
    if (mode == 'img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif (mode == 'np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dirs, mode='img'):
    if (not isinstance(dirs, list)):
        dirs = [dirs, ]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

###################### read images ######################
def read_img(path, size=None):
    '''
    read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]
    '''
    # resize to 256 x 256


    # read image
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if size and img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


###### img augument  ##############
def translate_img(img, max_shift=3.5):
    height, width = img.shape[:2]
    x_shift = random.uniform(-max_shift, max_shift)
    y_shift = random.uniform(-max_shift, max_shift)
    matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    trans_img = cv2.warpAffine(img, matrix, (width, height))
    return trans_img


####################
# image processing
# process on numpy image
####################

def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)



