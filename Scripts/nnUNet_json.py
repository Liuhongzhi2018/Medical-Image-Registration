import os
import json
import re
import glob
from collections import OrderedDict


def list_sort_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)',s)]
    l.sort(key=alphanum_key)
    return l

def basename(p):
    baselist = []
    for i in range(len(p)):
        img_label_base = os.path.basename(p[i])
        baselist.append(img_label_base)
    return baselist

if __name__ == '__main__':

    # path_originalData = '/mnt/lhz/Github/MedicalDetection/BrainFracSeg/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task06_Ankle21/'
    path_originalData = '/mnt/lhz/Github/SEU_Ankle_MONAI/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task04_ankletest/'
    # train_image = list_sort_nicely(glob.glob(path_originalData+"imagesTr/*"))
    # train_label = list_sort_nicely(glob.glob(path_originalData+"labelsTr/*"))
    # test_image = list_sort_nicely(glob.glob(path_originalData+"imagesTs/*"))
    # test_label = list_sort_nicely(glob.glob(path_originalData+"labelsTs/*"))
    train_image = basename(glob.glob(path_originalData+"imagesTr/*"))
    train_label = basename(glob.glob(path_originalData+"labelsTr/*"))
    test_image = basename(glob.glob(path_originalData+"imagesTs/*"))
    test_label = basename(glob.glob(path_originalData+"labelsTs/*"))
    # print(glob.glob(path_originalData+"imagesTr/*"))
    train_image.sort()
    train_label.sort()
    test_image.sort()
    test_label.sort()


    # json_dict = OrderedDict()
    # json_dict['name'] = "Bone lesions"
    # json_dict['description'] = "Gulou Bone lesions"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT"
    # }
    
    # json_dict = OrderedDict()
    # json_dict['name'] = "Fracture"
    # json_dict['description'] = "HaiNan Cranial Fracture"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT"
    # }

    # json_dict = OrderedDict()
    # json_dict['name'] = "OAR"
    # json_dict['description'] = "MICCAI 2015 Head and Neck Auto Segmentation Challenge"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT"
    # }
    

    # json_dict = OrderedDict()
    # json_dict['name'] = "OAR"
    # json_dict['description'] = "MICCAI 2019 Automatic Structure Segmentation for Radiotherapy Planning Challenge"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT"
    # }
    

    json_dict = OrderedDict()
    json_dict['name'] = "Ankle"
    json_dict['description'] = "first-imaging"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT"
    }
    

    # json_dict = OrderedDict()
    # json_dict['name'] = "Rib"
    # json_dict['description'] = "MICCAI 2020 RibFrac Challenge"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT"
    # }

    # # Gulou Hospital
    # json_dict['labels'] = {
    #     "0": "Background",
    #     "1": "Osteolysis",
    #     "2": "Osteogenesis",
    #     "3": "Boneisland",
    #     "4": "Mixture"
    # }

    # json_dict['labels'] = {
    #     "0": "Background",
    #     "1": "Mandible",
    # }

    # json_dict['labels'] = {
    #     "0": "Background",
    #     "1": "BrainStem",
    #     "2": "Chiasm",
    #     "3": "Mandible",
    #     "4": "OpticNerve_L",
    #     "5": "OpticNerve_R",
    #     "6": "Parotid_L",
    #     "7": "Parotid_R",
    #     "8": "Submandibular_L",
    #     "9": "Submandibular_R"
    # }
    
    
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "BrainStem",
    #     "2": "Eye_L",
    #     "3": "Eye_R",
    #     "4": "Lens_L",
    #     "5": "Lens_R",
    #     "6": "Optical_L",
    #     "7": "Optical_R",
    #     "8": "Optical_chiasma",
    #     "9": "Temporallobes_L",
    #     "10": "Temporallobes_R",
    #     "11": "Pituitary",
    #     "12": "Parotidgland_L",
    #     "13": "Parotidgland_R",
    #     "14": "Innerear_L",
    #     "15": "Innerear_R",
    #     "16": "Middleear_L",
    #     "17": "Middleear_R",
    #     "18": "Temporomandibularjoint_L",
    #     "19": "Temporomandibularjoint_R",
    #     "20": "Spinal_cord",
    #     "21": "Mandible_L",
    #     "22": "Mandible_R",
    # }
    
    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "Ankle",
    # }

    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "Tibia",
    #     "2": "Fibula",
    # }
    
    json_dict['labels'] = {
        "0": "background",
        "1": "Tibia",
        "2": "Fibula",
        "3": "Talus",
    }

    # json_dict = OrderedDict()
    # json_dict['name'] = "Ankle"
    # json_dict['description'] = "first-imaging"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "see challenge website"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "MRI"
    # }

    # json_dict['labels'] = {
    #     "0": "background",
    #     "1": "femur",
    #     "2": "tibia",
    #     "3": "femoral cart",
    #     "4": "medial tibial cartilage",
    #     "5": "lateral tibial cartilage",
    # }


# left eye (100), right eye (100), left lens (50) 晶状体, 
# right lens (50), left optical nerve (80), right optical nerve (80) 视神经, 
# optical chiasma (50) 视神经交叉, pituitary (80) 垂体, brain stem (100) 脑干, 
# left temporal lobes (80), right temporal lobes (80) 颞叶, 
# spinal cord (100) 脊髓, left parotid gland (50), right parotid gland (50) 腮腺, 
# left inner ear (70), right inner ear (70) 内耳, left middle ear (70), right middle ear (70) 中耳, 
# left temporomandibular joint (60), right temporomandibular joint (60) 颞下颌关节, 
# left mandible (100), right mandible (100) 下颌骨. 


    # json_dict['labels'] = {
    #         "0": "Background",
    #         "1": "Displaced",
    #         "2": "Nondisplaced",
    #         "3": "Buckle",
    #         "4": "Segmental",
    #         "5": "Ignore"
    # }

    # json_dict['labels'] = {
    #         "0": "Background",
    #         "1": "Displaced",
    #         "2": "Nondisplaced",
    #         "3": "Buckle",
    #         "4": "Segmental",
    #         "5": "Ignore"
    # }

    # json_dict['labels'] = {
    #         "0": "Background",
    #         "1": "Displaced",
    #         "2": "Nondisplaced",
    #         "3": "Buckle",
    #         "4": "Segmental",
    #         "5": "Ignore"
    # }

    json_dict['numTraining'] = len(train_image)
    json_dict['numTest'] = len(test_image)

    json_dict['training'] = []
    for idx in range(len(train_image)):
        print(train_image[idx], train_label[idx])
        json_dict['training'].append({'image': "./imagesTr/%s" % train_image[idx], "label": "./labelsTr/%s" % train_label[idx]})

    json_dict['test'] = ["./imagesTs/%s" % i for i in test_image]

    with open(os.path.join(path_originalData, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)