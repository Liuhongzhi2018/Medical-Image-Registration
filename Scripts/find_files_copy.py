import os
import shutil

# image_from_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/images_177"
# image_from_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/labels_177"
image_from_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/12156reori3label"
# image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/imagesTr_121"
# image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/imagesTs_56"
# image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/labelsTr_121"
# image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/labelsTs_56"
# image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/labelsTr_121_3c"
image_to_path = "/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v2/reorient_177/labelsTs_56_3c"
# txt_path = '/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v1/trimage.txt'
txt_path = '/mnt/lhz/Datasets/Ankle/Yiying_Ankle_Data_v1/tsimage.txt'
count = 0

if not os.path.exists(image_to_path):
	print("to_dir_path not exist, so create the dir")
	os.makedirs(image_to_path, exist_ok=True)

ftxt = open(txt_path,'r')
img_list = [x.strip() for x in ftxt]
ftxt.close()
# print(label)

for line in img_list:
   image_name = line.split('.')[0] + "_reori.nii.gz"
   # print("line: ",line)
   count +=1
   print("count {} image name {}".format(count, image_name))
   img_from_path = os.path.join(image_from_path, image_name)
   img_to_path = os.path.join(image_to_path, image_name)
   shutil.copy(img_from_path, img_to_path)



# image_from_path = "/mnt/no1/liuhz/Datasets/Brain/HaiNan_0928/VOCdevkit/VOC2007/JPEGImages/"
# image_to_path = "/mnt/no1/liuhz/Github/MedicalDetection/pytorch_classification/data/cal/normal_GT"
# class_num = 2

# txt_path = '/mnt/no1/liuhz/Github/MedicalDetection/pytorch_classification/data/cal/normal.txt'
# count = 0

# if not os.path.exists(image_to_path):
# 	print("to_dir_path not exist, so create the dir")
# 	os.makedirs(image_to_path, exist_ok=True)

# # for i in range(class_num):
# #     subclass_dir = os.path.join(image_to_path, str(i))
# #     print(subclass_dir)
# #     os.makedirs(subclass_dir, exist_ok=True)

# fr = open(txt_path)

# for line in fr.readlines():
#    full_info = line.strip()
#    print(full_info)
#    image_name, label = full_info.split(' ')[0], full_info.split(' ')[1]
#    # image_name = os.path.basename(full_name).split('.')[0] + '.jpg'
#    image_name = os.path.basename(image_name)
#    # print("line: ",line)
#    count +=1
#    print("count {} image name {}".format(count, image_name))
#    img_from_path = os.path.join(image_from_path, image_name)
#    img_to_path = os.path.join(image_to_path, image_name)
#    shutil.copy(img_from_path, img_to_path)

# for line in fr.readlines():
#    full_info = line.strip()
#    print(full_info)
#    image_name, label = full_info.split(' ')[0], full_info.split(' ')[1]
#    # image_name = os.path.basename(full_name).split('.')[0] + '.jpg'
#    image_name = os.path.basename(image_name)
#    # print("line: ",line)
#    count +=1
#    print("count {} image name {}".format(count, image_name))
#    img_from_path = os.path.join(image_from_path, image_name)
#    image_to_dir_path = os.path.join(image_to_path, str(label))
#    img_to_path = os.path.join(image_to_dir_path, image_name)
#    shutil.copy(img_from_path, img_to_path)