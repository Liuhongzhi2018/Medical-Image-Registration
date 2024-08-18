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

if __name__ == "__main__":
    # json_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train.json"
    # txt_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/train_list.txt"
    json_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/test.json"
    txt_file = "/home/liuhongzhi/Method/Registration/voxelmorph/images/ACDC/test_list.txt"
    json2txt(json_file, txt_file)