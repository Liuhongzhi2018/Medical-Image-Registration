#!/usr/bin/env bash

# code="train_nohup.py"
code="DM_train_ACDC_sample_nohup.py"
config_file="config/diffuseMorph_trainval_3D_ACDC.json"

CUDA_VISIBLE_DEVICES=1 python ${code} -c ${config_file}
