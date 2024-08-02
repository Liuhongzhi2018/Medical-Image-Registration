#!/usr/bin/env bash

code="./scripts/torch/train_ACDC.py"
data_file="./images/ACDC/train_list.txt"
model_dir="./models"
gpu="1"

python ${code} --img-list ${data_file} --model-dir ${model_dir} --gpu ${gpu}