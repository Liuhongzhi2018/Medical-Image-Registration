#!/usr/bin/env bash

code="train_fIQA.py"
config_file="options/train_test_yml/train_IQA_fIQA_part.yml"

CUDA_VISIBLE_DEVICES=1 python ${code} -opt ${config_file}