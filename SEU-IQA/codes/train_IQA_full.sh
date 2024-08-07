#!/usr/bin/env bash

code="train_PIPAL.py"
config_file="options/train_test_yml/train_IQA_PIPAL_full.yml"

CUDA_VISIBLE_DEVICES=1 python ${code} -opt ${config_file}