#!/bin/sh

for model in ../../OpenLID-v2/new_data/oci_no_pilar_frp/train/openlid-v3.bin
do
  for dataset in ../test_data/test_other_2_new.jsonl flores_plus_devtest.jsonl udhr.jsonl nordic_dsl_test50k.jsonl 
  do
    for threshold in 0.5
    do
      python3 evaluate.py --method fasttext_hf_hub --model $model --dataset $dataset --threshold $threshold
    done
  done
done