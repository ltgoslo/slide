#!/bin/sh

for dataset in ../test_data/test_other_2_new.jsonl flores_plus_devtest.jsonl udhr.jsonl nordic_dsl_test50k.jsonl
  do
    python3 evaluate.py --method fasttext_ensemble --dataset $dataset --k 1
  done