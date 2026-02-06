for dataset in ../test_data/test_other_2_new.jsonl flores_plus_devtest.jsonl udhr.jsonl nordic_dsl_test50k.jsonl
  do
    python3 evaluate.py --method fasttext_hf_hub --dataset $dataset --threshold 0.0
  done