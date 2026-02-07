for model in laurievb/OpenLID-v2
  do
    python3 evaluate.py --method fasttext_hf_hub --model $model --dataset ../test_data/test_other_2_new.jsonl --threshold 0.5 --run_name vardial_2026/v2-slide
    for dataset in flores_plus_devtest.jsonl udhr.jsonl nordic_dsl_test50k.jsonl
      do
        python3 evaluate.py --method fasttext_hf_hub --model $model --dataset $dataset --threshold 0.5 --run_name vardial_2026/v2-$dataset
      done
  done