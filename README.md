# SLIDE

[![arXiv](https://img.shields.io/badge/arXiv-2502.06692-b31b1b.svg)](http://arxiv.org/abs/2502.06692)

Data and code (still being added) for the paper Multi-label Scandinavian Language Identification (SLIDE) (presented at [RESOURCEFUL-2025](https://resourceful-workshop.github.io/resourceful-2025/papers.html)).

##### [Models on HuggingFace](https://huggingface.co/collections/ltg/slide-67d4538eac9736a6f068bf2a) and [how to use them](src/usage_example.py)

##### reproduce metrics (table 4):

```commandline
cd src/
./run_all.sh
```

##### reproduce evaluation on  (table 5)

obtain data from [nordic_langid on Huggingface](https://huggingface.co/datasets/strombergnlp/nordic_langid/tree/main) and put *test.csv into `src/evaluation`
```commandline
cd src/
python3 nordic_langid2jsonl.py 
--method bert --model ltg/SLIDE-base --dataset nordic_dsl_test50k.jsonl
--method bert --model ltg/SLIDE-base --dataset nordic_dsl_test10k.jsonl
```

The values that will be shown will be lower than those in table 5.
The values in table 5 were obtained with the understanding of loose accuracy as it is described in the paper.
The actual evaluate.py accepts a prediction if it is a subset of gold languages, not an intersection. (Values in table 4 were obtained with this understanding).
Sorry for that.