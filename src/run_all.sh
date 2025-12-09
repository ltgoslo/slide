#!/bin/sh

python3 evaluate.py --method random
python3 evaluate.py --method gpt2 --model nie3e/gpt2-lang-ident
python3 evaluate.py --method fasttext_hf_hub --model facebook/fasttext-language-identification # NLLB-218
python3 evaluate.py --method fasttext --model FastText-176
python3 evaluate.py --method fasttext_hf_hub --model laurievb/OpenLID
python3 evaluate.py --method fasttext_hf_hub --model cis-lmu/glotlid
python3 evaluate.py --method fasttext_hf_hub --model  NbAiLab/nb-nordic-lid
python3 evaluate.py --method bert --model ltg/SLIDE-base
python3 evaluate.py --method bert --model ltg/SLIDE-small
python3 evaluate.py --method bert --model ltg/SLIDE-x-small
