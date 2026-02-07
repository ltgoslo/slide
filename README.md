# SLIDE

Data and code (still being added) for the paper Multi-label Scandinavian Language Identification (SLIDE) (presented at [RESOURCEFUL-2025](https://resourceful-workshop.github.io/resourceful-2025/papers.html)).

##### [Models on HuggingFace](https://huggingface.co/collections/ltg/slide-67d4538eac9736a6f068bf2a) 

- [how to run BERTs](src/usage_example.py)
- [how to run SLIDE-Fast](src/fast_usage_example.py)

SLIDE-Fast available on Huggingface now is an updated version which scores Strict Accuracy **93.6** on our test dataset and **94.9** on [Haas and Derczynski, 2021](https://aclanthology.org/2021.vardial-1.8/).
 
##### reproduce metrics (table 4):

```commandline
cd src/
./run_all.sh
```

##### reproduce evaluation on [nordic_langid](https://huggingface.co/datasets/strombergnlp/nordic_langid) (table 5)

obtain data from [nordic_langid on Huggingface](https://huggingface.co/datasets/strombergnlp/nordic_langid/tree/main) and put *test.csv into `src/evaluation`
```commandline
cd src/
python3 nordic_langid2jsonl.py 
python3 evaluate.py --method bert --model ltg/SLIDE-base --dataset nordic_dsl_test50k.jsonl
python3 evaluate.py --method bert --model ltg/SLIDE-base --dataset nordic_dsl_test10k.jsonl
```

The values that will be shown will be different from those in table 5 in the paper.

The values in table 5 were obtained with the understanding of loose accuracy as it is described in the paper.

The actual evaluate.py accepts a prediction if it is a subset of gold languages, not an intersection. (Values in table 4 were obtained with this understanding).
However, while it influences exact values (less than 2%), the models' ranking remains the same.

## Cite us

```commandline
@inproceedings{fedorova-etal-2025-multi,
    title = "Multi-label {S}candinavian Language Identification ({SLIDE})",
    author = "Fedorova, Mariia  and
      Frydenberg, Jonas Sebulon  and
      Handford, Victoria  and
      Lang{\o}, Victoria Ovedie Chruickshank  and
      Willoch, Solveig Helene  and
      Midtgaard, Marthe L{\o}ken  and
      Scherrer, Yves  and
      M{\ae}hlum, Petter  and
      Samuel, David",
    editor = "Holdt, {\v{S}}pela Arhar  and
      Ilinykh, Nikolai  and
      Scalvini, Barbara  and
      Bruton, Micaella  and
      Debess, Iben Nyholm  and
      Tudor, Crina Madalina",
    booktitle = "Proceedings of the Third Workshop on Resources and Representations for Under-Resourced Languages and Domains (RESOURCEFUL-2025)",
    month = mar,
    year = "2025",
    address = "Tallinn, Estonia",
    publisher = "University of Tartu Library, Estonia",
    url = "https://aclanthology.org/2025.resourceful-1.33/",
    pages = "179--189",
    ISBN = "978-9908-53-121-2",
    abstract = "Identifying closely related languages at sentence level is difficult, in particular because it is often impossible to assign a sentence to a single language. In this paper, we focus on multi-label sentence-level Scandinavian language identification (LID) for Danish, Norwegian Bokm{\r{a}}l, Norwegian Nynorsk, and Swedish. We present the Scandinavian Language Identification and Evaluation, SLIDE, a manually curated multi-label evaluation dataset and a suite of LID models with varying speed{--}accuracy tradeoffs. We demonstrate that the ability to identify multiple languages simultaneously is necessary for any accurate LID method, and present a novel approach to training such multi-label LID models."
}
```

##### Reformat UDHR, FLORES+ test data to our format

! FLORES+ is gated on the Huggingface Hub, ask for access first

```shell
cd src/openlid-v3-evaluation/
git submodule init
git submodule update
cd src/evaluation/
python3 scripts/download_udhr.py
python3 scripts/download_flores_plus.py
cd ../../../
python3 udhr2jsonl.py
python3 flores2jsonl.py
```