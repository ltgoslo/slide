---
language:
- nb
- nn
- sv
- da
- 'no'
license: apache-2.0
---
## SLIDE-fast

This is an updated version of the fast multilabel Scandinavian language identification model described in our [paper](https://aclanthology.org/2025.resourceful-1.33/).
The updated version is `able' to distinguish Nynorsk from Icelandic/Faroese, scoring Strict Accuracy **93.6** on our test dataset and **94.9** on [Haas and Derczynski, 2021](https://aclanthology.org/2021.vardial-1.8/).

## Example usage

```commandline
git clone git@github.com:ltgoslo/slide.git
cd src/
python3 fast_usage_example.py
```

## Cite us
```
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