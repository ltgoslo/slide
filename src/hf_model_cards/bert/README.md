---
license: apache-2.0
language:
- da
- 'no'
- nb
- nn
- sv
base_model:
- ltg/norbert3-base
pipeline_tag: text-classification
---

# SLIDE-base

Scandinavian language identification model described in paper **[Multi-label Scandinavian Language Identification (SLIDE)
](https://arxiv.org/abs/2502.06692)**.

## SLIDE sizes
- [SLIDE-base (123M)](https://huggingface.co/ltg/SLIDE-base)
- [SLIDE-small (40M)](https://huggingface.co/ltg/SLIDE-small)
- [SLIDE-x-small (15M)](https://huggingface.co/ltg/SLIDE-x-small)

## Example usage
This model currently needs a custom wrapper from modeling_norbert.py, you should therefore load the model with trust_remote_code=True.

```commandline
git clone git@github.com:ltgoslo/slide.git
cd src/
python3 usage_example.py
```
## Cite us

```
@misc{fedorova2025multilabelscandinavianlanguageidentification,
      title={Multi-label Scandinavian Language Identification (SLIDE)}, 
      author={Mariia Fedorova and Jonas Sebulon Frydenberg and Victoria Handford and Victoria Ovedie Chruickshank Langø and Solveig Helene Willoch and Marthe Løken Midtgaard and Yves Scherrer and Petter Mæhlum and David Samuel},
      year={2025},
      eprint={2502.06692},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.06692}, 
}
```