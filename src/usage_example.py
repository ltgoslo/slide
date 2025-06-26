import os

import torch
from torch.cuda import is_available
from transformers import AutoModelForSequenceClassification, set_seed

from identifiers.custom_tokenizer import CustomTokenizer

SIGMOID_THRESHOLD = 0.5

if __name__ == '__main__':
    set_seed(42)
    model_name = 'ltg/SLIDE-base'
    device = torch.device('cuda') if is_available() else torch.device('cpu')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    tokenizer = CustomTokenizer(
        model_name,
        cache_dir=os.path.expanduser('~/.cache/huggingface/transformers/hub/')
    )

    texts = [
        'En dag i livet',  # partially correct (not nynorsk)
        'Jag vill ha deg',  # partially correct (should be sv only)
        'Jeg er hvalrossen',  # correct (da, nb)
        'Denne fuglen har flydd',  # correct (nb, nn)
        'not a Scandinavian text at all',  # correct (other)
        'i sit berømte værk die normen und ihre bertretung i',  # ideally, would be da, although includes other
    ]

    with torch.no_grad():
        batch = tokenizer(texts, padding=True)
        batch['input_ids'] = torch.Tensor(batch['input_ids']).long().to(
            device
        )
        batch['attention_mask'] = torch.Tensor(batch['attention_mask']).long(
        ).to(device)
        batch['token_type_ids'] = torch.Tensor(batch['token_type_ids']).long(
        ).to(device)
        out = torch.sigmoid(model(**batch).logits)
        greater = out > SIGMOID_THRESHOLD
        indices = greater.nonzero()
        text_indices = indices[:, 0].unique()
        n_samples = torch.arange(end=out.shape[0]).to(device)

        for n_sample in n_samples:
            print(texts[n_sample])
            if n_sample in text_indices:
                this_sample_indices = indices[torch.where(indices[:, 0]==n_sample)][:, 1]
                print([model.config.id2label[idx.item()] for idx in this_sample_indices])
            else:
                print(['other'])
            print('---------')
