from typing import List

import fasttext
from huggingface_hub import hf_hub_download
import torch

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier


class MlpIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        self.threshold = args.threshold
        model_path = hf_hub_download(repo_id=args.model,
                                     filename="pytorch_model.bin")
        self.clf = torch.load(model_path, weights_only=False)
        self.clf.eval()
        self.id2label = {}
        model_path = hf_hub_download(repo_id='cis-lmu/glotlid',
                                     filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.id2label = {i: label for i, label in enumerate(["nb", "nn", "da", "sv", "other"])}

    def identify(self, text: str) -> List[str]:
        vec = self.model.get_sentence_vector(text)
        with torch.no_grad():
            logits = self.clf(torch.from_numpy(vec))
            greater = logits > self.threshold
            indices = greater.squeeze().nonzero()
            if indices.shape[0] == 0:
                indices = torch.Tensor([[4]])
            label = [self.id2label[idx.item()] for idx in indices]
        return label