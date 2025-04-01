from typing import List
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier
from identifiers.custom_tokenizer import CustomTokenizer


class BERTIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        tokenizer_path = args.model
        model_path = args.model
        self.mapping = {0: "da", 1: "nb", 2: "nn", 3: "other", 4: "sv"}
        self.other_if_below_threshold = args.other_if_below_threshold
        # set random seed
        torch.manual_seed(42)

        # set device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        args.custom_tokenizer = True
        token = os.getenv('HF')
        if args.custom_tokenizer:
            self.tokenizer = CustomTokenizer(
                tokenizer_path,
                cache_dir=os.path.expanduser('~/.cache/huggingface/hub'),
                token=token,
            )
        else:
            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True,
                token=token,
            )

        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=5,
            trust_remote_code=True,
            token=token,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def identify(self, text: str) -> List[str]:
        # get input ids
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.device)

        # multilabel
        logits = self.model(**input_ids)["logits"]
        sigmoids = torch.sigmoid(logits)
        # if other > 0.80, output only "other"
        if sigmoids[0][3] > 0.50:
            preds = torch.zeros(5).int()
            preds[3] = 1
        else:
            preds = (sigmoids > self.args.threshold).int()

        ids = preds.squeeze().nonzero().squeeze(0)
        if len(ids.shape) > 1:
            ids = ids.squeeze(1)
        # if no pred over threshold, output argmax or other
        if ids.nelement() == 0:
            if self.other_if_below_threshold:
                preds = [self.mapping[sigmoids.argmax().item()]]
            else:
                preds = ["other"]
        else:
            preds = [self.mapping[i.item()] for i in ids]

        return preds
