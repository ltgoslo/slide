from typing import List
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import regex as re

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier

class CustomTokenizer:
    def __init__(self, pretrained_tokenizer_path, cache_dir, token):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_path,
            cache_dir= cache_dir,
            trust_remote_code = True,
            token=token,
        )

        new_tokens = ['<num>', '<url>', '<mail>']
        #self.tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
        self.tokenizer.add_tokens(new_tokens)

        # Define regex patterns for numbers and URLs
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.url_pattern = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
        self.mail_pattern = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')

        self.replacement_symbols = {
            "url": '<url>',#'Ħ',
            "num": '<num>',#'Ĳ',
            "mail": '<mail>',#'Ĵ',
        }


    def preprocess(self, text):
        text = self.number_pattern.sub(self.replacement_symbols["num"], text)
        text = self.url_pattern.sub(self.replacement_symbols["url"], text)
        text = self.mail_pattern.sub(self.replacement_symbols["mail"], text)

        return text

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying tokenizer to retain all its methods
        return getattr(self.tokenizer, attr)

    def __call__(self, text, **kwargs):
        preprocessed_text = self.preprocess(text)
        return self.tokenizer(preprocessed_text, **kwargs)

class BERTIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        tokenizer_path = args.model
        model_path = args.model
        self.mapping = {0: "da", 1: "nb", 2: "nn", 3: "other", 4: "sv"}

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
        sigmoids = torch.sigmoid(self.model(**input_ids)["logits"])
        # if other > 0.80, output only "other"
        if sigmoids[0][3] > 0.50:
            preds = torch.zeros(5).int()
            preds[3] = 1
        else:
            preds = (sigmoids > self.args.threshold).int()

        ids = [torch.where(row == 1)[0].tolist() for row in preds][0]
        # if no pred over threshold, output argmax
        if len(ids) == 0:
            preds = [self.mapping[
                         self.model(**input_ids)["logits"].argmax().item()]]
        else:
            preds = [self.mapping[i] for i in ids]

        return preds

    def logits_sigmoids(self, text: str):
        # get input ids
        input_ids = self.tokenizer(text, return_tensors="pt", padding=True).to(
            self.device)

        # multilabel
        logits = self.model(**input_ids)["logits"]
        sigmoids = torch.sigmoid(logits)

        return logits, sigmoids