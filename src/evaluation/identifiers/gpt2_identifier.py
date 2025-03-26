from typing import List

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, pipeline,
)

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier

class GPT2Identifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.other = "other"
        self.gpt2_languages_dict = {
            'no': 'nb',
            'nn': 'nn',
            'da': 'da',
            'sv': 'sv',
        }
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
        )
        self.model = self.model.eval()
        self.pipe = None
        # recommended way from https://huggingface.co/nie3e/gpt2-lang-ident
        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=5,
            device=self.model.device,
        )
        self.threshold = args.threshold

    def identify(self, text: str) -> List[str]:
        result = self.pipe(text)[0]
        label = [
            self.gpt2_languages_dict.get(res['label'], 'other')
            for res in result if res['score'] > self.threshold
        ]
        if (not label) or ('other' in label):
            label = [self.other]
        return label