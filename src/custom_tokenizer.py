import re

from transformers import AutoTokenizer


class CustomTokenizer:
    def __init__(self, pretrained_tokenizer_path, cache_dir, token=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_tokenizer_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=token,
            )

        new_tokens = ['<num>', '<url>', '<mail>']
        self.tokenizer.add_tokens(new_tokens)

        # Define regex patterns for numbers and URLs
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.url_pattern = re.compile(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
        )
        self.mail_pattern = re.compile(
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        )

        self.replacement_symbols = {
            "url": '<url>',  # 'Ħ',
            "num": '<num>',  # 'Ĳ',
            "mail": '<mail>',  # 'Ĵ',
        }

    def preprocess(self, text):
        new = []
        for sample in text: # a batch
            sample = self.number_pattern.sub(self.replacement_symbols["num"], sample)
            sample = self.url_pattern.sub(self.replacement_symbols["url"], sample)
            sample = self.mail_pattern.sub(self.replacement_symbols["mail"], sample)
            new.append(sample)
        return new

    def __getattr__(self, attr):
        # Delegate attribute access to the underlying tokenizer to retain all its methods
        return getattr(self.tokenizer, attr)

    def __call__(self, text, **kwargs):
        preprocessed_text = self.preprocess(text)
        return self.tokenizer(preprocessed_text, **kwargs)
