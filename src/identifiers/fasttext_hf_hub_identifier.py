import regex
from typing import List

import fasttext  # pip install fasttext
from huggingface_hub import hf_hub_download

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier


# defines what we want to remove from string for langID
NONWORD_REPLACE_STR = r"[^\p{Word}\p{Zs}]|\d"  # either (not a word nor a space) or (is digit)
NONWORD_REPLACE_PATTERN = regex.compile(NONWORD_REPLACE_STR)
SPACE_PATTERN = regex.compile(r"\s\s+")  # squeezes sequential whitespace


class FasttextHfHubIdentifier(AbstractLanguageIdentifier):
    def __init__(
            self,
            args,
            repo_id="facebook/fasttext-language-identification",
            filename="model.bin",
        ):
        super().__init__(args)
        self.repo_id = repo_id
        if not 'bin' in repo_id:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            model_path = repo_id
        self.model = fasttext.load_model(model_path)
        self.threshold = args.threshold

    def _preproccess_text(self, text: str) -> str:
        """Preprocesses a single line of text for lang ID."""
        if not isinstance(text, str):
            msg = "Input text must be a string."
            raise TypeError(msg)

        text = text.strip().replace('\n', ' ').lower()
        text = regex.sub(SPACE_PATTERN, " ", text)
        text = regex.sub(NONWORD_REPLACE_PATTERN, "", text)
        return text

    def identify(self, text: str) -> List[str]:
        if 'OpenLID' in self.repo_id:
            text = self._preproccess_text(text)
        prediction, scores = self.model.predict(text)
        if not prediction[0]:
            return ['other']
        predictions = []
        for pred, score in zip(prediction, scores):
            if score > self.threshold:
                language = pred.replace("__label__", "")
                if language.startswith("nob") or (language == 'nb'):
                    predictions.append("nb")
                if language.startswith("nno") or (language == 'nn'):
                    predictions.append("nn")
                if language.startswith("dan") or (language == 'da'):
                    predictions.append("da")
                if language.startswith("swe") or (language == 'sv'):
                    predictions.append("sv")
            else:
                return ['other']
        if not predictions:
            return ["other"]
        return predictions