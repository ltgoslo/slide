import regex
from typing import List

import fasttext  # pip install fasttext
from huggingface_hub import hf_hub_download

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier


# defines what we want to remove from string for langID
NONWORD_REPLACE_STR = r"[^\p{Word}\p{Zs}]|\d"  # either (not a word nor a space) or (is digit)
NONWORD_REPLACE_PATTERN = regex.compile(NONWORD_REPLACE_STR)
SPACE_PATTERN = regex.compile(r"\s\s+")  # squeezes sequential whitespace


class FasttextEnsembleIdentifier(AbstractLanguageIdentifier):
    def __init__(
            self,
            args,
            languages,
            filename="model.bin",
        ):
        super().__init__(args, languages)
        self.repo_id = args.model  # always GlotLid
        self.second_repo_id = args.second_model  # always OpenLID
        self.model = self._load_model(self.repo_id, filename)
        self.second_model = self._load_model(self.second_repo_id, filename)
        self.threshold = args.threshold
        self.k = args.k

    @staticmethod
    def _load_model(model_path, filename):
        if not 'bin' in model_path:
            model_path = hf_hub_download(repo_id=model_path, filename=filename)
        return fasttext.load_model(model_path)

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
        prediction, _ = self.model.predict(text, k=self.k)
        if 'OpenLID' in self.second_repo_id:
            text = self._preproccess_text(text)
        second_prediction, second_scores = self.second_model.predict(text)
        if not prediction[0]: # may be redundant
            prediction[0] = ['other']
        if not second_prediction[0]:
            second_prediction[0] = ['other']
        predictions = []
        second_pred = second_prediction[0]
        second_score = second_scores[0]
        if second_pred in prediction:
            if second_score > self.threshold:  # thresholding only for OpenLID. different from FastText where it is greater or equal
                language = second_pred.replace("__label__", "")
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
        else:
            return ['other']
        if not predictions:
            return ["other"]
        return predictions