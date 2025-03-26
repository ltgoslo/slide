from typing import List

import fasttext  # pip install fasttext
from huggingface_hub import hf_hub_download

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier


class FasttextHfHubIdentifier(AbstractLanguageIdentifier):
    def __init__(
            self,
            args,
                 repo_id="facebook/fasttext-language-identification",
                 filename="model.bin",
                 ):
        super().__init__(args)
        if not 'bin' in repo_id:
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            model_path = repo_id
        self.model = fasttext.load_model(model_path)

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        if not prediction:
            return ['other']
        predictions = []
        for pred in prediction:
            language = pred.replace("__label__", "")
            if language.startswith("nob") or (language == 'nb'):
                predictions.append("nb")
            if language.startswith("nno") or (language == 'nn') :
                predictions.append("nn")
            if language.startswith("dan")  or (language == 'da'):
                predictions.append("da")
            if language.startswith("swe")  or (language == 'sv'):
                predictions.append("sv")
        if not predictions:
            return ["other"]
        return predictions