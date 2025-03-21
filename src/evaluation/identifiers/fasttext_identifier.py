import os
from typing import List
import urllib

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier

class FasttextLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext

        path = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

        # download the model if it does not exist
        if not os.path.exists("lid.176.bin"):
            urllib.request.urlretrieve(path, "lid.176.bin")

        self.model = fasttext.load_model("lid.176.bin")

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language == "no":
            return ["nb"]

        if language in self.languages:
            return [language]
        return ["other"]
