import os
from typing import List
import urllib

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier

class OpenlidLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)
        import fasttext  # pip install fasttext

        path = "https://data.statmt.org/lid/lid201-model.bin.gz"

        # download the model if it does not exist
        if not os.path.exists("lid201-model.bin"):
            urllib.request.urlretrieve(path, "lid201-model.bin.gz")
            os.system("gunzip lid201-model.bin.gz")

        self.model = fasttext.load_model("lid201-model.bin")

    def identify(self, text: str) -> List[str]:
        prediction = self.model.predict(text)[0]
        language = prediction[0].replace("__label__", "")

        if language.startswith("nob_"):
            return ["nb"]
        if language.startswith("nno_"):
            return ["nn"]
        if language.startswith("dan_"):
            return ["da"]
        if language.startswith("swe_"):
            return ["sv"]
        return ["other"]