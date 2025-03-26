from typing import List

class AbstractLanguageIdentifier:
    def __init__(self, args):
        self.args = args
        self.languages = ["nb", "nn", "da", "sv", "other"]

    # This method should return a list of languages that the given text could be written in
    # A language can be either ("nb", "nn", "da", "sv" or "other")
    def identify(self, *args) -> List[str]:
        raise NotImplementedError
