from typing import List
import random

from identifiers.abstract_language_identifier import AbstractLanguageIdentifier


class RandomLanguageIdentifier(AbstractLanguageIdentifier):
    def __init__(self, args):
        super().__init__(args)

    def identify(self, text: str) -> List[str]:
        return [random.choice(self.languages)]