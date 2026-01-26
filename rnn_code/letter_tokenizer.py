# write a tokenizer that takes a string and returns a list of tokens
# each token should a letter
import logging
import unicodedata
import regex as re
from typing import Iterator

logger = logging.getLogger(__name__)


class LetterTokenizer:
    def __init__(
        self, spaces_are_tokens: bool = True, newlines_are_tokens: bool = True
    ):
        self.spaces_are_tokens = spaces_are_tokens
        self.newlines_are_tokens = newlines_are_tokens

    def tokenize(self, text: str) -> Iterator[str]:
        text = text.translate({91: None}).translate(
            {93: None}
        )  # remove square brackets
        text = re.sub(
            "<gap/>", "!", text
        )  # indicate lacuna of unknown length with "!" instead of "<gap/>"

        for char in text:
            for token in self.process_token(char):
                yield token

    def process_token(self, char: str) -> Iterator[str]:
        # lowercase
        char = char.lower()
        for c in unicodedata.normalize("NFD", char):
            if c in ".!#_":
                yield c
            elif c == "\u0345":
                # iota subscript
                yield "ι"
            elif c == " ":
                if self.spaces_are_tokens:
                    # handle space
                    yield c
            elif c == "\n":
                if self.newlines_are_tokens:
                    # handle newline
                    yield c
            elif unicodedata.category(c)[0] == "L":
                # sigma normalization
                if char in "σςϲ":
                    yield "ϲ"
                elif char in "φϕ":
                    yield "φ"
                elif char in "θϑ":
                    yield "θ"
                elif char in "Ϛϛ":
                    yield "Ϛ"
                elif char in "Ϝϝ":
                    yield "Ϝ"
                else:
                    if "GREEK" in unicodedata.name(c, ""):
                        if c in "αοιεϲντυρμωηκπλδγχβθφξζψ":
                            yield c
                        else:
                            yield "?"

            else:
                if unicodedata.category(c)[0] not in ("M", "P"):
                    logger.debug(f"Dropping unrecognized character: {c!r}")
