import json
import logging
import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from nltk.util import ngrams

from greek_rnn.letter_tokenizer import LetterTokenizer

seed = 1234

SPACES_ARE_TOKENS = False
NEWLINES_ARE_TOKENS = False

logger = logging.getLogger(__name__)

# This module contains the model hyperparameters as well as various functions for data processing and logging information about the model

tokenizer = LetterTokenizer(
    spaces_are_tokens=SPACES_ARE_TOKENS, newlines_are_tokens=NEWLINES_ARE_TOKENS
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logger.info(f"torch version & device: {torch.__version__, device}")

share = False
# share = True  # share embedding table with output layer for weight tying, cf. Jurafsky and Martin 9.2.3

# embed_size = 20
# embed_size = 50
# embed_size = 100
# embed_size = 150
# embed_size = 200
embed_size = 300


if share:
    proj_size = embed_size
else:
    # proj_size = 0 # disable projection layer
    proj_size = 150
    # proj_size = 200
    # proj_size = 250
    # proj_size = 350


# NOTE: torch expects hidden_size // 2 > proj_size
# (hidden_size gets halved because forward and backward outputs
# get concatenated in BiLSTM)

# hidden_size = 100
# hidden_size = 150
# hidden_size = 200
hidden_size = 300
# hidden_size = 400
# hidden_size = 500
# hidden_size = 1000

# rnn_nLayers = 2
# rnn_nLayers = 3
rnn_nLayers = 4

dropout = 0.0
# dropout = 0.1

masking_proportion = 0.15

specs = [
    embed_size,
    hidden_size,
    proj_size,
    rnn_nLayers,
    share,
    dropout,
    masking_proportion,
]

decoder_specs = {
    "gru": {
        "name": "gru",
        "input_size": proj_size if proj_size > 0 else hidden_size,
        "hidden_size": proj_size if proj_size > 0 else hidden_size,
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False,
        "use_teacher_labels": True,
    },
    "lstm": {
        "name": "lstm",
        "input_size": proj_size if proj_size > 0 else hidden_size,
        "hidden_size": proj_size if proj_size > 0 else hidden_size,
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0.0,
        "bidirectional": False,
        "use_teacher_labels": True,
    },
}

# learning_rate = 0.0001
learning_rate = 0.0003
# learning_rate = 0.001
# learning_rate = 0.003
# learning_rate = 0.01

# NOTE: batch_size just controls initial batching for gradient accumulation,
# not the forward pass
# initial batch size
batch_size = 1
# batch_size = 2
# batch_size = 5
# batch_size = 10
# batch_size = 20

# increase the batch size every epoch by this factor
# batch_size_multiplier = 1
batch_size_multiplier = 1.4
# batch_size_multiplier = 1.6
# batch_size_multiplier = 2

# nEpochs = 1
# nEpochs = 2
# nEpochs = 3
# nEpochs = 4
# nEpochs = 10
# nEpochs = 20
# nEpochs = 30
nEpochs = 50

L2_lambda = 0.0
# L2_lambda = 0.001

# Early stopping patience - number of epochs to wait before stopping if dev loss doesn't improve
# patience = 3
patience = 5
# patience = 10

model_path = str(Path(__file__).parent / "models")


class DataItem:
    def __init__(
        self,
        text: str | None = None,
        text_number: int | None = None,
        indexes: list[int] | None = None,
        mask: list[bool] | None = None,
        labels: list[int] | None = None,
        position_in_original: int | None = None,
    ):
        self.text_number = text_number
        self.text = text  # original text
        self.indexes = indexes  # tensor of indexes of characters or tokens
        self.mask = (
            mask  # list of indexes same size as index, true when character is masked
        )
        self.labels = labels  # list of indexes for attention mask
        self.position_in_original = position_in_original  # integer marking location of DataItem in original set of texts (including non-Greek texts)


def get_home_path() -> str:
    return os.path.expanduser("~")


UNICODE_MARK_NONSPACING = "Mn"
COMBINING_DOT = "COMBINING DOT BELOW"
MN_KEEP_LIST = [COMBINING_DOT]


def filter_diacritics(string: str) -> str:
    # First decompose the string to separate base characters from combining marks
    decomposed = unicodedata.normalize("NFD", string)
    new_string = ""
    for character in decomposed:
        if (
            unicodedata.category(character) != UNICODE_MARK_NONSPACING
            or unicodedata.name(character) in MN_KEEP_LIST
        ):
            new_string = new_string + character
    return new_string.lower()


def read_datafile(json_file: str, data_list=None) -> list[DataItem]:
    if data_list is None:
        data_list = []

    with open(json_file, "r") as file:
        for i, line in enumerate(file):
            jsonDict = json.loads(line)
            try:
                if len(jsonDict["text"]) == 0:
                    continue
                data_list.append(DataItem(text=jsonDict["text"]))
            except Exception:
                logger.warning(f"Incorrect formatting in line {i} of {json_file}")

    return data_list


def filter_brackets(input_string: str) -> str:
    input_string = re.sub(r"\|", "", input_string)
    input_string = re.sub(r"\{", "", input_string)
    input_string = re.sub(r"\}", "", input_string)
    input_string = re.sub(r"\(.*\)", "", input_string)
    input_string = re.sub(r"\[", "", input_string)
    input_string = re.sub(r"\]", "", input_string)
    return input_string


def skip_sentence(input_string: str) -> bool:
    skip = False
    if has_more_than_one_latin_character(input_string):
        skip = True
    elif "[----]" in input_string:
        skip = True
    elif len(input_string) < 5:
        skip = True
    return skip


def has_more_than_one_latin_character(input_string: str) -> bool:
    latin_count = sum(1 for char in input_string if char in string.ascii_letters)
    return latin_count > 1


def construct_trigram_lookup() -> dict[str, dict[str, int]]:
    # read in training data
    with open(f"{Path(__file__).parent}/data/train.json", "r") as jsonFile:
        texts = [json.loads(line)["text"].strip() for line in jsonFile]

    ngram_counts: dict[tuple, int] = Counter()
    for text in texts:
        text = list(tokenizer.tokenize(text))
        ngrams_obj = ngrams(text, 3, pad_left=True, left_pad_symbol="<s>")
        ngram_counts.update(ngrams_obj)

    look_up = {}
    for entry in ngram_counts:
        first_second = entry[0] + entry[1]
        third = entry[2]
        if first_second in look_up:
            if third in look_up[first_second]:
                raise ValueError(
                    "A trigram has been encountered twice, which shouldn't happen"
                )
            else:
                look_up[first_second][third] = ngram_counts[entry]
        else:
            look_up[first_second] = {entry[2]: ngram_counts[entry]}

    # write look_up tp file
    with open(f"{Path(__file__).parent}/data/trigram_lookup.json", "w") as json_file:
        json.dump(look_up, json_file, indent=4)

    return look_up
