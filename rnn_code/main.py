import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
import typer

import rnn_code.greek_char_data as greek_char_data
import rnn_code.greek_rnn as greek_rnn
import rnn_code.greek_utils as utils
from rnn_code.greek_char_generator import (
    DataItem,
    accuracy_evaluation,
    baseline_accuracy,
    predict_chars,
    predict_top_k,
    rank_reconstructions,
    train_model,
)
from rnn_code.greek_utils import device, model_path, seed, specs

# Note to self: the model at present seems to veer towards just predicting the most common character in the corpus


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        f"{Path(__file__).parent}/log/greek_data_processing.log"
    )
    file_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


model_name = "greek_rnn_no_dropout"

cur_path = Path(__file__).parent.absolute()

spaces_are_tokens = True
newlines_are_tokens = True

train_json = "train.json"
dev_json = "dev.json"
test_json = "test.json"
full_json = "full_data.json"
reconstruction_json = "reconstructions.json"

file_dir_path = cur_path.parent / "corpus"
json_list: list[str | os.PathLike] = [
    str(json_file) for json_file in file_dir_path.glob("*.json")
]
json_list += [str(json_file) for json_file in file_dir_path.glob("*.jsonl")]

app = typer.Typer(help="Greek character-level lacuna filler")

masking_strategies = ("random", "smart")

# These get rebound globally by bind_data()
train_data: list[DataItem] | None = None
dev_data: list[DataItem] | None = None
test_data: list[DataItem] | None = None
full_data: list[DataItem] | None = None
reconstructions: list[list[str]] | None = None
random_index: list[int] | None = None
position_in_original: list[int] | None = None


@app.command()
def train(
    masking_strategy: Annotated[
        Literal["random", "smart"],
        typer.Argument(
            autocompletion=lambda x: [s for s in masking_strategies if s.startswith(x)],
            help="pass 'random' for randomly picking single characters to mask, and 'smart' for masking randomly sized sequences",
        ),
    ],
    dynamic_remask: Annotated[
        bool,
        typer.Option(
            "--dynamic-remask",
            "-d",
            help="Remask texts each epoch",
        ),
    ] = False,
    force_partition: Annotated[
        bool,
        typer.Option(
            "--force-partition",
            "-f",
            help="By default data is only partitioned if partitions don't exist or if a change in the source data is detected (and there's data available to be partitioned)",
        ),
    ] = False,
    use_existing_partition: Annotated[bool, typer.Option("--use-existing")] = False,
):
    """Train LSTM for character filling; must pass masking strategy: [random, smart]."""

    if force_partition and use_existing_partition:
        raise typer.BadParameter(
            "Cannot use --force-partition and --use-existing together"
        )

    # NOTE: in prior version of code, masking_strategy was called masking, and instead of boolean dynamic remask was masking_strategy as ["once", "dynamic"]

    logger = logging.getLogger()

    logger.info(f"start greek data processing -- {datetime.now()}")
    if force_partition:
        do_partition = True
    elif use_existing_partition:
        do_partition = False
    else:
        do_partition = check_auto_partition(logger)

    if do_partition:
        partition(logger)
        write_data_hashes()

    train_data, dev_data, test_data, _, _, _, _ = bind_data(force_rebind=do_partition)

    # if masking_strategy is "dynamic", utils.mask_input returns the data as passed to it, and mask = True
    # if masking_strategy is "once", utils.mask_input adds masking to data by calling model.mask_and_label_characters(),
    # and returns newly masked ata and mask = False.

    logger.info("Training model")

    logger.info(
        f"Train {model_name} model specs: embed_size: {specs[0]}, hidden_size: {specs[1]}, proj_size: {specs[2]}, rnn n layers: {specs[3]}, share: {specs[4]}, dropout: {specs[5]}"
    )

    model = greek_rnn.RNN(specs)
    model = model.to(utils.device)
    # train_model is in greek_char_generator module
    # if mask = True, train_model() will call model.mask_and_label_characters() to remask.

    logger.info("Now training LSTM")
    logger.info(
        f"Masking strategy: {masking_strategy}\nTokens are masked: {'dynamically' if dynamic_remask else 'once'}"
    )
    if not dynamic_remask:
        train_data = model.mask_input(train_data, masking_strategy)
    # mask ultimately gets passed to greek_char_generator.train_batch(), to tell it whether to remask, which it should do if strategy is "dynamic"

    # actual_lacuna_mask_and_label replaces all characters within [] with '_', strips '[' and ']', and adds embedding indices to self.labels for
    # lacuna/mask characters.
    for i, data_item in enumerate(dev_data):
        dev_data[i] = model.actual_lacuna_mask_and_label(data_item)
    for i, data_item in enumerate(test_data):
        test_data[i] = model.actual_lacuna_mask_and_label(data_item)

    model = train_model(
        model,
        train_data=train_data,
        dev_data=dev_data,
        output_name=model_name,
        dynamic_remask=dynamic_remask,  # true if masking_strategy is dynamic, false if masking_strategy is once
        masking_strategy=masking_strategy,
        seed=seed,
    )
    run_eval(model, logger)
    logger.info(f"Training complete -- {datetime.now()}\n")


@app.command()
def predict(
    sentence: Annotated[
        str,
        typer.Argument(
            help="sentence with lacuna to fill marked as [..] with one . per missing character"
        ),
    ],
):
    """Returns top reconstruction of Greek sentence with lacunae in format [..] with one . per missing character"""
    logger = logging.getLogger()
    model = load_model(logger)
    sentence = re.sub("<gap/>", "!", sentence)
    pattern = re.compile(r"\[.*?\]")
    sentence = pattern.sub(lambda x: x.group().replace(" ", ""), sentence)
    instance = DataItem(text=sentence)
    data_item = model.actual_lacuna_mask_and_label(instance)
    predict_chars(model, data_item)


@app.command()
def predict_k(
    sentence: Annotated[
        str,
        typer.Argument(
            help="sentence with lacuna to fill marked as [..] with one . per missing character"
        ),
    ],
    k: Annotated[
        int, typer.Option("-k", "-K", "--k", help="get top k predictions for lacuna")
    ] = -1,
):
    """Returns top k reconstruction of Greek sentence with lacunae in format [..] with one . per missing character"""
    if k == -1:
        k = typer.prompt(
            "Please enter an integer k for the number of predictions to return (from most to least probable)",
            type=int,
        )
    typer.echo(f"Fetching top {k} reconstructions", err=True)
    logger = logging.getLogger()
    sentence = re.sub("<gap/>", "!", sentence)
    pattern = re.compile(r"\[.*?\]")
    sentence = pattern.sub(lambda x: x.group().replace(" ", ""), sentence)

    model = load_model(logger)

    instance = DataItem(text=sentence)
    data_item = model.actual_lacuna_mask_and_label(instance)
    predict_top_k(model, data_item, k)


@app.command()
def rank(
    sentence: Annotated[
        str,
        typer.Argument(
            help="sentence with one lacuna to fill marked as [..] with one . per missing character"
        ),
    ],
    options: Annotated[
        list[str] | None,
        typer.Argument(
            help="space-separated options for filling lacuna (no spaces in reconstruction itself)"
        ),
    ] = None,
):
    if options is None:
        opt_str = typer.prompt(
            "Please enter options of same length as lacuna, separated by spaces with quotation marks if needed:",
            type=str,
        )
        options = opt_str.split()

    logger = logging.getLogger()
    model = load_model(logger)
    ranking = rank_reconstructions(model, sentence, cast(list[str], options))
    typer.echo("Ranking:", err=True)
    typer.echo("(option, log_sum)", err=True)
    for option in ranking:
        typer.echo(option, err=False)


@app.command()
def eval():
    logger = logging.getLogger()
    logger.info("Evaluating a previously trained model")
    model = load_model(logger)
    # actual_lacuna_mask_and_label replaces all characters within [] with '_', strips '[' and ']', and adds embedding indices to self.labels for
    # lacuna/mask characters.
    bind_data()

    global test_data
    assert isinstance(test_data, list)
    for i, data_item in enumerate(cast(list[DataItem], test_data)):
        test_data[i] = model.actual_lacuna_mask_and_label(data_item)

    run_eval(model, logger)


def run_eval(model: greek_rnn.RNN, logger: logging.Logger):
    logger.info(model)
    greek_rnn.count_parameters(model)

    # run model on test set
    global test_data
    global reconstructions
    if test_data is None or len(test_data) == 0:
        raise ValueError("test_data not properly initialized for run_eval()")
    if reconstructions is None or len(reconstructions) == 0:
        raise ValueError("Reconstructions not properly initialized for run_eval()")
    test_indexes = [
        d.text_number for d in test_data if isinstance(d.text_number, int)
    ]  # these indexes tell where to look in reconstructed_lacunae to get gold standard scholarly reconstructions
    test_indexes = cast(list[int], test_indexes)

    logger.info(f"Test data has {len(test_data)} texts")

    test_reconstructions = [reconstructions[i] for i in test_indexes]

    # now test_texts gives us our texts with lacunae as masks in the format [##] for all the texts in the test partition (in random order).
    # reconstructed_lacunae is a list of lists giving all lacunae reconstructions for all texts in json_blocks
    # note that reconstructions are in the order of the original json_blocks, while test_texts is in the random order from the dataset partition.

    num_reconstructions = 0
    num_texts_with_reconstructions = 0
    for text in test_reconstructions:
        num_reconstructions += len(text)
        if len(text) > 0:
            num_texts_with_reconstructions += 1

    logger.info(
        f"{num_reconstructions} reconstructed lacunae read in accross {num_texts_with_reconstructions} texts"
    )

    test_list = [i for i in range(len(test_data))]
    # accuracy evaluation
    logger.info("Test Reconstructed:")
    model = cast(greek_rnn.RNN, model)
    accuracy_evaluation(model, test_data, test_list)  # in greek_char_generator
    baseline_accuracy(model, test_data, test_list)  # in greek_char_generator


def blake2_file(path: str | os.PathLike, chunk_size=16 * 1024 * 1024):
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_data_hashes() -> None:
    for file in json_list:
        hash = blake2_file(file)
        Path(str(file) + ".hash").write_text(hash)


def check_auto_partition(logger: logging.Logger) -> bool:
    if not json_list:
        logger.info(
            f"Can't repartition, because there are no source data files in {file_dir_path}"
        )
        return False
    for file in json_list:
        file = str(file)
        hash_path = Path(file + ".hash")
        if not hash_path.exists():
            logger.info("Autodetect partition: hashes missing for data files")
            return True
        if blake2_file(file) != hash_path.read_text():
            logger.info("Autodetect partition: data files have changed")
            return True
    logging.info("Autodetect: no need for repartition")
    return False


def partition(logger: logging.Logger) -> None:
    # Do full data partition
    logger.info("Now partitioning data")
    logger.info(f"{len(json_list)} json file(s) found")
    if not json_list:
        raise FileNotFoundError("No json files were found")
    for json_file in json_list:
        logger.info(f"{json_file}")
    (
        local_train_data,
        local_dev_data,
        local_test_data,
        local_random_index,
        local_reconstructions,
        mapping_greek_to_original,
    ) = greek_char_data.read_datafiles(json_list)

    local_full_data = local_train_data + local_dev_data + local_test_data
    logger.info(f"train: {len(local_train_data)} texts")
    logger.info(f"dev: {len(local_dev_data)} texts")
    logger.info(f"test: {len(local_test_data)} texts")
    logger.info(f"full: {len(local_full_data)} texts")

    # write to partition files
    # write_to_json() writes file with format {"text_index": int, "text": "..."}
    greek_char_data.write_to_json(
        train_json,
        local_train_data,
        local_random_index[: len(local_train_data)],
        mapping_greek_to_original,
    )
    greek_char_data.write_to_json(
        dev_json,
        local_dev_data,
        local_random_index[
            len(local_train_data) : (len(local_train_data) + len(local_dev_data))
        ],
        mapping_greek_to_original,
    )
    greek_char_data.write_to_json(
        test_json,
        local_test_data,
        local_random_index[(len(local_train_data) + len(local_dev_data)) :],
        mapping_greek_to_original,
    )

    with open(f"{cur_path}/data/{reconstruction_json}", "w") as jsonfile:
        for i, lst in enumerate(local_reconstructions):
            json_block = {"text_index": i, "reconstructions": lst}
            json_line = json.dumps(json_block, ensure_ascii=False)
            jsonfile.write(json_line + "\n")


def load_model(logger: logging.Logger) -> greek_rnn.RNN:
    preload_model = Path(model_path) / "best"
    preload_model = [mod for mod in preload_model.glob("*.pth")]
    if not preload_model:
        raise FileNotFoundError(f"No pth file found in {cur_path}/models/best")
    # get most recent model in models/best dir
    preload_model = str(max(preload_model, key=lambda f: f.stat().st_mtime))
    logger.info(f"Loading model: {preload_model}")
    model = torch.load(preload_model, map_location=device, weights_only=False)
    logger.info(
        f"Load model: {model} with specs: embed_size: {model.specs[0]}, hidden_size: {model.specs[1]}, proj_size: {model.specs[2]}, rnn n layers: {model.specs[3]}, share: {model.specs[4]}, dropout: {model.specs[5]}"
    )
    return model


def bind_data(
    force_rebind: bool = False,
) -> tuple[
    list[DataItem],
    list[DataItem],
    list[DataItem],
    list[DataItem],
    list[str],
    list[int],
    list[int],
]:
    # this function binds global variables to the following lists:
    # train_data: list of dictionaries of format {"text_index": int, "text": "..."}
    # -> scholarly reconstructions are indistinguishable from regular text
    # dev_data: list of dictionaries of format {"text_index": int, "text": "..."}
    # -> lacunae of known length are present as "[_]" with '_' * number of lacuna characters.
    # -> lacunae of unknown length are present as '!'.
    # test_data: list of dictionaries of format {"text_index": int, "text": "..."}
    # -> lacunae of known length are present as "[_]" with '_' * number of lacuna characters.
    # -> lacunae of unknown length are present as '!'.
    # random_index: list of ints. Order of random_index corresponds to
    # [train_data + dev_data + test_data].
    # reconstructions: list of lists, for [a, b], a is a list of reconstructions for the lacunae in the first text in the original json file.
    # To find the reconstructions for a text in one of the data files,
    # take its index i in [train_data + dev_data + test_data], then take random_index[i], then take reconstructions[random_index[i]].
    # bind_data then converts relevant JSON dict lists to lists of DataItem objects

    global train_data
    global dev_data
    global test_data
    global full_data
    global reconstructions
    global random_index
    global position_in_original

    if train_data is None or force_rebind:
        train_data = []
        with open(f"{cur_path}/data/{train_json}", "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                train_data.append(jsonDict)

    if dev_data is None or force_rebind:
        dev_data = []
        with open(f"{cur_path}/data/{dev_json}", "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                dev_data.append(jsonDict)

    if test_data is None or force_rebind:
        test_data = []
        with open(f"{cur_path}/data/{test_json}", "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                test_data.append(jsonDict)

    if full_data is None or force_rebind:
        full_data = (
            cast(list, train_data) + cast(list, dev_data) + cast(list, test_data)
        )

    if random_index is None or position_in_original is None or force_rebind:
        random_index = []
        position_in_original = []
        # Make it possible to match each text in the partition to a block in the original
        for block in full_data:
            block = cast(dict, block)
            random_index.append(block["text_index"])
            position_in_original.append(block["position_in_original"])

        # get the file in same order as original_json with scholarly reconstructions of lacunae
        reconstructions = []
        with open(f"{cur_path}/data/{reconstruction_json}", "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                reconstructions.append(jsonDict["reconstructions"])

    # Now, convert each of train_data, dev_data, and test_data into lists of objects of class greek_utils.DataItem.
    for i, data_dict in enumerate(cast(list, train_data)):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        train_data[i] = new_data_item

    for i, data_dict in enumerate(cast(list, dev_data)):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        dev_data[i] = new_data_item

    for i, data_dict in enumerate(cast(list, test_data)):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        test_data[i] = new_data_item
    return (
        cast(list[DataItem], train_data),
        cast(list[DataItem], dev_data),
        cast(list[DataItem], test_data),
        cast(list[DataItem], full_data),
        cast(list[str], reconstructions),
        cast(list[int], random_index),
        cast(list[int], position_in_original),
    )


def main():
    setup_logging()
    app()
