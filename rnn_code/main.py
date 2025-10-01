import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import torch

import rnn_code.greek_char_data as greek_char_data
import rnn_code.greek_rnn as greek_rnn
import rnn_code.greek_utils as utils
from rnn_code.greek_char_generator import (
    DataItem,
    accuracy_evaluation,
    baseline_accuracy,
    predict,
    predict_top_k,
    rank,
    train_model,
)
from rnn_code.greek_utils import device, logger, model_path, seed, specs

# Note to self: the model at present seems to veer towards just predicting the most common character in the corpus

if __name__ == "__main__":
    cur_path = Path(__file__).parent.absolute()
    parser = argparse.ArgumentParser(description="Greek character level generator")
    parser.add_argument(
        "-tr",
        "--train",
        required=False,
        help="True to train the model",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--masking",
        required=False,
        help="masking=random, smart; required argument unless ranking reconstructions",
        choices=["random", "smart"],
        action="store",
    )
    parser.add_argument(
        "-ms",
        "--masking-strategy",
        required=False,
        help="masking-strategy=once, dynamic; required argument unless ranking reconstructions",
        choices=["once", "dynamic"],
        action="store",
    )
    parser.add_argument(
        "-p",
        "--partition",
        required=False,
        help="create the data set partition",
        action="store_true",
    )
    parser.add_argument(
        "-ev",
        "--eval",
        required=False,
        help="conduct evaluation for actual lacuna test sets",
        action="store_true",
    )
    parser.add_argument(
        "-pr",
        "--predict",
        required=False,
        help="sentence to predict",
        action="store",
    )
    parser.add_argument(
        "-prk",
        "--predict_top_k",
        required=False,
        help="sentence to predict",
        action="store",
    )
    parser.add_argument(
        "-r",
        "--rank",
        required=False,
        help="ranking likelihood of options",
        action="store_true",
    )
    args = parser.parse_args()

    logger.info(f"start greek data processing -- {datetime.now()}")

    spaces_are_tokens = True
    newlines_are_tokens = True

    # step 1 - set data files, partition the in data if needed (train/dev/test split)
    train_json = "train.json"
    dev_json = "dev.json"
    test_json = "test.json"
    full_json = "full_data.json"
    reconstruction_json = "reconstructions.json"

    file_dir_path = cur_path.parent.parent / "Corpora/Lacuna"
    # read in json file
    json_list = [str(json_file) for json_file in file_dir_path.glob("*.json")]

    if args.partition:
        # Do full data partition
        logging.info("Now partitioning data")
        logging.info(f"{len(json_list)} json file(s) found: ")
        for json_file in json_list:
            logging.info(f"{json_file}")
        (
            train_data,
            dev_data,
            test_data,
            random_index,
            reconstructions,
            mapping_greek_to_original,
        ) = greek_char_data.read_datafiles(json_list)

        full_data = train_data + dev_data + test_data
        logger.info(f"train: {len(train_data)} texts")
        logger.info(f"dev: {len(dev_data)} texts")
        logger.info(f"test: {len(test_data)} texts")
        logger.info(f"full: {len(full_data)} texts")

        # write to partition files
        # write_to_json() writes file with format {"text_index": int, "text": "..."}
        greek_char_data.write_to_json(
            train_json,
            train_data,
            random_index[: len(train_data)],
            mapping_greek_to_original,
        )
        greek_char_data.write_to_json(
            dev_json,
            dev_data,
            random_index[len(train_data) : (len(train_data) + len(dev_data))],
            mapping_greek_to_original,
        )
        greek_char_data.write_to_json(
            test_json,
            test_data,
            random_index[(len(train_data) + len(dev_data)) :],
            mapping_greek_to_original,
        )

        with open(f"{cur_path}/data/{reconstruction_json}", "w") as jsonfile:
            for i, lst in enumerate(reconstructions):
                json_block = {"text_index": i, "reconstructions": lst}
                json_line = json.dumps(json_block, ensure_ascii=False)
                jsonfile.write(json_line + "\n")

    train_data = []
    dev_data = []
    test_data = []
    with open(f"{cur_path}/data/{train_json}", "r") as file:
        for line in file:
            jsonDict = json.loads(line)
            train_data.append(jsonDict)
    with open(f"{cur_path}//data/{dev_json}", "r") as file:
        for line in file:
            jsonDict = json.loads(line)
            dev_data.append(jsonDict)
    with open(f"{cur_path}//data/{test_json}", "r") as file:
        for line in file:
            jsonDict = json.loads(line)
            test_data.append(jsonDict)
    full_data = train_data + dev_data + test_data

    random_index = []
    position_in_original = []
    # Make it possible to match each text in the partition to a block in the original
    for block in full_data:
        random_index.append(block["text_index"])
        position_in_original.append(block["position_in_original"])

    # get the file in same order as original_json with scholarly reconstructions of lacunae
    reconstructions = []
    for json_file in json_list:
        with open(f"{cur_path}/data/{reconstruction_json}", "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                reconstructions.append(jsonDict["reconstructions"])

    # At this stage there are the following lists:
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

    # Now, convert each of train_data, dev_data, and test_data into lists of objects of class greek_utils.DataItem.
    for i, data_dict in enumerate(train_data):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        train_data[i] = new_data_item
    for i, data_dict in enumerate(dev_data):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        dev_data[i] = new_data_item
    for i, data_dict in enumerate(test_data):
        new_data_item = utils.DataItem(
            text_number=data_dict["text_index"],
            text=data_dict["text"],
            position_in_original=data_dict["position_in_original"],
        )
        test_data[i] = new_data_item

    model_name = "greek_rnn_no_dropout"

    # step 2 - model training
    mask_type = args.masking
    masking_strategy = args.masking_strategy

    if args.train:
        logger.info("Training model")

        logger.info(
            f"Train {model_name} model specs: embed_size: {specs[0]}, hidden_size: {specs[1]}, proj_size: {specs[2]}, rnn n layers: {specs[3]}, share: {specs[4]}, dropout: {specs[5]}"
        )

        model = greek_rnn.RNN(specs)
        model = model.to(utils.device)
    else:
        logger.info("Using a pre-trained model")
        preload_model = Path(model_path) / "best"
        preload_model = [mod for mod in preload_model.glob("*.pth")]
        if not preload_model:
            raise FileNotFoundError(f"No pth file found in {model_path}/best")
        # get most recent model in models/best dir
        preload_model = str(max(preload_model, key=lambda f: f.stat().st_mtime))
        logger.info(f"Loading model: {preload_model}")
        model = torch.load(preload_model, map_location=device, weights_only=False)
        logger.info(
            f"Load model: {model} with specs: embed_size: {model.specs[0]}, hidden_size: {model.specs[1]}, proj_size: {model.specs[2]}, rnn n layers: {model.specs[3]}, share: {model.specs[4]}, dropout: {model.specs[5]}"
        )

    logger.info(model)
    utils.count_parameters(model)

    # Masking functions: greek_utils.mask_input, greek_rnn.RNN.mask_and_label_characters, greek_rnn.RNN.actual_lacuna_mask_and_label.
    # mask_input is a gate that runs mask_and_label_characters if masking of training data happens once,
    # otherwise, since masking is dynamic, it simply signals to greek_char_generator.train_model and greek_char_generator.train_batch
    # that mask_and_label_characters needs to be run again on each sequence in each epoch.
    # actual_lacuna_mask_and_label is for converting lacunae in the dev and test sets into masks.

    # Now convert lacunas in dev and test sets into masks with labels to reconstructions using model.actual_lacuna_mask_and_label().
    # actual_lacuna_mask_and_label replaces all characters within [] with '_', strips '[' and ']', and adds embedding indices to self.labels for
    # lacuna/mask characters.

    for i, data_item in enumerate(dev_data):
        dev_data[i] = model.actual_lacuna_mask_and_label(data_item)
    for i, data_item in enumerate(test_data):
        test_data[i] = model.actual_lacuna_mask_and_label(data_item)

    if args.train:
        # if masking_strategy is "dynamic", utils.mask_input returns the data as passed to it, and mask = True
        # if masking_strategy is "once", utils.mask_input adds masking to data by calling model.mask_and_label_characters(),
        # and returns newly masked ata and mask = False.
        logging.info("Now training LSTM")
        # utils.mask_input expects train_data to be a list of objects of class greek_utils.DataItem.
        training_data, mask = utils.mask_input(
            model, train_data, mask_type, masking_strategy
        )
        # mask ultimately gets passed to greek_char_generator.train_batch(), to tell it whether to remask, which it should do if strategy is "dynamic"

        # train_model is in greek_char_generator module
        # if mask = True, train_model() will call model.mask_and_label_characters() to remask.
        model = train_model(
            model,
            train_data=training_data,
            dev_data=dev_data,
            output_name=model_name,
            mask=mask,  # true if masking_strategy is dynamic, false if masking_strategy is once
            mask_type=mask_type,
            seed=seed,
        )

    if args.eval:
        # run model on test set
        # random_test_data, _ = utils.mask_input(model, test_json, "random", "once") # I don't think we need to mask test_data
        # random_test_list = [i for i in range(len(random_test_data))]
        # smart_test_data, _ = utils.mask_input(model, test_json, "smart", "once")
        # smart_test_list = [i for i in range(len(smart_test_data))]

        # logging.info("Test Random:")
        # accuracy_evaluation(model, random_test_data, random_test_list)
        # baseline_accuracy(model, random_test_data, random_test_list)
        # logging.info("Test Smart:")
        # accuracy_evaluation(model, smart_test_data, smart_test_list)
        # baseline_accuracy(model, smart_test_data, smart_test_list)

        # load sentences
        # to eval pull from reconstructions according to the random_index entries corresponding to test_data

        file_path = f"{Path(__file__).parent}/data/" + test_json
        test_texts = utils.read_datafile(
            file_path
        )  # returns of a list of utils.DataItem objects

        test_indexes = []  # these indexes tell where to look in reconstructed_lacunae to get gold standard scholarly reconstructions
        with open(file_path, "r") as jsonfile:
            for line in jsonfile:
                jsonDict = json.loads(line)
                test_indexes.append(jsonDict["text_index"])

        logger.info(f"File {test_json} read in with {len(test_texts)} texts")

        reconstructed_lacunae = []

        with open(f"./data/{reconstruction_json}", "r") as jsonfile:
            for line in jsonfile:
                jsonDict = json.loads(line)
                reconstructed_lacunae.append(jsonDict["reconstructions"])

        test_reconstructions = []
        for i in test_indexes:
            test_reconstructions.append(reconstructed_lacunae[i])

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

        # make data_items and mask lacunae
        for i in range(len(test_texts)):
            test_texts[i].text_number = test_indexes[i]
        test_texts = [
            model.actual_lacuna_mask_and_label(test_texts[i])
            for i in range(len(test_texts))
        ]

        test_list = [i for i in range(len(test_texts))]
        # accuracy evaluation
        logging.info("Test Reconstructed:")
        accuracy_evaluation(model, test_texts, test_list)  # in greek_char_generator
        baseline_accuracy(model, test_texts, test_list)  # in greek_char_generator

    if args.predict:
        sentence = args.predict
        sentence = re.sub("<gap/>", "!", sentence)
        pattern = re.compile(r"\[.*?\]")
        sentence = pattern.sub(lambda x: x.group().replace(" ", ""), sentence)

        if not isinstance(sentence, str):
            logging.warning("Input to predict is not a string.")
        else:
            instance = DataItem(text=sentence)
            data_item = model.actual_lacuna_mask_and_label(instance)
            predict(model, data_item)

    if args.predict_top_k:
        k = 1000
        sentence = args.predict_top_k
        data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
        predict_top_k(model, data_item, k)

    if args.rank:
        # sentence = "ⲛⲛⲉⲧⲛⲟⲩⲱϣϥⲛⲟⲩⲕⲁⲥⲉⲃⲟⲗⲛϩⲏⲧϥⲉⲕⲉⲁⲥⲡⲉϫⲁϥⲛⲧⲉ###ⲛⲟⲩⲟⲩϣⲏⲛⲟⲩⲱⲧⲉⲧⲉⲧⲛⲉⲟⲩⲟⲙϥⲕⲁⲧⲁⲛⲉⲧⲙⲡⲁⲧⲣⲓⲁⲙⲛⲛⲉⲧⲛⲇⲏⲙⲟⲥ"
        # options = ["ⲩⲉⲓ", "ⲓϩⲉ", "ⲧⲉⲓ", "ⲉⲉⲉ", "ⲁⲁⲗ"]
        # sentence = "ⲁⲕⲛⲟϭⲛⲉϭⲡϫⲟⲉⲓⲥⲁⲕϫⲟⲟⲥϫⲉϩⲙⲡⲁϣⲁⲓⲛⲛϩⲁⲣⲙⲁϯ#####ⲉϩⲣⲁⲓⲉⲡϫⲓⲥⲉ"
        # options = ["ⲟⲁⲟⲟⲓ", "ⲛⲁⲟⲩⲉ", "ⲛⲁⲁⲗⲉ", "ⲙⲟⲟϣⲉ"]
        # options = ["ⲛⲁⲃⲱⲕ", "ⲛⲁⲁⲗⲉ", "ⲙⲟⲟϣⲉ"]
        sentence = "ⲁⲥⲡⲁⲍⲉⲙⲙⲟⲥⲁⲧⲉⲥ#####ⲛϩⲁϩⲛⲥⲟⲡ"
        # sentence = "ⲁⲥⲡⲁⲍⲉⲙⲙⲟⲥⲉⲧⲉⲥ#####ⲛϩⲁϩⲛⲥⲟⲡ"
        options = ["ϩⲏⲩⲉⲛ", "ⲧⲁⲡⲣⲟ", "ⲡⲁⲓϭⲉ", "ⲟⲩⲟϭⲉ", "ϭⲁⲗⲟϫ", "ⲧⲉϩⲛⲉ", "ϩⲟⲟⲉⲉ"]
        char_indexes = [ind for ind, ele in enumerate(sentence) if ele == "#"]
        ranking = rank(model, sentence, options, char_indexes)
        print("Ranking:")
        print("(option, log_sum)")
        for option in ranking:
            print(option)

    logger.info(f"end generator -- {datetime.now()}\n")
