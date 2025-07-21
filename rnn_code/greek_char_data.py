import regex as re
import unicodedata
import random
import json
import os

import greek_utils as utils

#RECONSTRUCTED_LACUNA_MAX = 100
# We may want to use this so that the  model avoids texts that are hopelessly lacunose

MASK = '_'
USER_MASK = '#'

def read_datafiles(json_list):
    '''
    This function takes in a list of files and looks at the Greek text in all of them.
    Sentences with reconstructed lacunae, sentences with empty lacunae, and sentences with no lacunae are separated out.
    Sentences without lacunae are split into train, test, dev (80:10:10). The reconstructed lacunae are then split, and some are masked.
    Function returns a tuple of lists of sentences (train, dev, test, no lacunae, unmasked lacunae, masked lacunae.)
    '''

    json_blocks = []
    mapping_greek_to_original = {} # maps position of text in list of all Greek texts to position of text in list of all texts
    i = 0
    for json_file in json_list:
        with open(json_file, "r") as file:
            for line in file:
                jsonDict = json.loads(line)
                if jsonDict["language"] == "grc":
                    mapping_greek_to_original[len(json_blocks)] = [i]
                    json_blocks.append(jsonDict)                    
                i += 1 

    training_texts = []
    lacuna_texts = []

    # in training_texts, all reconstructions are folded into text, missing characters are left as '.', and gaps of unknown length are '!'
    # in lacuna_texts, all reconstructions are converted to the format [# * number of missing characters], and gaps of unkown length are '!'
    # in lacuna_texts, whitespace is eliminated within lacunae
    # trianing_texts loses '[' and ']', lacuna_texts keeps them
    for i, block in enumerate(json_blocks):
        training_texts.append(block["training_text"].translate({91: None}).translate({93: None}))
        training_texts[i] = re.sub("<gap/>", "!", training_texts[i])
        lacuna_texts.append(re.sub("<gap/>", "!", block["training_text"]))
        pattern = re.compile(r"\[.*?\]")
        lacuna_texts[i] = pattern.sub(lambda x: x.group().replace(" ", ""), lacuna_texts[i])

    # calculate lengths for each partition based on ratios (80:10:10)
    n_texts = len(training_texts)
    train_length = int(n_texts * 0.8)
    dev_test_length = int(n_texts * 0.1)

    # partition the list: random_index will keep track of where text stands in original json file
    random_index = [i for i in range(n_texts)]
    random.shuffle(random_index)

    # dev_data and test_data are coming from lacuna_texts because the masking is actual lacunae, and character generation is evaluated against multiple reconstructions
    train_data = [training_texts[i] for i in random_index[: train_length]]
    dev_data = [lacuna_texts[i] for i in random_index[train_length: train_length + dev_test_length]]
    test_data = [lacuna_texts[i] for i in random_index[train_length + dev_test_length: ]]

    # create a list (for each text in json_blocks) of tuples of format ("abcd[##]ghi[#]kl", ["ef", "j"])
    # In the few (7) cases where a lacuna in a Greek text has multiple available reconstructions, the first one is used
    # if a text contains no lacunae, then in reconstructions it stands as an empty list

    # entries of the form: "alternatives": [" "] are included as "_" 
    reconstructions = [] 

    for i, block in enumerate(json_blocks):
        lacunae = []
        for case in block["test_cases"]:
            if case["alternatives"][0] == " ":
                lacunae.append("_")
            else:
                lacunae.append(case["alternatives"][0])
        reconstructions.append(lacunae) 

    # Return includes json_blocks (with alternate readings) and random_index 
    # random_index is in same order as train_data + dev_data + test_data, and lists indexes to json_blocks
    # json_blocks contains texts in the order they stand in the json files, and the json files in the order in which they stand in the file_list argument
    return (
        train_data,
        dev_data,
        test_data,
        random_index,
        reconstructions,
        mapping_greek_to_original
    )

def write_to_json(file_name: str, text_list: list, text_index: list, mapping_greek_to_original: dict):
    path = f"{__file__}/data/{file_name}"
    os.makedirs(os.path.dirname(path), exist_ok = True) 
    with open(f"{__file__}/data/{file_name}", "w") as jsonfile:
        for i, text in enumerate(text_list):
            json_block = {"text_index": text_index[i], "text": text, "position_in_original": mapping_greek_to_original[text_index[i]]}
            json_line = json.dumps(json_block, ensure_ascii = False)
            jsonfile.write(json_line + "\n")

    
