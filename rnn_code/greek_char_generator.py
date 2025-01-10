# Description

import time
from math import log
import json
import csv
import random
from random import shuffle
import logging
import re
import itertools

import torch
import numpy
import torch.nn.functional as nnf
from torch import nn
import wandb

from rnn_code.greek_rnn import *
from rnn_code import greek_utils as utils
from rnn_code.greek_utils import *



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def check_accuracy(target, orig_data_item):
    masked = 0
    correct = 0
    mismatch = 0 # We don't want to skip sentences with a restoration of different length than the original text

    # DataItem should have label[i] = -100 for unmasked tokens, otherwise the label should give the embedding index for the masked token
    if len(target) != len(orig_data_item.labels):
        logging.info(
            "Model predicted different number of characters - text skipped"
        )
        mismatch += 1
    else:
        for j in range(len(orig_data_item.labels)):
            if orig_data_item.labels[j] > 0:
                # masked token
                masked += 1
                if target[j] == orig_data_item.labels[j]:
                    # prediction is correct
                    correct += 1

    # masked is the # of masked tokens in input, correct is the number of predictions that much the masked token, and 
    # mismatch returns 1 if model returned different number of characters than there were characters in the lacunae 
    return masked, correct, mismatch


def train_batch(
    model,
    optimizer,
    criterion,
    data,
    data_indexes,
    mask_type,
    update=True, # update=False is used for dev set
    mask=True
):
    model.zero_grad()
    total_loss, total_tokens, total_chars = 0, 0, 0

    total_masked = 0

    dev_masked = 0
    dev_correct = 0

    for i in data_indexes:
        data_item = data[i]

        if mask:
            data_item, mask_count = model.mask_and_label_characters(
                data_item, mask_type=mask_type
            )
            total_masked += mask_count

        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        label_tensor = torch.tensor(data_item.labels, dtype=torch.int64).to(device)
        out = model([index_tensor])  # [:-1]
        # splice out just predicted indexes to go into loss
        masked_idx = torch.BoolTensor(data_item.mask)
        # loss = criterion(out[0], label_tensor.view(-1))  # [1:] old loss method

        masked_out = out[0, masked_idx]
        masked_label = label_tensor[masked_idx]
        loss = criterion(masked_out, masked_label)

        total_loss += loss.data.item()
        total_tokens += len(out[0])
        total_chars += len(data_item.text) + 1

        if update:
            loss.backward()
        else:
            target = []
            for emb in out[0]:
                scores = emb
                _, best = scores.max(0)
                best = best.data.item()
                target.append(best)

            # compare target to label
            # logger.debug(f"self attn labels: {data_item.labels}")
            # logger.debug(f"target labels: {target}")
            # logger.info("No update")
            dev_masked_current, dev_correct_current, _ = check_accuracy(target, data_item)
            dev_masked += dev_masked_current
            dev_correct += dev_correct_current

    if update:
        optimizer.step()

    total_masked += dev_masked

    return total_loss, total_tokens, total_chars, total_masked, dev_masked, dev_correct


def train_model(
    model,
    train_data,
    dev_data=None, 
    output_name="greek_lacuna",
    mask=True,
    mask_type=None
):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="greek_rnn",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "LSTM",
        "dataset": "MAAT",
        "epochs": nEpochs,
        }
    )
    
    train_data = train_data[:] # to avoid modifying list passed to train_model()
    if dev_data is None:
        data_set = train_data 
        shuffle(data_set)
        num_dev_items = min(int(0.05 * len(train_data)), 2000)
        dev_data = data_set[:num_dev_items]
        train_data = data_set[num_dev_items:]
    else:
        dev_data = dev_data[:] 
        
    # Now whether dev_data was passed or not, there's a train_data list object and a dev_data list object
    train_list = [i for i in range(len(train_data))]
    dev_list = [i for i in range(len(dev_data))]

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=L2_lambda
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    bs = batch_size
    prev_dev_loss = 0
    prev_train_loss = 0
    prev_prev_dev_loss = 0
    prev_prev_train_loss = 0

    for epoch in range(nEpochs):
        if epoch > 0:
            bs *= batch_size_multiplier
        incremental_batch_size = int(bs + 0.5)
        shuffle(train_data)

        model.train()
        train_loss, train_tokens, train_chars, train_mask_count = 0, 0, 0, 0
        for i in range(0, len(train_data), incremental_batch_size):
            loss, num_tokens, num_characters, total_masked, _, _ = train_batch(
                model,
                optimizer,
                criterion,
                train_data,
                train_list[i : i + incremental_batch_size],
                mask_type,
                update=True,
                mask=mask,
            )

            logger.debug(f"masked total: {train_mask_count}")

            train_loss += loss
            train_tokens += num_tokens
            train_chars += num_characters
            train_mask_count += total_masked
            wandb.log({"Epoch": epoch, "avg loss in epoch": train_loss/train_chars, "batch_size": incremental_batch_size})

            if num_characters > 0:
                logger.debug(
                    f"{epoch:4} {i:6} {num_tokens:5} {num_characters:6} loss {loss / num_tokens:7.3f} {loss / num_characters:7.3f} -- tot tr loss: {train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
                )

        model.eval()
        (
            dev_loss,
            dev_tokens,
            dev_chars,
            dev_masked,
            dev_masked,
            dev_correct,
        ) = train_batch(
            model,
            optimizer,
            criterion,
            dev_data,
            dev_list,
            mask_type,
            update=False,
            mask=False, # Dev set is already masked, since lacunas are masked. 
        )

        if epoch == 0:
            logger.info(
                f"train={len(train_list):,} {train_tokens:,} {train_chars:,} {train_chars / train_tokens:0.1f} "
                f"dev={len(dev_list):,} {dev_tokens:,} {dev_chars:,} {dev_chars / dev_tokens:0.1f} "
                f"bs={batch_size} lr={learning_rate} {model.specs}"
            )

        msg_trn = f"{train_loss / train_tokens:8.4f} {train_loss / train_chars:8.4f}"
        msg_dev = f"{dev_loss / dev_tokens:8.4f} {dev_loss / dev_chars:8.4f}"
        logger.info(
            f"{epoch} tr loss {msg_trn} -- dev loss {msg_dev} -- incremental_batch_size: {incremental_batch_size:4} time elapsed: {time.time() - start:6.1f}"
        )

        train_loss = train_loss / train_chars
        dev_loss = dev_loss / dev_chars

        wandb.log({"dev loss after epoch": dev_loss})

        if train_loss < prev_train_loss and dev_loss > prev_dev_loss:
            logger.info("early exit")
            break
        
        if epoch > 0:
            prev_prev_dev_loss = prev_dev_loss
            prev_prev_train_loss = prev_train_loss
        prev_dev_loss = dev_loss
        prev_train_loss = train_loss

        logging.info(
            f"dev masked total: {dev_masked}, correct predictions: {dev_correct}, simple accuracy: {round(dev_correct / dev_masked, 3)}"
        )
        torch.save(model, f"{model_path}/{output_name}.pth")

        # sample_masked = 0
        # sample_correct = 0
        #
        # test_sentence = "ϯⲙⲟⲕⲙⲉⲕⲙⲙⲟⲓⲉⲓⲥϩⲉⲛⲣⲟⲙⲡⲉⲉⲧⲙⲧⲣⲉⲣⲱⲙⲉϭⲛϣⲁϫⲉⲉϫⲱⲕⲁⲧⲁⲗⲁⲁⲩⲛⲥⲙⲟⲧ·"
        # test_sentence = utils.filter_diacritics(test_sentence)
        # _, masked, correct = fill_masks(model, test_sentence, mask_type, temp=0)
        # sample_masked += masked
        # sample_correct += correct
        #
        # logging.info(f"sample accuracy: {round(sample_correct/sample_masked, 3)}")

    # dev_data is being passed in here in the same format as training data is passed to RNN model
    accuracy_evaluation(model, dev_data, dev_list)
    # baseline_accuracy(model, dev_data, dev_list)

    return model


def fill_masks(model, text, mask_type, temp=0):
    logging.info(f"prompt: {text}")
    test_data_item = DataItem(text=text)
    data_item, _ = model.mask_and_label_characters(test_data_item, mask_type=mask_type)
    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    sample_out = model([index_tensor])

    target = []
    for emb in sample_out[0]:
        scores = emb  # [0,-1]
        if temp <= 0:
            _, best = scores.max(0)
            best = best.data.item()
        else:
            output_dist = nn.functional.softmax(scores.view(-1).div(temp))  # .exp()
            best = torch.multinomial(output_dist, 1)[0]
            best = best.data.item()
        target.append(best)

    target_text = model.decode(target)

    # input vs masked pairs
    pairs = []
    pairs_index = []
    for i in range((len(data_item.mask))):
        if data_item.mask[i]:
            pairs.append((model.decode(data_item.labels[i]), model.decode(target[i])))
            pairs_index.append((data_item.labels[i], target[i]))
    logging.info(f"orig vs predicted char: {pairs}")
    logging.info(f"orig vs predicted char: {pairs_index}")

    sample_masked, sample_correct, _ = check_accuracy(target, test_data_item)
    return target_text, sample_masked, sample_correct

# this should be able to evaluate both on dev set and test set
def accuracy_evaluation(model, data, data_indexes):
    # first pass at simple accuracy function
    masked_total = 0
    correct = 0
    mismatch_total = 0

    for i in data_indexes:
        # get model output
        data_item = data[i]
        # 
        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        out = model([index_tensor])

        # get target indexes
        target = []
        for emb in out[0]:
            scores = emb
            _, best = scores.max(0)
            best = best.data.item()
            target.append(best)

        masked, correct_guess, mismatch = check_accuracy(target, data_item)
        masked_total += masked
        correct += correct_guess
        mismatch_total += mismatch

    if masked_total > 0:
        logging.info(
            f"masked total: {masked_total}, correct predictions: {correct}, simple accuracy: {round(correct/masked_total, 3)}, mismatch: {mismatch_total}"
        )
    else:
        logging.info(
            f"masked total: {masked_total}, correct predictions: {correct}, mismatch {mismatch_total}"
        )

# this is only for evaluating on the test set
# everything involving trigrams needs work
def baseline_accuracy(model, data, data_indexes):
    masked_total = 0
    correct_most_common_char = 0
    correct_random = 0
    correct_trigram = 0
    # ο is the most common character in MAAT corpus
    target_char_index = model.token_to_index["ο"] 
    # Load tri-gram look-up if already constructed, else construct it
    try:
        with open('data/trigram_lookup.json', 'r') as file:
            trigram_lookup = json.load(file)
    except:
        # return a dictionary of format {bigram: {char1: count, char2: count, ...}} that tells how often each character completes the trigram
        trigram_lookup = utils.construct_trigram_lookup()
    count_rand = 0
    for i in data_indexes:
        data_item = data[i]
        most_common_char_target = [target_char_index] * len(data_item.labels)
        random_target = [
            random.randint(3, model.num_tokens - 1)
            for i in range(len(data_item.labels))
        ]
        # make trigram prediction target
        trigram_target = []
        for j in range(len(data_item.labels)):
            if data_item.labels[j] > 0:
                # if label is above 0, use trigram lookup
                if len(trigram_target) >= 2:
                    look_up_key = model.decode([trigram_target[-2]]) + model.decode([trigram_target[-1]])
                elif len(trigram_target) == 1:
                    look_up_key = '<s>' + model.decode([trigram_target[-1]])
                else:
                    look_up_key = '<s><s>'
                if look_up_key in trigram_lookup:
                    y = trigram_lookup[look_up_key]
                    greek_char = max(y, key=lambda x: y[x])
                    greek_char_index = model.token_to_index[greek_char]
                    trigram_target.append(greek_char_index)
                else:
                    # if trigram lookup fails, resort to random
                    count_rand += 1
                    trigram_target.append(random.randint(3, model.num_tokens - 1))
            else:
                # if label is 0, keep what is in the data item
                trigram_target.append(data_item.indexes[j])

        _, correct_guess_correct_most_common, _ = check_accuracy(
            most_common_char_target, data_item
        )
        masked, correct_guess_random, _ = check_accuracy(random_target, data_item)
        _, correct_guess_trigram, _ = check_accuracy(trigram_target, data_item)
        masked_total += masked
        correct_most_common_char += correct_guess_correct_most_common
        correct_random += correct_guess_random
        correct_trigram += correct_guess_trigram
    #print(count_rand)
    logging.info(
        f"Most Common Char Baseline; dev masked total: {masked_total}, correct predictions: {correct_most_common_char}, baseline accuracy: {round(correct_most_common_char / masked_total, 3)}"
    )
    logging.info(
        f"Random Baseline; dev masked total: {masked_total}, correct predictions: {correct_random}, baseline accuracy: {round(correct_random / masked_total, 3)}"
    )
    logging.info(
        f"Trigram Baseline; dev masked total: {masked_total}, correct predictions: {correct_trigram}, baseline accuracy: {round(correct_trigram / masked_total, 3)}"
    )


def predict(model, data_item):
    logging.info(f"input text: {data_item.text}")

    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    out = model([index_tensor])

    # get target indexes
    target = []
    for emb in out[0]:
        scores = emb
        _, best = scores.max(0)
        best = best.data.item()
        target.append(best)

    out_indexes = []

    # Get the characters generated by the model to fill lacunae
    gen_chars = (char for char in model.decode(itertools.compress(target, data_item.mask)))

    # Iterate through all sequences enclosed in square brackets
    pattern = re.compile(r"\[(.+?)\]")

    text_buffer = list(data_item.text)

    for lacuna in re.finditer(pattern, data_item.text):
        for i in range(len(lacuna.group(1))):
            try:
                text_buffer[lacuna.start()+i+1] = next(gen_chars)
            except:
                text_buffer[lacuna.start()+i+1] = '_'

    output_string = "".join(text_buffer)

    logging.info(f"output text: {output_string}")
    return output_string


def predict_top_k(model, data_item, k=10):
    # beam search
    logging.info(f"input text: {data_item.text}")

    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    out = model([index_tensor])

    # get target candidates
    target_candidates = []
    for emb in out[0]:
        scores = emb
        probabilities = nnf.softmax(scores, dim=0)
        vocabid_probs = []
        for i in range(len(probabilities)):
            vocabid_probs.append((i, probabilities[i]))
        sorted_vocabid_probs = sorted(vocabid_probs, key=lambda x: x[1], reverse=True)

        target_candidates.append(sorted_vocabid_probs)

    lacuna_candidates = []
    for i in range(len(data_item.indexes)):
        if data_item.indexes[i] == model.mask:
            lacuna_candidates.append(target_candidates[i])

    top_k = [[list(), 0.0]]
    # walk over each step in sequence
    for row in lacuna_candidates:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(top_k)):
            seq, score = top_k[i]
            for j in range(len(row)):
                candidate = [seq + [row[j][0]], score + log(row[j][1])]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        # select k best
        top_k = ordered[:k]

    return_list = []

    # write top k to file
    with open('top_k.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Rank', 'Candidate', 'LogSum'])  # Write header
        for index, seq_value in enumerate(top_k):
            seq = seq_value[0]
            value = seq_value[1]
            lacuna_string = model.decode(seq)
            return_list.append(lacuna_string)
            writer.writerow([index+1, lacuna_string, value])

    return return_list


def rank(model, sentence, options, char_indexes):
    # filter diacritics
    sentence = utils.filter_diacritics(sentence)
    data_item = model.actual_lacuna_mask_and_label(DataItem(), sentence)
    # adjust char indexes for padding of data item
    char_indexes = [x + 2 for x in char_indexes]
    option_indexes = []
    option_probs = []
    for i in range(len(options)):
        options[i] = utils.filter_diacritics(options[i])
        opt_i_indexes = model.lookup_indexes(options[i], add_control=False)[1:]
        option_indexes.append(opt_i_indexes)
        option_probs.append([])

    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    out = model([index_tensor])

    # get target indexes
    target = []
    char_index = 0
    for emb in out[0]:
        scores = emb
        probabilities = nnf.softmax(scores, dim=0)
        _, best = scores.max(0)
        best = best.data.item()
        target.append(best)
        if char_index in char_indexes:
            current = char_indexes.index(char_index)
            for i in range(len(options)):
                option_index = option_indexes[i][current]
                prob = probabilities[option_index]
                option_probs[i].append(prob)
        char_index += 1

    option_log_sums = []
    for opt in option_probs:
        opt_log_sum = torch.sum(torch.log(torch.tensor(opt)))
        option_log_sums.append(opt_log_sum)
    option_log_sums = numpy.array(option_log_sums)
    sorted_index = numpy.argsort(option_log_sums)[::-1]
    ranking = []
    for index in sorted_index:
        ranking.append((options[index], option_log_sums[index]))
    return ranking
