# Description

import csv
import json
import logging
import os
import random
import time
from math import log
from pathlib import Path
from random import shuffle
from typing import List, Tuple, cast

import numpy
import torch
import torch.nn.functional as nnf
from torch import nn

import rnn_code.greek_utils as utils
import wandb
from rnn_code.greek_rnn import RNN
from rnn_code.greek_utils import (
    DataItem,
    L2_lambda,
    batch_size,
    batch_size_multiplier,
    device,
    learning_rate,
    model_path,
    nEpochs,
    patience,
)

logger = logging.getLogger(__name__)


def check_accuracy(target: list[int], orig_data_item: DataItem) -> tuple[int, int, int]:
    masked = 0
    correct = 0
    mismatch = 0  # We don't want to skip sentences with a restoration of different length than the original text

    if orig_data_item.labels is None:
        raise ValueError(
            "check_accuracy must be called with DataItem with initialized labels"
        )

    # DataItem should have label[i] = -100 for unmasked tokens, otherwise the label should give the embedding index for the masked token
    if len(target) != len(orig_data_item.labels):
        logger.info("Model predicted different number of characters - text skipped")
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
    model: nn.Module,
    optimizer: torch.optim.optimizer.Optimizer,
    criterion: nn.Module,
    data: list[DataItem],
    data_indexes: list[int],
    mask_type: str,
    update: bool = True,  # update=False is used for dev set
    mask: bool = True,
) -> tuple[float, int, int, int, int, int]:
    model.zero_grad()
    total_loss, total_tokens, total_chars = 0, 0, 0

    train_masked = 0

    dev_masked = 0
    dev_correct = 0

    for i in data_indexes:
        data_item = data[i]

        if mask:
            data_item, _ = model.mask_and_label_characters(
                data_item, mask_type=mask_type
            )

        # Ensure indexes are populated even if not masking dynamically
        if data_item.indexes is None:
            data_item.indexes = model.lookup_indexes(data_item.text)

        index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
        label_tensor = torch.tensor(data_item.labels, dtype=torch.int64).to(device)
        out = model([index_tensor])  # [:-1]
        # splice out just predicted indexes to go into loss
        masked_idx = torch.BoolTensor(data_item.mask)
        # loss = criterion(out[0], label_tensor.view(-1))  # [1:] old loss method

        train_masked += torch.numel(masked_idx)  # to find the average loss

        masked_out = out[0, masked_idx]
        masked_label = label_tensor[masked_idx]
        loss = criterion(masked_out, masked_label)

        try:
            total_loss += loss.item()
        except RuntimeError as e:
            raise RuntimeError(
                f"Error accessing loss as scalar, criterion probably not applying reduction to loss tensor: {e}"
            )
        total_tokens += len(out[0])
        if data_item.text is None:
            raise ValueError("DataItem passed to train_batch has uninitialized text")
        total_chars += len(data_item.text)

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
            dev_masked_current, dev_correct_current, _ = check_accuracy(
                target, data_item
            )
            dev_masked += dev_masked_current
            dev_correct += dev_correct_current

    if update:
        optimizer.step()

    return total_loss, total_tokens, total_chars, train_masked, dev_masked, dev_correct


def train_model(
    model: RNN,
    train_data: list[DataItem],
    dev_data=None,
    output_name: str = "greek_lacuna",
    mask: bool = True,
    mask_type: str | None = None,
    seed=None,
):
    if mask_type is None:
        raise ValueError(
            "mask_type from [random, smart] must be specified for training from"
        )
    if seed is not None:
        random.seed(seed)
    # start a new wandb run to track this script
    wandb.init(
        project="greek_rnn",
        config={
            "learning_rate": learning_rate,
            "architecture": "LSTM",
            "dataset": "MAAT",
            "epochs": nEpochs,
            "batch_size": batch_size,
            "patience": patience,
            "mask_type": mask_type,
            "embed_size": model.specs[0],
            "hidden_size": model.specs[1],
            "rnn_layers": model.specs[3],
            "dropout": model.specs[5],
            "masking_proportion": model.specs[6],
        },
    )

    # Move model to device
    model = model.to(device)

    train_data = train_data[:]  # to avoid modifying list passed to train_model()
    if dev_data is None:
        data_set = train_data
        shuffle(data_set)
        num_dev_items = min(int(0.05 * len(train_data)), 2000)
        dev_data = data_set[:num_dev_items]
        train_data = data_set[num_dev_items:]
    else:
        dev_data = dev_data[:]

    # Convert train_data strings to DataItem objects
    processed_train_data = []
    for text in train_data:
        if isinstance(text, DataItem):
            # Already a DataItem
            data_item = text
        else:
            # String, create DataItem
            data_item = DataItem(text=text)
        processed_train_data.append(data_item)
    train_data = processed_train_data

    # Convert dev_data to DataItem objects with processed lacunae (once at start)
    processed_dev_data = []
    for item in dev_data:
        if isinstance(item, DataItem):
            # Already a DataItem, check if already processed
            if hasattr(item, "mask") and item.mask is not None:
                # Already processed, use as-is
                processed_item = item
            else:
                # Process lacunae
                processed_item = model.actual_lacuna_mask_and_label(item)
        else:
            # String or dict, create DataItem first
            if isinstance(item, str):
                data_item = DataItem(text=item)
            else:
                data_item = DataItem(
                    text=item.text if hasattr(item, "text") else str(item)
                )
            processed_item = model.actual_lacuna_mask_and_label(data_item)
        processed_dev_data.append(processed_item)
    dev_data = processed_dev_data

    # Now whether dev_data was passed or not, there's a train_data list object and a dev_data list object (DataItems)
    train_list = [i for i in range(len(train_data))]
    dev_list = [i for i in range(len(dev_data))]

    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.adamw.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=L2_lambda
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    bs = batch_size

    # Early stopping and best model tracking
    best_dev_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    best_epoch = 0

    for epoch in range(nEpochs):
        if epoch > 0:
            bs *= batch_size_multiplier
        incremental_batch_size = int(bs + 0.5)
        shuffle(train_data)

        model.train()
        train_loss, train_tokens, train_chars, train_mask_count = 0, 0, 0, 0
        for i in range(0, len(train_data), incremental_batch_size):
            loss, num_tokens, num_characters, train_masked, _, _ = train_batch(
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
            train_mask_count += train_masked

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
            mask=False,  # Dev set is already masked, since lacunas are masked.
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

        train_loss = train_loss / train_tokens
        dev_loss = dev_loss / dev_tokens
        dev_accuracy = dev_correct / dev_masked if dev_masked > 0 else 0.0

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "dev_accuracy": dev_accuracy,
            }
        )

        # Early stopping with patience and best model checkpointing
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"New best dev loss: {dev_loss:.6f} at epoch {epoch}")
            # Save best model
            Path(model_path).mkdir(parents=True, exist_ok=True)
            torch.save(model, f"{model_path}/{output_name}_best.pth")
        else:
            patience_counter += 1
            logger.info(
                f"Dev loss did not improve. Patience: {patience_counter}/{patience}"
            )

        # Check if we should stop early
        if patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch}. Best dev loss: {best_dev_loss:.6f} at epoch {best_epoch}"
            )
            break

        logger.info(
            f"dev masked total: {dev_masked}, correct predictions: {dev_correct}, simple accuracy: {round(dev_correct / dev_masked, 3)}"
        )

        # Save current model (for debugging/backup)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(model, f"{model_path}/{output_name}_latest.pth")

        # sample_masked = 0
        # sample_correct = 0
        #
        # test_sentence = "ϯⲙⲟⲕⲙⲉⲕⲙⲙⲟⲓⲉⲓⲥϩⲉⲛⲣⲟⲙⲡⲉⲉⲧⲙⲧⲣⲉⲣⲱⲙⲉϭⲛϣⲁϫⲉⲉϫⲱⲕⲁⲧⲁⲗⲁⲁⲩⲛⲥⲙⲟⲧ·"
        # test_sentence = utils.filter_diacritics(test_sentence)
        # _, masked, correct = fill_masks(model, test_sentence, mask_type, temp=0)
        # sample_masked += masked
        # sample_correct += correct
        #
        # logger.info(f"sample accuracy: {round(sample_correct/sample_masked, 3)}")

    # Restore best model state if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(
            f"Restored best model from epoch {best_epoch} with dev loss {best_dev_loss:.6f}"
        )
        # Save final best model
        Path(model_path).mkdir(parents=True, exist_ok=True)
        torch.save(model, f"{model_path}/{output_name}.pth")
    else:
        logger.warning("No best model state found - using final epoch model")

    # dev_data is being passed in here in the same format as training data is passed to RNN model
    accuracy_evaluation(model, dev_data, dev_list)
    # baseline_accuracy(model, dev_data, dev_list)

    return model


def fill_masks(model: RNN, text: str, mask_type: str, temp=0):
    logger.info(f"prompt: {text}")
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
    if not isinstance(data_item.mask, list):
        raise ValueError("data_item.mask not initalized")
    if not isinstance(data_item.labels, list):
        raise ValueError("data_item.labels not initialized")
    for i in range((len(data_item.mask))):
        if data_item.mask[i]:
            pairs.append((model.decode(data_item.labels[i]), model.decode(target[i])))
            pairs_index.append((data_item.labels[i], target[i]))
    logger.info(f"orig vs predicted char: {pairs}")
    logger.info(f"orig vs predicted char: {pairs_index}")

    sample_masked, sample_correct, _ = check_accuracy(target, test_data_item)
    return target_text, sample_masked, sample_correct


# this should be able to evaluate both on dev set and test set
def accuracy_evaluation(model: RNN, data: List[DataItem], data_indexes):
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
        logger.info(
            f"masked total: {masked_total}, correct predictions: {correct}, simple accuracy: {round(correct / masked_total, 3)}, mismatch: {mismatch_total}"
        )
    else:
        logger.info(
            f"masked total: {masked_total}, correct predictions: {correct}, mismatch {mismatch_total}"
        )


# this is only for evaluating on the test set
def baseline_accuracy(model: RNN, data: list[DataItem], data_indexes: list[int]):
    masked_total = 0
    correct_most_common_char = 0
    correct_random = 0
    correct_trigram = 0
    # ο is the most common character in MAAT corpus
    target_char_index = model.token_to_index["ο"]
    # Load tri-gram look-up if already constructed, else construct it
    try:
        with open(f"{Path(__file__).parent}/data/trigram_lookup.json", "r") as file:
            trigram_lookup = json.load(file)
    except FileNotFoundError:
        # return a dictionary of format {bigram: {char1: count, char2: count, ...}} that tells how often each character completes the trigram
        trigram_lookup = utils.construct_trigram_lookup()
    count_rand = 0
    for i in data_indexes:
        data_item = data[i]
        if data_item.labels is None:
            raise ValueError(
                "data passed to baseline_accuracy contains DataItem with unitialized labels"
            )
        most_common_char_target = [target_char_index] * len(data_item.labels)
        random_target = [
            random.randint(3, model.num_tokens - 1)
            for _ in range(len(data_item.labels))
        ]
        # make trigram prediction target
        trigram_target = []
        for j in range(len(data_item.labels)):
            if data_item.labels[j] > 0:
                # if label is above 0, use trigram lookup
                if len(trigram_target) >= 2:
                    look_up_key = model.decode([trigram_target[-2]]) + model.decode(
                        [trigram_target[-1]]
                    )
                elif len(trigram_target) == 1:
                    look_up_key = "<s>" + model.decode([trigram_target[-1]])
                else:
                    look_up_key = "<s><s>"
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
                if data_item.indexes is None:
                    raise ValueError(
                        "data passed to baseline_accuracy contains DataItem with unitialized indexes"
                    )
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
    logger.info(
        f"Most Common Char Baseline; dev masked total: {masked_total}, correct predictions: {correct_most_common_char}, baseline accuracy: {round(correct_most_common_char / masked_total, 3)}"
    )
    logger.info(
        f"Random Baseline; dev masked total: {masked_total}, correct predictions: {correct_random}, baseline accuracy: {round(correct_random / masked_total, 3)}"
    )
    logger.info(
        f"Trigram Baseline; dev masked total: {masked_total}, correct predictions: {correct_trigram}, baseline accuracy: {round(correct_trigram / masked_total, 3)}"
    )


def predict(model: RNN, data_item: DataItem):
    if data_item.indexes is None:
        raise ValueError("data_item passed to predict has unitialized indexes")
    if data_item.mask is None:
        raise ValueError("data_item passed to predict has unitialized mask")

    logger.info(f"input text: {data_item.text}")

    index_tensor = torch.tensor(data_item.indexes, dtype=torch.int64).to(device)
    out = model([index_tensor])

    # get target indexes
    target = []
    for emb in out[0]:
        scores = emb
        _, best = scores.max(0)
        best = best.data.item()
        target.append(best)

    # Build output text by replacing masked positions with predictions
    output_chars = []
    for i, token_index in enumerate(data_item.indexes):
        if i < len(data_item.mask) and data_item.mask[i]:
            # This position was masked, use the model's prediction
            predicted_char = model.index_to_token[target[i]]
            output_chars.append(predicted_char)
        else:
            # This position was not masked, use the original character
            original_char = model.index_to_token[token_index]
            output_chars.append(original_char)

    # Remove control characters (BOT/EOT)
    output_text = "".join(output_chars)
    output_text = output_text.replace(model.bot_char, "").replace(model.eot_char, "")

    logger.info(f"output text: {output_text}")
    return output_text


# file names for csv will be automatically generated from timestamp if save_to_file=True and output_file=None
def predict_top_k(
    model: RNN,
    data_item: DataItem,
    k: int = 10,
    save_to_file: bool = True,
    output_file: str | os.PathLike | None = None,
):
    if data_item.mask is None:
        raise ValueError("DataItem passed to predict_top_k has uninitialized mask")
    # beam search
    logger.info(f"input text: {data_item.text}")

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
    for i in range(len(data_item.mask)):
        if i >= len(target_candidates):
            logger.warning(
                f"Mask array longer than target_candidates: mask[{len(data_item.mask)}] vs target_candidates[{len(target_candidates)}]"
            )
            continue
        if data_item.mask[i]:
            lacuna_candidates.append(target_candidates[i])

    if not lacuna_candidates:
        logger.warning(
            "No lacuna positions found for top-k prediction for data_item with text number {data_item.text_number}, and text {data_item.text}\n"
        )
        return []

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

    if save_to_file:
        # Generate filename if not provided
        if output_file is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"{Path(__file__).parent}/results/top_k_{timestamp}.csv"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Rank", "Candidate", "LogSum"])  # Write header
            for index, seq_value in enumerate(top_k):
                seq = seq_value[0]
                value = seq_value[1]
                lacuna_string = model.decode(seq)
                return_list.append(lacuna_string)
                writer.writerow([index + 1, lacuna_string, value])
    else:
        # No file writing, just build return list
        for index, seq_value in enumerate(top_k):
            seq = seq_value[0]
            lacuna_string = model.decode(seq)
            return_list.append(lacuna_string)

    return return_list


def rank(model: RNN, sentence: str, options: List[str]) -> List[Tuple[str, float]]:
    # filter diacritics
    sentence = utils.filter_diacritics(sentence)
    data_item = DataItem(sentence)
    print(data_item.text)
    data_item = model.actual_lacuna_mask_and_label(data_item)
    char_indexes = [
        ind for ind, ele in enumerate(cast(str, data_item.text)) if ele == "_"
    ]
    print(data_item.text)
    # adjust char indexes for padding of data item
    char_indexes = [x + 2 for x in char_indexes]
    option_indexes = []
    option_probs = []
    for i in range(len(options)):
        options[i] = utils.filter_diacritics(options[i])
        opt_i_indexes = model.lookup_indexes(options[i], add_control=False)
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
