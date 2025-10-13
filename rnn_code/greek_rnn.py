import random

import torch
import torch.nn as nn

import rnn_code.letter_tokenizer as letter_tokenizer
from rnn_code.greek_utils import DataItem

# MASK = "<mask>"
MASK = "_"
USER_MASK = "#"


class RNN(nn.Module):
    def __init__(self, specs, spaces_are_tokens=False, newlines_are_tokens=True):
        super(RNN, self).__init__()
        # ensure that character dictionary doesn't change
        self.tokenizer = letter_tokenizer.LetterTokenizer(
            spaces_are_tokens, newlines_are_tokens
        )
        self.unk_char = "?"
        self.bot_char = "<"
        self.eot_char = ">"
        self.mask_char = MASK
        self.user_mask_char = USER_MASK
        tokens = (
            "αοιεϲντυρμωηκπλδγχβθφξζψϛϚϜ"
            + ".?<>\n!"
            + self.mask_char
            + self.user_mask_char
        )
        if spaces_are_tokens:
            tokens += " "
        self.num_tokens = len(tokens)
        self.specs = specs + [self.num_tokens]

        (
            embed_size,
            hidden_size,
            proj_size,
            rnn_nLayers,
            self.share,
            dropout,
            masking_proportion,
        ) = specs

        self.token_to_index = {}
        self.index_to_token = {}
        for i, token in enumerate(tokens):
            self.token_to_index[token] = i
            self.index_to_token[i] = token

        self.embed = nn.Embedding(self.num_tokens, embed_size)
        self.masking_proportion = masking_proportion

        self.scale_up = nn.Linear(embed_size, hidden_size)

        self.rnn = nn.LSTM(
            hidden_size,
            int(hidden_size / 2),
            num_layers=rnn_nLayers,
            bidirectional=True,
            dropout=dropout,
            batch_first=True,
        )

        if not self.share:
            self.out = nn.Linear(hidden_size, embed_size)
        else:
            self.scale_down = nn.Linear(hidden_size, embed_size)

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)

    def forward(self, seqs):
        num_batches = len(seqs)
        num_tokens = len(seqs[0])
        seqs = torch.cat(seqs).view(num_batches, num_tokens)

        embed = self.embed(seqs)
        embed = self.dropout(embed)
        embed = self.scale_up(embed)

        output, _ = self.rnn(embed)
        output = self.dropout(output)

        if not self.share:
            output = self.out(output)
            output = torch.matmul(
                output, torch.t(self.embed.weight)
            )  # this was added as a fix
        else:
            # use embedding table as output layer
            output = self.scale_down(output)
            output = torch.matmul(output, torch.t(self.embed.weight))

        return output

    # Better for this to return list than tensor
    def lookup_indexes(self, text, add_control=True) -> list:
        text = self.tokenizer.tokenize(text)
        indexes = [self.token_to_index[token] for token in text]
        if add_control:
            indexes = (
                [self.token_to_index[self.bot_char]]
                + indexes
                + [self.token_to_index[self.eot_char]]
            )
        return indexes

    def decode(self, indexes) -> str:
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.type(torch.int64).tolist()
        text = "".join([self.index_to_token[index] for index in indexes])
        return text

    # Function is used in greek_utils: data_item is a DataItem object as defined in greek_utils
    def mask_and_label_characters(self, data_item, mask_type="random"):
        data_item.indexes = self.lookup_indexes(data_item.text)

        text_length = len(data_item.indexes)
        mask = [True] * text_length
        labels = (
            [-100] * text_length
        )  # labels mark where actual masking is with a value > 0, and this value is the index of the masked token

        mask_count = 0  # This counts only tokens actually masked, not those swapped for random character or retained
        random_sub = 0
        orig_token = 0

        if mask_type == "random":
            for i in range(text_length):
                current_token = data_item.indexes[i]
                r_mask_status = random.random()
                r_mask_type = random.random()

                if r_mask_status < self.masking_proportion:
                    if r_mask_type < 0.8:
                        # replace with MASK symbol
                        replacement = self.token_to_index[self.mask_char]
                        mask_count += 1
                    elif r_mask_type < 0.9:
                        # replace with random character
                        # The integers that index to the embeddings are in the closed interval [0, num_tokens - 1]
                        replacement = random.randint(0, self.num_tokens - 1)
                        random_sub += 1
                    else:
                        # retain original (these retained tokens are still True in mask list of booleans)
                        replacement = current_token
                        orig_token += 1

                    data_item.indexes[i] = replacement
                    labels[i] = current_token

                else:
                    mask[i] = False

                data_item.mask = mask
                data_item.labels = labels

        elif mask_type == "smart":
            r_mask_quantity = random.randint(1, 5)

            mask_index = [0] * text_length
            i = 0

            while i < r_mask_quantity:
                r_start_loc = random.randint(0, text_length)
                r_mask_length = random.random()
                if r_mask_length <= 0.48:
                    mask_length = 1
                elif 0.48 < r_mask_length <= 0.70:
                    mask_length = 2
                elif 0.70 < r_mask_length <= 0.82:
                    mask_length = 3
                else:
                    mask_length = random.randint(4, 35)
                mask_end = r_start_loc + mask_length
                mask_type = random.random()
                mask_index[r_start_loc:mask_end] = [mask_type] * mask_length
                i += 1

            mask_start = 0
            for i in range(text_length):
                current_token = data_item.indexes[i]
                if mask_index[i] > 0:
                    if mask_index[i] < 0.8:
                        # replace with MASK symbol
                        replacement = self.token_to_index[self.mask_char]
                        mask_count += 1
                    elif mask_index[i] < 0.9:
                        # replace with random character
                        replacement = random.randint(0, self.num_tokens - 1)
                        random_sub += 1
                    else:
                        # retain original
                        replacement = current_token
                        orig_token += 1

                    data_item.indexes[i] = replacement
                    labels[i] = current_token

                    mask_start += 1
                else:
                    mask[i] = False

                data_item.mask = mask
                data_item.labels = labels

        total_mask = mask_count + random_sub + orig_token

        return (
            data_item,
            total_mask,
        )  # Total mask also includes tokens swapped for random character, and tokens retained

    # This is the function to use to mask and label the test set incorporating the actual lacunae as masks
    def actual_lacuna_mask_and_label(self, data_item: DataItem) -> DataItem:
        if not isinstance(data_item.text, str):
            raise TypeError(
                "DataItem passed to actual_lacuna_mask_and_label must a string associated with it as text attribute"
            )
        text_buffer = []

        mask = []
        labels = []
        in_lacuna = False
        mask.append(False)  # For BOT character, '<'
        labels.append(-100)

        for i in range(len(data_item.text)):
            if data_item.text[i] == "[":
                in_lacuna = True
                continue
            if data_item.text[i] == "]":
                in_lacuna = False
                continue
            tokenized = [char for char in self.tokenizer.tokenize(data_item.text[i])]
            if len(tokenized) > 1:  # handles iota subscripts
                text_buffer += tokenized
            else:
                text_buffer += data_item.text[i]
            # Skip any characters not recognized by the tokenizer, so that data_item.mask lines up with data_item.indexes
            if len(tokenized) == 0:
                continue
            if in_lacuna:
                text_buffer[-len(tokenized) :] = MASK * len(tokenized)
                mask += [True] * len(tokenized)
                labels += [self.token_to_index[token] for token in tokenized]
            else:
                mask += [False] * len(
                    tokenized
                )  # This handles characters with and without subscripts
                labels += [-100] * len(tokenized)
        data_item.text = "".join(text_buffer)
        data_item.indexes = self.lookup_indexes(data_item.text)
        mask.append(False)  # for EOT character, '>'
        labels.append(-100)
        data_item.mask = mask
        data_item.labels = labels

        # returned data_item has self.text with the masked text with [] removed,
        # self.mask marking where there's masking,
        # self.labels with -100 for unmasked tokens and >= 0 with the embedding index for masked tokens.
        return data_item
