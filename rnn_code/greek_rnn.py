import logging
import random
import warnings
from typing import Any, Iterator, Literal, cast

import torch
import torch.nn as nn

import rnn_code.letter_tokenizer as letter_tokenizer
from rnn_code.greek_utils import DataItem, decoder_specs

logger = logging.getLogger(__name__)

MASK = "_"
USER_MASK = "#"


class RNN(nn.Module):
    def __init__(
        self,
        specs: list[int | float],
        spaces_are_tokens: bool = False,
        newlines_are_tokens: bool = True,
        decoder_type: str | None = None,
    ):
        super().__init__()
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
            list(letter_tokenizer.LetterTokenizer.greek_chars)
            + list(".?<>\n!")
            + [self.mask_char]
            + [self.user_mask_char]
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
            share,
            dropout,
            masking_proportion,
        ) = specs

        embed_size, hidden_size, proj_size = (
            cast(int, embed_size),
            cast(int, hidden_size),
            cast(int, proj_size),
        )

        self.embed_size = int(embed_size)
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.share = share

        if hidden_size % 2 == 1 or proj_size % 2 == 1:
            raise ValueError("hidden_size and proj_size must be even numbers")

        self.token_to_index = {}
        self.index_to_token = {}
        for i, token in enumerate(tokens):
            self.token_to_index[token] = i
            self.index_to_token[i] = token

        self.embed = nn.Embedding(self.num_tokens, embed_size)
        self.masking_proportion = masking_proportion

        # currently, the hidden_size is the same as the embedding size, so
        # this layer is unnecessary
        # self.scale_up = nn.Linear(embed_size, hidden_size)

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.filterwarnings(
                "ignore", "LSTM with projections is not supported with oneDNN"
            )
            self.rnn = nn.LSTM(
                embed_size,
                hidden_size // 2,
                num_layers=rnn_nLayers,
                bidirectional=True,
                dropout=dropout,
                batch_first=True,
                proj_size=proj_size // 2,
            )

        if caught_warnings:
            logger.info(
                "LSTM projection layer not using oneDNN optimization in __init__, which shouldn't matter as long as a GPU is used for training/inference."
            )

        if self.share:
            if proj_size > 0 and proj_size != embed_size:
                raise ValueError(
                    "if weight tying is enabled with share and projection layer, proj_size must equal embed_size"
                )
            elif hidden_size != embed_size:
                raise ValueError(
                    "if weight tying is enabled with share and without projection layer, hidden_size must equal embed_size"
                )
            self.out = None
        else:
            self.out = nn.Linear(
                proj_size if proj_size > 0 else hidden_size, self.num_tokens
            )

        self.dropout = nn.Dropout(dropout)

        self.decoder_type = decoder_type
        match self.decoder_type:
            case None:
                self.decoder = None
            case "gru":
                self.decoder = GRUDecoder(decoder_specs[cast(str, decoder_type)], self)
            case "lstm":
                raise NotImplementedError(
                    "unidirectional LSTM decoder not implemented yet"
                )

        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
                # nn.init.kaiming_uniform_(p)
                nn.init.kaiming_normal_(p)

    def forward(
        self,
        input_seqs: list[torch.Tensor] | tuple[torch.Tensor],
        masks: list[torch.Tensor] | None = None,
        labels: list[torch.Tensor] | None = None,
    ):
        if self.decoder and not masks:
            raise ValueError(
                "Calling RNN object requires 'masks' argument if using a decoder"
            )
        batch_size = len(input_seqs)
        num_tokens = len(input_seqs[0])
        seqs = torch.cat(input_seqs).view(batch_size, num_tokens)

        embed = self.embed(seqs)
        embed = self.dropout(embed)
        # embed = self.scale_up(embed)

        encoder_output, (hidden_state, cell_state) = self.rnn(embed)
        encoder_output = self.dropout(encoder_output)

        if self.share:
            # use embedding table as output layer
            logits = torch.matmul(encoder_output, torch.t(self.embed.weight))
        else:
            assert self.out
            logits = self.out(encoder_output)
        match self.decoder_type:
            case None:
                return logits
            case "gru":
                assert isinstance(self.decoder, nn.Module)
                assert masks is not None
                # multi-character lacunae get overriden here
                return self.decode_lacuna_regions(encoder_output, logits, masks, labels)
            case "lstm":
                raise NotImplementedError(
                    "unidirectional LSTM decoder not implemented yet"
                )
            case _:
                raise ValueError(
                    f"invalid decoder requested; decoder should be None or one of {list(decoder_specs.keys())}"
                )

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

    @staticmethod
    def find_contiguous_masked_regions(mask) -> Iterator[tuple[int, int]]:
        i = 0
        n = len(mask)
        while i < n:
            if not mask[i]:
                i += 1
                continue
            start = i
            while i < n and mask[i]:
                i += 1
            yield (start, i)

    def decode_lacuna_regions(
        self,
        encoder_output: torch.Tensor,
        logits: torch.Tensor,
        mask,
        teacher_labels=None,
    ) -> torch.Tensor:
        assert self.decoder

        for batch_i in range(encoder_output.size(0)):
            regions = RNN.find_contiguous_masked_regions(mask[batch_i])
            for start, end in regions:
                if end - start < 2:
                    continue
                region_labels = (
                    teacher_labels[batch_i][start:end]
                    if teacher_labels is not None
                    else None
                )
                logits[batch_i, start:end] = self.decoder.forward_region(
                    encoder_output[batch_i], start, end, teacher_labels=region_labels
                )
        return logits

    def mask_and_label_characters(
        self,
        data_item: DataItem,
        masking_strategy: Literal["random", "smart"] = "random",
    ) -> tuple[DataItem, int]:
        data_item.indexes = self.lookup_indexes(data_item.text)

        text_length = len(data_item.indexes)

        data_item.labels = (
            [-100] * text_length
        )  # labels mark where actual masking is with a value > 0, and this value is the index of the masked token

        # guard against short texts
        short_text = False
        if text_length < 3:
            short_text = True
            data_item.mask = [False] * text_length
            total_mask = 0
            return data_item, total_mask

        if short_text:
            logger.info(
                f"Encountered texts with less than 3 characters (original position: {data_item.position_in_original}, which have not been masked"
            )
        data_item.mask = [True] * text_length

        mask_count = 0  # This counts only tokens actually masked, not those swapped for random character or retained
        random_sub = 0
        orig_token = 0

        if masking_strategy == "random":
            for i in range(text_length):
                # we want to skip masking actual lacunae
                current_token = data_item.indexes[i]
                token_char = self.index_to_token[current_token]
                if token_char in ("!", "."):
                    data_item.mask[i] = False
                    continue

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
                    data_item.labels[i] = current_token

                else:
                    data_item.mask[i] = False

        elif masking_strategy == "smart":
            r_mask_quantity = random.randint(1, 5)

            should_mask = [False] * text_length

            for _ in range(r_mask_quantity):
                r_start_loc = random.randint(0, text_length - 1)
                r_mask_length = random.random()
                if r_mask_length <= 0.48:
                    mask_length = 1
                elif 0.48 < r_mask_length <= 0.70:
                    mask_length = 2
                elif 0.70 < r_mask_length <= 0.82:
                    mask_length = 3
                else:
                    mask_length = random.randint(4, 35)
                mask_end = r_start_loc + min(mask_length, text_length)
                should_mask[r_start_loc:mask_end] = [True] * mask_length

            for i in range(text_length):
                current_token = data_item.indexes[i]
                token_char = self.index_to_token[current_token]

                # Skip masking actual lacunae
                if token_char in ("!", "."):
                    data_item.mask[i] = False
                    continue

                if should_mask[i]:
                    r_mask_type = random.random()
                    if r_mask_type < 0.8:
                        # replace with MASK symbol
                        replacement = self.token_to_index[self.mask_char]
                        mask_count += 1
                    elif r_mask_type < 0.9:
                        # replace with random character
                        replacement = random.randint(0, self.num_tokens - 1)
                        random_sub += 1
                    else:
                        # retain original
                        replacement = current_token
                        orig_token += 1

                    data_item.indexes[i] = replacement
                    data_item.labels[i] = current_token
                else:
                    data_item.mask[i] = False

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

    def mask_input(
        self,
        data: list[DataItem],
        masking_strategy: Literal["random", "smart"],
    ) -> list[DataItem]:
        """Masking function: this is only used if masking is one-and-done at start of training."""
        logger.info(f"Training data read in with {len(data)} lines")

        data_for_model = []

        logger.info(f"Masking strategy is {masking_strategy}, masking sentences...")
        for data_item in data:
            masked_data_item, _ = self.mask_and_label_characters(
                data_item, masking_strategy=masking_strategy
            )
            data_for_model.append(masked_data_item)
        logger.info("Masking complete")

        return data_for_model


class GRUDecoder(nn.Module):
    """
    GRU decoder for use with LSTM encoder.
    Note that forward() is not implemented: this class is
    embedded in RNN above, which calls forward_region() below
    in the decode_lacuna_regions() method.
    """

    def __init__(
        self,
        decoder_specs: dict[str, Any],
        encoder: RNN,
    ):
        super().__init__()
        self.num_tokens = encoder.num_tokens
        self.encoder_specs = encoder.specs
        self.embed = encoder.embed
        self.embed_size = encoder.embed_size
        self.encoder_proj_size = encoder.proj_size
        self.encoder_hidden_size = encoder.hidden_size

        self.specs = decoder_specs

        gru_params = {
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
            "device",
        }

        self.gru = torch.nn.GRU(
            **{k: v for k, v in decoder_specs.items() if k in gru_params}
        )

        # marks beginning of masked region / lacuna
        self.start_embedding = nn.Parameter(torch.randn(self.embed_size))

        hidden_size = self.specs["hidden_size"]

        # NOTE: we have a linear layer from encoder outputs to GRU inputs
        # even where projection isn't necessary, because dimensions already match
        if self.encoder_proj_size > 0:
            self.encoder_projection = nn.Linear(self.encoder_proj_size, hidden_size)
        else:
            self.encoder_projection = nn.Linear(self.encoder_hidden_size, hidden_size)

        # projection layer that projects concatenated [biLSTM_context, prev_token_embedding]
        # to the decoder hidden_size
        encoder_output_size = (
            self.encoder_proj_size
            if self.encoder_proj_size > 0
            else self.encoder_hidden_size
        )
        self.input_projection = nn.Linear(
            encoder_output_size + self.embed_size, hidden_size
        )

        # this is the projection layer before using embeddings to score all tokens
        if self.specs["hidden_size"] != self.embed_size:
            self.output_projection = torch.nn.Linear(
                self.specs["hidden_size"], self.embed_size
            )
        else:
            self.output_projection = None

        # NOTE: parameters with > 1 dimensions will get initialized in encoder

    def forward_region(
        self,
        encoder_output: torch.Tensor,
        start: int,
        end: int,
        teacher_labels=None,
    ):
        h = self.compute_init_hidden(encoder_output, start, end)
        prev_embed = self.start_embedding
        all_logits = []
        for t in range(end - start):
            pos = start + t
            context = encoder_output[pos]

            decoder_input = self.input_projection(
                torch.cat([context, prev_embed], dim=-1)
            )
            decoder_input = torch.relu(decoder_input)
            gru_out, h = self.gru(decoder_input.unsqueeze(0).unsqueeze(0), h)

            # Project to logits
            out = gru_out.squeeze(0).squeeze(0)
            if self.output_projection:
                out = self.output_projection(out)
            logits = torch.matmul(out, torch.t(self.embed.weight))
            all_logits.append(logits)

            if teacher_labels is not None:
                prev_embed = self.embed(teacher_labels[t])
            else:
                prev_embed = self.embed(logits.argmax(dim=-1))

        return torch.stack(all_logits, dim=0)  # (region_length, num_tokens)

    def compute_init_hidden(self, encoder_output: torch.Tensor, start: int, end: int):
        seq_len = encoder_output.size(0)
        boundaries = []
        if start > 0:
            boundaries.append(encoder_output[start - 1])
        if end < seq_len:
            boundaries.append(encoder_output[end])

        if not boundaries:
            boundary_repr = torch.zeros(
                self.encoder_proj_size
                if self.encoder_proj_size > 0
                else self.encoder_hidden_size,
                device=encoder_output.device,
            )
        elif len(boundaries) == 1:
            boundary_repr = boundaries[0]
        else:
            boundary_repr = (boundaries[0] + boundaries[1]) / 2.0

        h0 = torch.tanh(self.encoder_projection(boundary_repr))

        return h0.unsqueeze(0).unsqueeze(0)


def count_parameters(model: nn.Module):
    total = 0
    for name, p in model.named_parameters():
        if p.dim() > 1:
            logger.debug(f"{p.numel():,}\t{name}")
            total += p.numel()

    logger.info(f"total parameter count = {total:,}")
