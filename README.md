# Lacuna filler for Greek Papyri

## Purpose

The approach here is inspired by the Coptic RNN developed by
[Levine et al.](https://arxiv.org/html/2407.12247v1). As with the Coptic RNN
project, the main goal here is to generate textual reconstructions of a given
lacuna, ranked according to their probability of being correct.

## Data

The RNN is set up to train on the
[Machine-Actionable Ancient Text Corpus](https://aclanthology.org/2024.ml4al-1.7.pdf)
of Greek papyri texts created by Will Fitzgerald and Justin Barney. This corpus
provides convenient JSON representations for the texts of just about all
digitally available Greek papyri, along with indications of lacunae (gaps) in
the texts and available scholarly reconstructions of these lacunae.

## How Training and Evaluation Work

### Data Preprocessing

The corpus contains texts with three types of lacunae markers:

- `[reconstruction]`: Lacunae with scholarly reconstructions
- `.`: Single missing characters without reconstruction
- `<gap/>`: Variable-length lacunae without reconstruction

During data partitioning (`--partition` flag), all texts are randomly split into
train (80%), dev (10%), and test (10%) sets. Importantly, **all texts are
included regardless of whether they contain lacunae or reconstructions** - there
is no filtering by lacuna density.

### Training Phase

**How scholarly reconstructions are used:**

Training data treats scholarly reconstructions as ground truth by merging them
into the text as if they were preserved characters:

```
Input corpus:     "word[abc]word<gap/>text"
Training text:    "wordabcword!text"
```

- Brackets `[]` are removed and reconstruction text is incorporated directly
- `<gap/>` becomes `!` (a special token for unknown-length gaps)
- Periods `.` representing single missing characters are kept as-is

**Masking during training:**

The model uses masked language modeling. For each training example, a masking
function randomly or contiguously masks 15% of characters:

- 80% of masked tokens → replaced with `_` (MASK character)
- 10% of masked tokens → replaced with random character
- 10% of masked tokens → kept as original

The model learns to predict the original character at each masked position.
Loss is computed **only on masked positions**, not on unmasked context.

**Note:** The model trains on scholarly reconstructions as if they are
certain, preserved text. In general, whether right or wrong, scholarly
reconstructions should be plausible enough so as not to degrade
the quality of the training data.

### Validation and Test Phase

**How scholarly reconstructions are used:**

Dev and test data preserve the distinction between preserved text and
reconstructions using brackets:

```
Input corpus:     "word[abc]word"
Dev/test text:    "word___word"
Mask:             [F, F, F, F, T, T, T, F, F, F, F]
Labels:           [-100, -100, -100, -100, idx_a, idx_b, idx_c, -100, -100, -100, -100]
```

The `actual_lacuna_mask_and_label()` function:

1. Replaces all characters within `[]` with `_` (mask character)
2. Removes the bracket characters themselves
3. Stores token indices of the scholarly reconstruction in the `labels` field

**Evaluation metric:**

During evaluation, the model:

1. Receives masked text (e.g., `"word___word"`)
2. Predicts what should fill each masked position
3. Has its predictions compared character-by-character against the scholarly
   reconstruction stored in `labels`

```
Accuracy = (correct predictions) / (total masked tokens)
```

Where "correct" means the model's prediction **exactly matches** the scholarly
reconstruction.

**Important notes:**

- **Scholarly reconstructions serve as ground truth** for evaluation. The model
  is rewarded for reproducing the same reconstruction that scholars proposed,
  whether or not that reconstruction is actually correct.
- Only the **first alternative** is used when multiple scholarly reconstructions
  exist for the same lacuna (per comment at `greek_char_data.py:82`).
  With the current MAAT corpus, this makes the most sense, since alternative
  reconstructions are very rarely present.
- The model never sees the actual lacunae (with brackets) during training - it
  only sees them during dev/test evaluation.

### Training vs. Evaluation Summary

| Phase          | Input Format    | Reconstructions             | Purpose                 |
| -------------- | --------------- | --------------------------- | ----------------------- |
| **Training**   | `"wordabcword"` | Merged as certain text      | Learn language patterns |
| **Validation** | `"word___word"` | Used as ground truth labels | Tune hyperparameters    |
| **Testing**    | `"word___word"` | Used as ground truth labels | Measure performance     |

This approach means the model learns from reconstructions (treating them as
correct) and is evaluated on how well it reproduces those same reconstructions.
If scholarly reconstructions contain errors, the model will learn those errors
and be rewarded for reproducing them.

## File Structure

### `rnn_code/` Directory

- **`main.py`**: Entry point script with argument parsing for training,
  evaluation, prediction, and ranking tasks. Handles data partitioning, model
  initialization, and coordinates the overall workflow.

- **`greek_rnn.py`**: Core RNN model implementation. Contains the bidirectional
  LSTM architecture with embedding layers, masking functions for training data
  (`mask_and_label_characters`) and test data (`actual_lacuna_mask_and_label`),
  and character encoding/decoding utilities.

- **`greek_char_generator.py`**: Training and evaluation engine. Implements the
  training loop (`train_model`), batch processing (`train_batch`), accuracy
  evaluation functions, prediction utilities, and baseline comparison methods.

- **`greek_utils.py`**: Configuration and utility functions. Contains
  hyperparameters, logging setup, the `DataItem` class for text processing,
  diacritic filtering, and various data processing utilities.

- **`greek_char_data.py`**: Data loading and preprocessing. Handles reading
  JSON files, train/dev/test splitting, data partitioning, and writing
  processed data to files.

- **`letter_tokenizer.py`**: Character-level tokenization for Greek text.
  Handles special characters, diacritics, and creates mappings between
  characters and indices.

- **`data_stats.py`**: Statistical analysis utilities for the dataset. Provides
  functions to analyze character distributions, text lengths, and other corpus
  statistics.

## Usage

### Using locally trained model

When training is run successfully, on each run the weights will be saved
`{model_path}/{output_name}_latest.pth`, which by default will be
`rnn_code/models/greek_lacuna_latest.pth`. To use these weights for inference,
this file needs to be moved to `rnn_code/models/best/`.

### Loading pretrained model

If the `-t` / `--train` option isn't passed to `main.py` in the CLI, the program
will search for an existing model as a `.pth` file in `rnn_code/models/best`,
and will select the most recently modified file.
To set this up based on the compressed `.pth` file in `rnn_code/models/compressed-weights`,
run `source setup.sh` in the project root directory.

### Inference

The three inference options to pass to the `main.py` CLI are
`-pr` (`--predict`), `-prk` (`--predict_top_k`), and `-r` (`--rank`).
Each takes a Greek sentence as a positional argument, with lacunae to be filled
indicated either with `_` for known-length lacunae or `<gap>` for unkown length.
For instance,

- `python main.py -pr 'ἀγαθὸς [___] ἐστιν'`

This fills in a lacuna of three characters.

- `python main.py -pr 'ἀγαθὸς <gap/> ἐστιν'`

This will also try to predict the length of the lacuna.

If `-prk` is used instead of `-pr`, the top 1000 most likely predictions
will be saved to a CSV file in `rnn_code/results` with a filename in the format
`top_k_{timestamp}.csv`.

Passing `-r` followed by a sentence with a lacuna and options
of the same length as the lacuna will rank the proposed
reconstructions from most to least likely. White space
and punctuation don't count towards matching the lacuna length,
and currently this doesn't work with lacunae of unspecified length.

- `python main.py -r 'ἄνδρες [___] γυναῖκες' και 'τα δ' γαρ τον`
