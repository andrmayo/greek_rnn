# Lacuna filler for Greek Papyri

## Purpose

The code here borrows a great deal from the Coptic RNN developed by
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

The two inference options to pass to the `main.py` CLI are `-pr` and `-prk`.
Both take a Greek sentence as a positional argument, with lacunae to be filled
indicated either with `_` for known-length lacunae or `<gap>` for unkown length.
For instance,

- `python main.py -pr 'ἀγαθὸς [___] ἐστιν'`

This fills in a lacuna of three characters.

- `python main.py -pr 'ἀγαθὸς <gap/> ἐστιν'`

This will also try to predict the length of the lacuna.

If `-prk` is used instead of `-pr`, the top 1000 most likely predictions
will be saved to a CSV file in `rnn_code/results` with a filename in the format
`top_k_{timestamp}.csv`.
