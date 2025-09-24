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
