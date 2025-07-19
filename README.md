# Lacuna filler for Greek Papyri

## Purpose

The code here borrows a great deal from the Coptic RNN developed by [Levine et al.](https://arxiv.org/html/2407.12247v1). As with
the Coptic RNN probject, the main goal here is to generate textual reconstructions of a given lacuna, ranked according to their probability
of being correct.

## Data

The RNN is set up to train on the [Machine-Actionable Ancient Text Corpus](https://aclanthology.org/2024.ml4al-1.7.pdf) of Greek papyri texts created by Will Fitzgerald and Justin Barney. This corpus provides convenient JSON representations for the texts of just about all digitally available Greek papyri,
along with indications of lacunae (gaps) in the texts and available scholarly reconstructions of these lacunae.

## Usage
