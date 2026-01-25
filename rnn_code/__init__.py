"""
GREEK RNN (LSTM) for lacuna filling (i.e. gap filling) in ancient papyrus texts.

This package provides neural-network based tools for reconstructing missing text
in Greek papyri from the ancient world using bidirectional LSTM architecture
with character-level tokenization.
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(
    f"{Path(__file__).parent}/log/greek_data_processing.log"
)
file_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
