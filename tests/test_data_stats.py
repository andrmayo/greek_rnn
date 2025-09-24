import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from collections import Counter
from rnn_code.data_stats import char_histogram, char_counts, gap_counts


class TestCharHistogram:
    def test_char_histogram_basic(self):
        # Create a temporary file with known content
        content = "ααβγγγ"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            result = char_histogram(temp_file_path)
            assert isinstance(result, Counter)
            assert result['α'] == 2
            assert result['β'] == 1
            assert result['γ'] == 3
        finally:
            os.unlink(temp_file_path)

    def test_char_histogram_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name

        try:
            result = char_histogram(temp_file_path)
            assert isinstance(result, Counter)
            assert len(result) == 0
        finally:
            os.unlink(temp_file_path)

    def test_char_histogram_with_whitespace(self):
        content = "α β\nγ"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            result = char_histogram(temp_file_path)
            assert result[' '] == 1
            assert result['\n'] == 1
            assert result['α'] == 1
            assert result['β'] == 1
            assert result['γ'] == 1
        finally:
            os.unlink(temp_file_path)


class TestCharCounts:
    def test_char_counts_basic(self, caplog):
        content = "αβγ\nδεζ\nηθι"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            char_counts(temp_file_path)

            # Check that appropriate log messages were generated
            log_messages = caplog.text
            assert "Number of sentences: 3" in log_messages
            assert "Number of characters: 9" in log_messages  # 3*3 characters (newlines not counted)
            assert "Shortest sentence:" in log_messages
            assert "Longest sentence:" in log_messages
            assert "Average sentence length:" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_char_counts_single_sentence(self, caplog):
        content = "αβγδε"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            char_counts(temp_file_path)

            log_messages = caplog.text
            assert "Number of sentences: 1" in log_messages
            assert "Number of characters: 5" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_char_counts_empty_file(self, caplog):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name

        try:
            char_counts(temp_file_path)

            log_messages = caplog.text
            assert "Number of sentences: 1" in log_messages  # Empty string becomes 1 sentence
            assert "Number of characters: 0" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_char_counts_varying_lengths(self, caplog):
        content = "α\nαβγδε\nαβ"  # Lengths: 1, 5, 2
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            char_counts(temp_file_path)

            log_messages = caplog.text
            assert "Shortest sentence: 1" in log_messages
            assert "Longest sentence: 5" in log_messages
            # Average should be (1+5+2)/3 = 2.67
            assert "Average sentence length: 2.67" in log_messages
        finally:
            os.unlink(temp_file_path)


class TestGapCounts:
    def test_gap_counts_basic(self, caplog):
        content = "αβ##γδ\nεζ###ηθ\nικλμν"  # 2 and 3 char gaps, plus no gap
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 5" in log_messages  # 2 + 3
            assert "Masks per sentence:" in log_messages
            assert "Length per gap:" in log_messages
            assert "Total gap characters: 5, total gaps: 2" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_no_gaps(self, caplog):
        content = "αβγδε\nζηθικ"
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 0" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_multiple_gaps_per_sentence(self, caplog):
        content = "α##β###γ#δ"  # 3 gaps in one sentence: lengths 2, 3, 1
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 6" in log_messages  # 2 + 3 + 1
            # Should have 1 sentence with 3 masks
            assert "{3: 1}" in log_messages or "3: 1" in log_messages
            # Should show gap length distribution
            assert "Length per gap:" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_single_char_gaps(self, caplog):
        content = "α#β#γ#δ"  # 3 single-character gaps
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 3" in log_messages
            assert "total gaps: 3" in log_messages
            # Length distribution should show 3 gaps of length 1
            assert "{1: 3}" in log_messages or "1: 3" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_mixed_sentence_types(self, caplog):
        content = "α##β\nγδεζ\nη###θ#ι"  # Mixed: gap, no gap, multiple gaps
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 6" in log_messages  # 2 + 3 + 1
            # Masks per sentence: {0: 1, 1: 1, 2: 1} - 1 with 0, 1 with 1, 1 with 2
            assert "Masks per sentence:" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_empty_file(self, caplog):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("")
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "Gap characters: 0" in log_messages
        finally:
            os.unlink(temp_file_path)

    def test_gap_counts_average_calculation(self, caplog):
        content = "α##β###γ"  # 2 gaps: lengths 2 and 3, average = 2.5
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            gap_counts(temp_file_path)

            log_messages = caplog.text
            assert "average length per gap 2.5" in log_messages
        finally:
            os.unlink(temp_file_path)