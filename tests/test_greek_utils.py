import pytest
import torch
import tempfile
import json
import os
from unittest.mock import patch, MagicMock

from rnn_code.greek_utils import (
    DataItem,
    filter_diacritics,
    count_parameters,
    read_datafile,
    filter_brackets,
    skip_sentence,
    has_more_than_one_latin_character,
    mask_input,
    construct_trigram_lookup,
    get_home_path
)


class TestDataItem:
    def test_init_default(self):
        item = DataItem()
        assert item.text_number is None
        assert item.text is None
        assert item.indexes is None
        assert item.mask is None
        assert item.labels is None
        assert item.position_in_original is None

    def test_init_with_values(self):
        item = DataItem(
            text="test text",
            text_number=123,
            indexes=[1, 2, 3],
            mask=[True, False, True],
            labels=[10, -100, 20],
            position_in_original=5
        )
        assert item.text == "test text"
        assert item.text_number == 123
        assert item.indexes == [1, 2, 3]
        assert item.mask == [True, False, True]
        assert item.labels == [10, -100, 20]
        assert item.position_in_original == 5


class TestFilterDiacritics:
    def test_basic_filtering(self):
        text = "Τέστ"
        result = filter_diacritics(text)
        assert result == "τεστ"

    def test_combining_dot_preserved(self):
        text = "test\u0323"  # combining dot below
        result = filter_diacritics(text)
        assert "\u0323" in result

    def test_empty_string(self):
        result = filter_diacritics("")
        assert result == ""

    def test_lowercase_conversion(self):
        result = filter_diacritics("ΑΒΓΔ")
        assert result == "αβγδ"


class TestCountParameters:
    def test_count_parameters(self, caplog):
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 1)
        )

        count_parameters(model)

        # Check that logging occurred
        assert "total parameter count" in caplog.text


class TestReadDatafile:
    def test_read_valid_json(self):
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump({"text": "sample text"}, f)
            f.write('\n')
            json.dump({"text": "another text"}, f)
            temp_path = f.name

        try:
            result = read_datafile(temp_path)
            assert len(result) == 2
            assert result[0].text == "sample text"
            assert result[1].text == "another text"
        finally:
            os.unlink(temp_path)

    def test_read_empty_text_skipped(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump({"text": ""}, f)
            f.write('\n')
            json.dump({"text": "valid text"}, f)
            temp_path = f.name

        try:
            result = read_datafile(temp_path)
            assert len(result) == 1
            assert result[0].text == "valid text"
        finally:
            os.unlink(temp_path)


class TestFilterBrackets:
    def test_remove_various_brackets(self):
        text = "{test}|pipe|(content)[square]"
        result = filter_brackets(text)
        assert result == "testpipesquare"  # Square brackets preserve content (for lacuna processing)

    def test_empty_string(self):
        result = filter_brackets("")
        assert result == ""

    def test_no_brackets(self):
        text = "plain text"
        result = filter_brackets(text)
        assert result == "plain text"


class TestSkipSentence:
    def test_skip_multiple_latin_chars(self):
        assert skip_sentence("Greek text with abc") is True

    def test_skip_dashes(self):
        assert skip_sentence("text[----]more") is True

    def test_skip_short_text(self):
        assert skip_sentence("abc") is True

    def test_dont_skip_valid_text(self):
        assert skip_sentence("αβγδεζηθ") is False

    def test_single_latin_char_ok(self):
        assert skip_sentence("Gρεεκ τεχτ ωιτη") is False  # Only one Latin char 'G'


class TestHasMoreThanOneLatinCharacter:
    def test_multiple_latin_chars(self):
        assert has_more_than_one_latin_character("abc") is True

    def test_single_latin_char(self):
        assert has_more_than_one_latin_character("a") is False

    def test_no_latin_chars(self):
        assert has_more_than_one_latin_character("αβγ") is False

    def test_exactly_two_latin_chars(self):
        assert has_more_than_one_latin_character("ab") is True


class TestMaskInput:
    def test_once_strategy(self, caplog):
        mock_model = MagicMock()
        mock_model.mask_and_label_characters.return_value = ("masked", 5)

        data = [DataItem(text="test1"), DataItem(text="test2")]

        result_data, mask = mask_input(mock_model, data, "random", "once")

        assert mask is False
        assert len(result_data) == 2
        assert "Masking strategy is once" in caplog.text
        assert mock_model.mask_and_label_characters.call_count == 2

    def test_dynamic_strategy(self, caplog):
        mock_model = MagicMock()
        data = [DataItem(text="test1"), DataItem(text="test2")]

        result_data, mask = mask_input(mock_model, data, "smart", "dynamic")

        assert mask is True
        assert result_data is data
        assert "dynamic" in caplog.text
        assert mock_model.mask_and_label_characters.call_count == 0


class TestGetHomePath:
    def test_returns_string(self):
        result = get_home_path()
        assert isinstance(result, str)
        assert len(result) > 0


class TestConstructTrigramLookup:
    @patch('builtins.open', create=True)
    @patch('json.dump')
    @patch('rnn_code.greek_utils.tokenizer')
    def test_construct_trigram_lookup(self, mock_tokenizer, mock_json_dump,
                                      mock_open):
        # Mock file reading
        mock_open.return_value.__enter__.return_value.__iter__.return_value = [
            '{"text": "αβγ"}',
            '{"text": "δεζ"}'
        ]

        # Mock tokenizer
        mock_tokenizer.tokenize.side_effect = [
            ['α', 'β', 'γ'],
            ['δ', 'ε', 'ζ']
        ]

        result = construct_trigram_lookup()

        assert isinstance(result, dict)
        mock_json_dump.assert_called_once()