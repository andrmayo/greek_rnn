import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from rnn_code.greek_char_data import read_datafiles, write_to_json


class TestReadDatafiles:
    def create_mock_json_file(self, content_list):
        """Helper to create a temporary JSON file with given content"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False,
                                                suffix='.json')
        for content in content_list:
            json.dump(content, temp_file, ensure_ascii=False)
            temp_file.write('\n')
        temp_file.close()
        return temp_file.name

    def test_read_datafiles_basic(self):
        # Create mock JSON data
        json_data = [
            {
                "language": "grc",
                "training_text": "αβγ[δε]ζη",
                "test_cases": [{"alternatives": ["δε"]}]
            },
            {
                "language": "grc",
                "training_text": "ικλ<gap/>μνξ",
                "test_cases": [{"alternatives": ["οπ"]}]
            }
        ]

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            # Check basic structure
            assert isinstance(train_data, list)
            assert isinstance(dev_data, list)
            assert isinstance(test_data, list)
            assert isinstance(random_index, list)
            assert isinstance(reconstructions, list)
            assert isinstance(mapping, dict)

            # Check total length matches
            total_texts = len(train_data) + len(dev_data) + len(test_data)
            assert total_texts == 2

        finally:
            os.unlink(temp_file)

    def test_read_datafiles_filters_non_greek(self):
        # Include non-Greek language entries
        json_data = [
            {
                "language": "lat",  # Latin, should be filtered out
                "training_text": "lorem ipsum",
                "test_cases": []
            },
            {
                "language": "grc",  # Greek, should be included
                "training_text": "αβγδε",
                "test_cases": []
            }
        ]

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            # Should only have 1 text (the Greek one)
            total_texts = len(train_data) + len(dev_data) + len(test_data)
            assert total_texts == 1

        finally:
            os.unlink(temp_file)

    def test_training_text_processing(self):
        # Use multiple texts so some go to train_data (need at least 10 for 80% to give 8+ for train)
        json_data = []
        for i in range(10):
            json_data.append({
                "language": "grc",
                "training_text": f"α{i}[βγ]δ<gap/>ε",
                "test_cases": [{"alternatives": ["βγ"]}, {"alternatives": ["ζ"]}]
            })

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            # Training data should have brackets removed and gap replaced
            assert len(train_data) > 0, "Should have training data with 10 texts"
            train_text = train_data[0]
            assert '[' not in train_text
            assert ']' not in train_text
            assert '<gap/>' not in train_text
            assert '!' in train_text  # <gap/> should become !

        finally:
            os.unlink(temp_file)

    def test_lacuna_text_processing(self):
        json_data = [
            {
                "language": "grc",
                "training_text": "α[β γ]δ<gap/>ε",  # Space in lacuna
                "test_cases": [{"alternatives": ["βγ"]}, {"alternatives": ["ζ"]}]
            }
        ]

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            # Find a text in dev or test data (lacuna texts)
            lacuna_text = None
            for text_list in [dev_data, test_data]:
                if text_list:
                    lacuna_text = text_list[0]
                    break

            if lacuna_text:
                # Spaces within brackets should be removed
                assert "[ " not in lacuna_text and " ]" not in lacuna_text
                assert '<gap/>' not in lacuna_text
                assert '!' in lacuna_text

        finally:
            os.unlink(temp_file)

    def test_reconstructions_processing(self):
        json_data = [
            {
                "language": "grc",
                "training_text": "αβγ[δε]ζη",
                "test_cases": [{"alternatives": ["δε"]}]
            },
            {
                "language": "grc",
                "training_text": "ικλμνξ",  # No lacunae
                "test_cases": []
            }
        ]

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            assert len(reconstructions) == 2
            assert reconstructions[0] == ["δε"]  # First text has reconstruction
            assert reconstructions[1] == []      # Second text has no lacunae

        finally:
            os.unlink(temp_file)

    def test_space_alternative_becomes_underscore(self):
        json_data = [
            {
                "language": "grc",
                "training_text": "αβγ[_]δε",
                "test_cases": [{"alternatives": [" "]}]  # Space alternative
            }
        ]

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            assert reconstructions[0] == ["_"]  # Space becomes underscore

        finally:
            os.unlink(temp_file)

    def test_data_partition_ratios(self):
        # Create enough data to test partitioning
        json_data = []
        for i in range(100):  # 100 texts
            json_data.append({
                "language": "grc",
                "training_text": f"text{i}",
                "test_cases": []
            })

        temp_file = self.create_mock_json_file(json_data)

        try:
            result = read_datafiles([temp_file])
            train_data, dev_data, test_data, random_index, reconstructions, mapping = result

            # Check approximate ratios (80:10:10)
            total = len(train_data) + len(dev_data) + len(test_data)
            assert total == 100

            # Allow some flexibility in ratios due to integer division
            assert 75 <= len(train_data) <= 85
            assert 5 <= len(dev_data) <= 15
            assert 5 <= len(test_data) <= 15

        finally:
            os.unlink(temp_file)


class TestWriteToJson:
    def test_write_to_json_basic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = "test.json"
            text_list = ["text1", "text2"]
            text_index = [0, 1]
            mapping = {0: [0], 1: [1]}

            # Mock the Path to use temp directory
            with patch('rnn_code.greek_char_data.Path') as mock_path:
                mock_path.return_value.absolute.return_value.parent = temp_dir

                write_to_json(file_name, text_list, text_index, mapping)

                # Check that file was created
                file_path = os.path.join(temp_dir, "data", file_name)
                assert os.path.exists(file_path)

                # Check file contents
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    assert len(lines) == 2

                    # Parse first line
                    first_entry = json.loads(lines[0])
                    assert first_entry["text_index"] == 0
                    assert first_entry["text"] == "text1"
                    assert first_entry["position_in_original"] == [0]

    def test_write_to_json_creates_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = "test.json"
            text_list = ["text1"]
            text_index = [0]
            mapping = {0: [0]}

            with patch('rnn_code.greek_char_data.Path') as mock_path:
                mock_path.return_value.absolute.return_value.parent = temp_dir

                write_to_json(file_name, text_list, text_index, mapping)

                # Check that data directory was created
                data_dir = os.path.join(temp_dir, "data")
                assert os.path.exists(data_dir)
                assert os.path.isdir(data_dir)

    def test_write_to_json_unicode_handling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_name = "test_unicode.json"
            text_list = ["αβγδε", "ζηθικ"]  # Greek text
            text_index = [0, 1]
            mapping = {0: [0], 1: [1]}

            with patch('rnn_code.greek_char_data.Path') as mock_path:
                mock_path.return_value.absolute.return_value.parent = temp_dir

                write_to_json(file_name, text_list, text_index, mapping)

                file_path = os.path.join(temp_dir, "data", file_name)

                # Read back and verify Unicode is preserved
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    first_entry = json.loads(lines[0])
                    assert first_entry["text"] == "αβγδε"