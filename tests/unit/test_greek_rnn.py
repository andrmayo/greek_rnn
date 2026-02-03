from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from rnn_code.greek_rnn import RNN, count_parameters
from rnn_code.greek_utils import DataItem


class TestRNN:
    @pytest.fixture
    def sample_specs(self):
        return [300, 300, 150, 4, False, 0.0, 0.15]  # Standard specs

    @pytest.fixture
    def rnn_model(self, sample_specs):
        return RNN(sample_specs)

    def test_init(self, sample_specs):
        model = RNN(sample_specs)
        assert model.specs == sample_specs + [34]  # 34 is num_tokens
        assert model.num_tokens == 34
        assert model.mask_char == "_"
        assert model.user_mask_char == "#"
        assert model.bot_char == "<"
        assert model.eot_char == ">"
        assert model.unk_char == "?"

    def test_token_mappings(self, rnn_model):
        # Test that token mappings are bidirectional
        for token, index in rnn_model.token_to_index.items():
            assert rnn_model.index_to_token[index] == token

        # Test specific tokens
        assert rnn_model.token_to_index["_"] is not None
        assert rnn_model.token_to_index["#"] is not None
        assert rnn_model.token_to_index["<"] is not None
        assert rnn_model.token_to_index[">"] is not None
        assert rnn_model.token_to_index["α"] is not None

    def test_lookup_indexes_basic(self, rnn_model):
        text = "αβγ"
        indexes = rnn_model.lookup_indexes(text)

        # Should include BOT and EOT tokens by default
        assert len(indexes) == len(text) + 2
        assert indexes[0] == rnn_model.token_to_index["<"]
        assert indexes[-1] == rnn_model.token_to_index[">"]

    def test_lookup_indexes_no_control(self, rnn_model):
        text = "αβγ"
        indexes = rnn_model.lookup_indexes(text, add_control=False)

        # Should not include BOT and EOT tokens
        assert len(indexes) == len(text)

    def test_decode_tensor(self, rnn_model):
        # Test with tensor input
        indexes = torch.tensor([1, 2, 3])
        result = rnn_model.decode(indexes)
        assert isinstance(result, str)

    def test_decode_list(self, rnn_model):
        # Test with list input
        alpha_idx = rnn_model.token_to_index["α"]
        beta_idx = rnn_model.token_to_index["β"]
        indexes = [alpha_idx, beta_idx]
        result = rnn_model.decode(indexes)
        assert result == "αβ"

    def test_forward_basic(self, rnn_model):
        # Create sample input
        text = "αβ"
        indexes = rnn_model.lookup_indexes(text)
        seqs = [torch.tensor(indexes, dtype=torch.int64)]

        output = rnn_model.forward(seqs)

        # Output should have correct shape
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == len(indexes)  # sequence length
        assert output.shape[2] == rnn_model.num_tokens  # vocabulary size

    def test_mask_and_label_characters_random(self, rnn_model):
        data_item = DataItem(text="αβγδε")

        masked_item, total_mask = rnn_model.mask_and_label_characters(
            data_item, mask_type="random"
        )

        assert hasattr(masked_item, "indexes")
        assert hasattr(masked_item, "mask")
        assert hasattr(masked_item, "labels")
        assert len(masked_item.mask) == len(masked_item.indexes)
        assert len(masked_item.labels) == len(masked_item.indexes)
        assert isinstance(total_mask, int)

    def test_mask_and_label_characters_smart(self, rnn_model):
        data_item = DataItem(text="αβγδεζηθικλμνξοπ")

        masked_item, total_mask = rnn_model.mask_and_label_characters(
            data_item, mask_type="smart"
        )

        assert hasattr(masked_item, "indexes")
        assert hasattr(masked_item, "mask")
        assert hasattr(masked_item, "labels")
        assert len(masked_item.mask) == len(masked_item.indexes)
        assert len(masked_item.labels) == len(masked_item.indexes)
        assert isinstance(total_mask, int)

    def test_actual_lacuna_mask_and_label(self, rnn_model):
        # Test with lacuna brackets
        data_item = DataItem(text="αβ[γδ]εζ")

        result = rnn_model.actual_lacuna_mask_and_label(data_item)

        assert hasattr(result, "indexes")
        assert hasattr(result, "mask")
        assert hasattr(result, "labels")
        assert len(result.mask) == len(result.indexes)
        assert len(result.labels) == len(result.indexes)

        # Text should have brackets removed and lacuna replaced with mask chars
        assert "[" not in result.text
        assert "]" not in result.text
        assert "_" in result.text

    def test_actual_lacuna_mask_and_label_no_lacuna(self, rnn_model):
        # Test with no lacuna
        data_item = DataItem(text="αβγδε")

        result = rnn_model.actual_lacuna_mask_and_label(data_item)

        # Should have BOT and EOT tokens
        assert len(result.mask) == len(result.text) + 2
        # All masks should be False except for positions with actual lacunae
        lacuna_positions = [i for i, mask in enumerate(result.mask) if mask]
        # In this case, no lacunae, so no True masks except possibly from tokenization
        assert isinstance(result.mask, list)
        assert len(lacuna_positions) == 0

    def test_weight_sharing_false(self, sample_specs):
        sample_specs[4] = False  # share = False
        model = RNN(sample_specs)
        assert hasattr(model, "out")
        assert not hasattr(model, "scale_down")

    def test_weight_sharing_true(self, sample_specs):
        sample_specs[4] = True  # share = True
        model = RNN(sample_specs)
        assert not hasattr(model, "out")
        assert hasattr(model, "scale_down")

    def test_dropout_initialization(self):
        # Test with dropout
        specs_with_dropout = [300, 300, 150, 4, False, 0.5, 0.15]
        model = RNN(specs_with_dropout)
        assert model.dropout.p == 0.5

        # Test without dropout
        specs_no_dropout = [300, 300, 150, 4, False, 0.0, 0.15]
        model = RNN(specs_no_dropout)
        assert model.dropout.p == 0.0

    def test_model_parameters_exist(self, rnn_model):
        # Test that all expected parameters exist
        param_names = [name for name, _ in rnn_model.named_parameters()]

        expected_params = [
            "embed.weight",
            "rnn.weight_ih_l0",
            "rnn.weight_hh_l0",
            "rnn.bias_ih_l0",
            "rnn.bias_hh_l0",
            "out.weight",
            "out.bias",
        ]

        for expected in expected_params:
            assert any(expected in param for param in param_names)

    def test_bidirectional_lstm(self, rnn_model):
        # Test that LSTM is bidirectional
        assert rnn_model.rnn.bidirectional is True
        assert rnn_model.rnn.hidden_size == 150  # proj_size
        # Effective hidden size should be 300 (150 * 2 for bidirectional)

    def test_embedding_dimensions(self, rnn_model):
        assert rnn_model.embed.num_embeddings == 34  # num_tokens
        assert rnn_model.embed.embedding_dim == 300  # embed_size

    @pytest.mark.parametrize("mask_type", ["random", "smart"])
    def test_masking_preserves_sequence_length(self, rnn_model, mask_type):
        original_text = "αβγδεζηθικλμν"
        data_item = DataItem(text=original_text)

        masked_item, _ = rnn_model.mask_and_label_characters(data_item, mask_type)

        # Sequence length should be preserved (plus BOT/EOT)
        expected_length = len(rnn_model.lookup_indexes(original_text))
        assert len(masked_item.indexes) == expected_length

    @pytest.mark.parametrize("mask_type", ["random", "smart"])
    def test_single_char_lacuna_marker_not_masked(self, rnn_model, mask_type):
        """Test that '.' (single missing char marker) is never masked."""
        # Use text with '.' characters representing unknown single characters
        text = "αβγ.δε.ζηθ"

        # Run masking multiple times to account for randomness
        for _ in range(10):
            data_item_copy = DataItem(text=text)
            masked_item, _ = rnn_model.mask_and_label_characters(
                data_item_copy, mask_type=mask_type
            )

            # Find positions of '.' in the token sequence (accounting for BOT token)
            dot_index = rnn_model.token_to_index["."]
            for i, idx in enumerate(masked_item.indexes):
                if idx == dot_index:
                    # '.' should never be masked
                    assert masked_item.mask[i] is False, (
                        f"'.' at position {i} was masked in {mask_type} mode"
                    )
                    assert masked_item.labels[i] == -100, (
                        f"'.' at position {i} has label in {mask_type} mode"
                    )

    @pytest.mark.parametrize("mask_type", ["random", "smart"])
    def test_gap_marker_not_masked(self, rnn_model, mask_type):
        """Test that '!' (variable-length gap marker) is never masked."""
        # Use text with '!' characters representing gaps of unknown length
        text = "αβγ!δεζ!ηθι"
        positions = [i + 1 for i, c in enumerate(text) if c == "!"]

        # Run masking multiple times to account for randomness
        for _ in range(10):
            data_item_copy = DataItem(text=text)
            masked_item, _ = rnn_model.mask_and_label_characters(
                data_item_copy, mask_type=mask_type
            )

            # Find positions of '!' in the token sequence
            for pos in positions:
                # '!' should never be masked
                assert masked_item.mask[pos] is False, (
                    f"'!' at position {pos} was masked in {mask_type} mode"
                )
                assert masked_item.labels[pos] == -100, (
                    f"'!' at position {pos} has label in {mask_type} mode"
                )

    @pytest.mark.parametrize("mask_type", ["random", "smart"])
    def test_mixed_lacuna_markers_not_masked(self, rnn_model: RNN, mask_type: str):
        """Test that both '.' and '!' markers are skipped when mixed in text."""
        text = "αβ.γδ!εζ.ηθ!ικλμνξοπ"
        positions = [i + 1 for i, c in enumerate(text) if c in ("!", ".")]

        for _ in range(10):
            data_item_copy = DataItem(text=text)
            masked_item, _ = rnn_model.mask_and_label_characters(
                data_item_copy, mask_type=mask_type
            )

            for pos in positions:
                assert cast(list[bool], masked_item.mask)[pos] is False, (
                    f"Lacuna marker at position {pos} was masked in {mask_type} mode"
                )
                assert cast(list[int], masked_item.labels)[pos] == -100, (
                    f"Lacuna marker at position {pos} has label in {mask_type} mode"
                )


class TestCountParameters:
    def test_count_parameters(self, caplog):
        # Create a simple model
        model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.Linear(5, 1))

        count_parameters(model)

        # Check that logging occurred
        assert "total parameter count" in caplog.text


class TestMaskInput:
    def test_once_strategy(self, caplog):
        mock_model = MagicMock()
        mock_model.mask_and_label_characters.return_value = ("masked", 5)

        data = [DataItem(text="test1"), DataItem(text="test2")]

        result_data, mask = RNN.mask_input(mock_model, data, "random", "once")

        assert mask is False
        assert len(result_data) == 2
        assert "Masking strategy is once" in caplog.text
        assert mock_model.mask_and_label_characters.call_count == 2

    def test_dynamic_strategy(self, caplog):
        mock_model = MagicMock()
        data = [DataItem(text="test1"), DataItem(text="test2")]

        result_data, mask = RNN.mask_input(mock_model, data, "smart", "dynamic")

        assert mask is True
        assert result_data is data
        assert "dynamic" in caplog.text
        assert mock_model.mask_and_label_characters.call_count == 0
