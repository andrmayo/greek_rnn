import pytest
import torch
from unittest.mock import Mock, patch
from rnn_code.greek_char_generator import predict
from rnn_code.greek_rnn import RNN
from rnn_code.greek_utils import DataItem


class TestPredict:
    @pytest.fixture
    def sample_specs(self):
        return [300, 300, 150, 4, False, 0.0, 0.15]  # Standard specs

    @pytest.fixture
    def rnn_model(self, sample_specs):
        return RNN(sample_specs)

    def test_predict_basic(self, rnn_model):
        """Test predict function with a simple masked input"""
        # Create a data item with some masked positions
        # Using actual_lacuna_mask_and_label to get proper mask/labels setup
        original_data = DataItem(text="αβ[γδ]εζ")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        # Mock the model's forward pass to return predictable output
        with patch.object(rnn_model, 'forward') as mock_forward:
            # Create mock output tensor - shape should match (1, seq_len, vocab_size)
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            # Set specific high values for positions we want to predict
            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    # Make 'γ' token have highest probability at masked positions
                    gamma_index = rnn_model.token_to_index.get('γ', 0)
                    mock_output[0, i, gamma_index] = 10.0  # High logit

            mock_forward.return_value = mock_output

            # Call predict
            result = predict(rnn_model, data_item)

            # Verify the result is a string
            assert isinstance(result, str)
            # Should not contain control characters
            assert '<' not in result
            assert '>' not in result
            # Should contain our predicted characters
            assert 'α' in result  # Original unmasked character
            assert 'ε' in result  # Original unmasked character

    def test_predict_no_masking(self, rnn_model):
        """Test predict function when no positions are masked"""
        # Create a data item with no lacunae
        original_data = DataItem(text="αβγδε")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)
            mock_forward.return_value = mock_output

            result = predict(rnn_model, data_item)

            # Should return the original text (minus control chars)
            assert isinstance(result, str)
            assert 'αβγδε' == result

    def test_predict_all_masked(self, rnn_model):
        """Test predict function when all positions are masked"""
        # Create a data item where everything is in lacunae
        original_data = DataItem(text="[αβγ]")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            # Make specific tokens have high probability
            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    # Predict 'α' for all masked positions
                    alpha_index = rnn_model.token_to_index.get('α', 0)
                    mock_output[0, i, alpha_index] = 10.0

            mock_forward.return_value = mock_output

            result = predict(rnn_model, data_item)

            assert isinstance(result, str)
            # Result should contain predicted characters
            assert len(result) > 0
            assert '<' not in result
            assert '>' not in result

    def test_predict_mixed_content(self, rnn_model):
        """Test predict function with mixed Greek and punctuation"""
        original_data = DataItem(text="αβ[γ]δε.ζη")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            # Set predictions for masked positions
            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    theta_index = rnn_model.token_to_index.get('θ', 0)
                    mock_output[0, i, theta_index] = 10.0

            mock_forward.return_value = mock_output

            result = predict(rnn_model, data_item)

            assert isinstance(result, str)
            # Should contain original punctuation
            assert '.' in result
            # Should contain original Greek characters
            assert 'α' in result
            assert 'β' in result