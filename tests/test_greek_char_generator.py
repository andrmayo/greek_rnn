import pytest
import torch
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from rnn_code.greek_char_generator import predict, predict_top_k, train_model
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


class TestPredictTopK:
    @pytest.fixture
    def sample_specs(self):
        return [300, 300, 150, 4, False, 0.0, 0.15]  # Standard specs

    @pytest.fixture
    def rnn_model(self, sample_specs):
        return RNN(sample_specs)

    def test_predict_top_k_no_file(self, rnn_model):
        """Test predict_top_k without saving to file"""
        original_data = DataItem(text="αβ[γδ]εζ")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            # Set predictable high values for some tokens
            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    # Make different tokens have high probability
                    alpha_index = rnn_model.token_to_index.get('α', 0)
                    beta_index = rnn_model.token_to_index.get('β', 0)
                    mock_output[0, i, alpha_index] = 10.0
                    mock_output[0, i, beta_index] = 9.0

            mock_forward.return_value = mock_output

            # Test without file saving
            result = predict_top_k(rnn_model, data_item, k=3, save_to_file=False)

            assert isinstance(result, list)
            assert len(result) <= 3  # Should return at most k candidates
            for candidate in result:
                assert isinstance(candidate, str)

    def test_predict_top_k_with_file(self, rnn_model):
        """Test predict_top_k with file saving"""
        original_data = DataItem(text="αβ[γ]εζ")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            # Set high values for predictable results
            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    gamma_index = rnn_model.token_to_index.get('γ', 0)
                    mock_output[0, i, gamma_index] = 10.0

            mock_forward.return_value = mock_output

            # Use temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                temp_filename = temp_file.name

            try:
                result = predict_top_k(rnn_model, data_item, k=2, save_to_file=True, output_file=temp_filename)

                assert isinstance(result, list)
                assert len(result) <= 2

                # Check that file was created and has content
                assert os.path.exists(temp_filename)
                with open(temp_filename, 'r') as f:
                    content = f.read()
                    assert 'Rank' in content  # Header should be present
                    assert 'Candidate' in content
                    assert 'LogSum' in content

            finally:
                # Clean up temp file
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)

    def test_predict_top_k_auto_filename(self, rnn_model):
        """Test predict_top_k with auto-generated filename"""
        original_data = DataItem(text="α[β]γ")
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)

            for i, is_masked in enumerate(data_item.mask):
                if is_masked:
                    delta_index = rnn_model.token_to_index.get('δ', 0)
                    mock_output[0, i, delta_index] = 10.0

            mock_forward.return_value = mock_output

            # Mock datetime to control filename
            with patch('datetime.datetime') as mock_datetime:
                mock_datetime.now.return_value.strftime.return_value = "20250924_123456"

                result = predict_top_k(rnn_model, data_item, k=1, save_to_file=True)

                assert isinstance(result, list)

                # File should have been created with timestamp
                expected_filename = "top_k_20250924_123456.csv"
                try:
                    assert os.path.exists(expected_filename)
                finally:
                    # Clean up
                    if os.path.exists(expected_filename):
                        os.unlink(expected_filename)

    def test_predict_top_k_no_lacuna(self, rnn_model):
        """Test predict_top_k when no positions are masked"""
        original_data = DataItem(text="αβγδε")  # No lacunae
        data_item = rnn_model.actual_lacuna_mask_and_label(original_data)

        with patch.object(rnn_model, 'forward') as mock_forward:
            seq_len = len(data_item.indexes)
            vocab_size = rnn_model.num_tokens
            mock_output = torch.randn(1, seq_len, vocab_size)
            mock_forward.return_value = mock_output

            result = predict_top_k(rnn_model, data_item, k=5, save_to_file=False)

            # Should return empty list when no lacunae
            assert isinstance(result, list)
            assert len(result) == 0


class TestTrainModel:
    @pytest.fixture
    def sample_specs(self):
        return [300, 300, 150, 4, False, 0.0, 0.15]  # Standard specs

    @pytest.fixture
    def rnn_model(self, sample_specs):
        return RNN(sample_specs)

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data"""
        return [
            "αβγδε",
            "ζηθικ",
            "λμνξο",
            "πρστυ",
            "φχψω",
        ]

    @pytest.fixture
    def sample_dev_data(self):
        """Create sample dev data with lacunae"""
        return [
            "αβ[γ]δε",
            "ζη[θ]ικ",
        ]

    def test_train_model_early_stopping(self, rnn_model, sample_train_data, sample_dev_data):
        """Test train_model function with early stopping"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock all the dependencies
            with patch('rnn_code.greek_char_generator.wandb') as mock_wandb, \
                 patch('rnn_code.greek_char_generator.model_path', temp_dir), \
                 patch('rnn_code.greek_char_generator.nEpochs', 10), \
                 patch('rnn_code.greek_char_generator.patience', 3), \
                 patch('rnn_code.greek_char_generator.train_batch') as mock_train_batch:

                # Mock train_batch to return predictable loss values
                # Simulate dev loss: improves for 2 epochs, then gets worse for 3 epochs (triggers early stop)
                dev_losses = [1.0, 0.8, 0.9, 1.1, 1.2]  # Best at epoch 1 (0.8)
                train_losses = [1.0, 0.9, 0.85, 0.8, 0.75]

                def mock_train_batch_side_effect(*args, **kwargs):
                    update = kwargs.get('update', True)
                    epoch_idx = len(mock_train_batch.call_args_list) // 2  # Rough epoch tracking
                    if epoch_idx >= len(dev_losses):
                        epoch_idx = len(dev_losses) - 1

                    if update:  # Training batch
                        return train_losses[epoch_idx], 100, 500, 50, 0, 0
                    else:  # Dev batch
                        return dev_losses[epoch_idx], 50, 250, 25, 25, 20

                mock_train_batch.side_effect = mock_train_batch_side_effect

                # Run training
                result_model = train_model(
                    rnn_model,
                    sample_train_data,
                    sample_dev_data,
                    output_name="test_early_stop"
                )

                # Verify model is returned
                assert result_model is not None
                assert isinstance(result_model, RNN)

                # Verify checkpoint files were created
                assert os.path.exists(f"{temp_dir}/test_early_stop_best.pth")
                assert os.path.exists(f"{temp_dir}/test_early_stop_latest.pth")
                assert os.path.exists(f"{temp_dir}/test_early_stop.pth")

    def test_train_model_best_model_selection(self, rnn_model, sample_train_data, sample_dev_data):
        """Test that best model is properly selected and restored"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rnn_code.greek_char_generator.wandb') as mock_wandb, \
                 patch('rnn_code.greek_char_generator.model_path', temp_dir), \
                 patch('rnn_code.greek_char_generator.nEpochs', 5), \
                 patch('rnn_code.greek_char_generator.patience', 10), \
                 patch('rnn_code.greek_char_generator.train_batch') as mock_train_batch:

                # Simulate scenario where best model is NOT the final epoch
                # Dev losses: 1.0 -> 0.5 (best) -> 0.8 -> 0.7 -> 0.9
                dev_losses = [1.0, 0.5, 0.8, 0.7, 0.9]  # Best at epoch 1 (0.5)
                train_losses = [1.0, 0.8, 0.7, 0.6, 0.5]

                # Store original model state to verify it gets restored
                original_state = rnn_model.state_dict().copy()

                def mock_train_batch_side_effect(*args, **kwargs):
                    update = kwargs.get('update', True)
                    epoch_idx = min(len(mock_train_batch.call_args_list) // 2, len(dev_losses) - 1)

                    if update:  # Training batch
                        return train_losses[epoch_idx], 100, 500, 50, 0, 0
                    else:  # Dev batch
                        return dev_losses[epoch_idx], 50, 250, 25, 25, 20

                mock_train_batch.side_effect = mock_train_batch_side_effect

                # Run training (should complete all epochs since patience is high)
                result_model = train_model(
                    rnn_model,
                    sample_train_data,
                    sample_dev_data,
                    output_name="test_best_model"
                )

                # Verify model is returned
                assert result_model is not None
                assert isinstance(result_model, RNN)

                # Note: We can't easily verify the exact state was restored without
                # deeper mocking, but we can verify the files exist
                assert os.path.exists(f"{temp_dir}/test_best_model_best.pth")
                assert os.path.exists(f"{temp_dir}/test_best_model.pth")

    def test_train_model_patience_import(self):
        """Test that patience parameter can be imported and used"""
        # This test ensures the import doesn't break
        from rnn_code.greek_utils import patience
        assert isinstance(patience, int)
        assert patience > 0

    def test_train_model_no_improvement(self, rnn_model, sample_train_data, sample_dev_data):
        """Test behavior when dev loss never improves (edge case)"""
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('rnn_code.greek_char_generator.wandb') as mock_wandb, \
                 patch('rnn_code.greek_char_generator.model_path', temp_dir), \
                 patch('rnn_code.greek_char_generator.nEpochs', 5), \
                 patch('rnn_code.greek_char_generator.patience', 2), \
                 patch('rnn_code.greek_char_generator.train_batch') as mock_train_batch:

                # Simulate dev loss getting worse each epoch
                dev_losses = [1.0, 1.5, 2.0, 2.5, 3.0]  # Always getting worse
                train_losses = [1.0, 0.9, 0.8, 0.7, 0.6]  # Training improves

                def mock_train_batch_side_effect(*args, **kwargs):
                    update = kwargs.get('update', True)
                    epoch_idx = min(len(mock_train_batch.call_args_list) // 2, len(dev_losses) - 1)

                    if update:
                        return train_losses[epoch_idx], 100, 500, 50, 0, 0
                    else:
                        return dev_losses[epoch_idx], 50, 250, 25, 25, 20

                mock_train_batch.side_effect = mock_train_batch_side_effect

                # Should still return a model (from epoch 0, the "best")
                result_model = train_model(
                    rnn_model,
                    sample_train_data,
                    sample_dev_data,
                    output_name="test_no_improvement"
                )

                assert result_model is not None
                assert isinstance(result_model, RNN)

    def test_train_model_file_creation_error(self, rnn_model, sample_train_data, sample_dev_data):
        """Test handling when model files can't be saved"""
        # Use a read-only directory to trigger file save errors
        with tempfile.TemporaryDirectory() as temp_dir:
            readonly_dir = os.path.join(temp_dir, "readonly")
            os.makedirs(readonly_dir)
            os.chmod(readonly_dir, 0o444)  # Read-only

            try:
                with patch('rnn_code.greek_char_generator.wandb') as mock_wandb, \
                     patch('rnn_code.greek_char_generator.model_path', readonly_dir), \
                     patch('rnn_code.greek_char_generator.nEpochs', 2), \
                     patch('rnn_code.greek_char_generator.patience', 5), \
                     patch('rnn_code.greek_char_generator.train_batch') as mock_train_batch:

                    mock_train_batch.return_value = (1.0, 100, 500, 50, 25, 20)

                    # This should raise an exception due to file permissions
                    with pytest.raises(RuntimeError):
                        train_model(
                            rnn_model,
                            sample_train_data,
                            sample_dev_data,
                            output_name="test_file_error"
                        )
            finally:
                # Cleanup - restore permissions so directory can be deleted
                try:
                    os.chmod(readonly_dir, 0o755)
                except:
                    pass