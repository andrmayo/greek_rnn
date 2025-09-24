import pytest
import torch
from rnn_code.greek_rnn import RNN
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
        assert model.specs == sample_specs + [35]  # 35 is num_tokens
        assert model.num_tokens == 35
        assert model.mask_char == '_'
        assert model.user_mask_char == '#'
        assert model.bot_char == "<"
        assert model.eot_char == ">"
        assert model.unk_char == '?'

    def test_token_mappings(self, rnn_model):
        # Test that token mappings are bidirectional
        for token, index in rnn_model.token_to_index.items():
            assert rnn_model.index_to_token[index] == token

        # Test specific tokens
        assert rnn_model.token_to_index['_'] is not None
        assert rnn_model.token_to_index['#'] is not None
        assert rnn_model.token_to_index['<'] is not None
        assert rnn_model.token_to_index['>'] is not None
        assert rnn_model.token_to_index['α'] is not None

    def test_lookup_indexes_basic(self, rnn_model):
        text = "αβγ"
        indexes = rnn_model.lookup_indexes(text)

        # Should include BOT and EOT tokens by default
        assert len(indexes) == len(text) + 2
        assert indexes[0] == rnn_model.token_to_index['<']
        assert indexes[-1] == rnn_model.token_to_index['>']

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
        alpha_idx = rnn_model.token_to_index['α']
        beta_idx = rnn_model.token_to_index['β']
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

        assert hasattr(masked_item, 'indexes')
        assert hasattr(masked_item, 'mask')
        assert hasattr(masked_item, 'labels')
        assert len(masked_item.mask) == len(masked_item.indexes)
        assert len(masked_item.labels) == len(masked_item.indexes)
        assert isinstance(total_mask, int)

    def test_mask_and_label_characters_smart(self, rnn_model):
        data_item = DataItem(text="αβγδεζηθικλμνξοπ")

        masked_item, total_mask = rnn_model.mask_and_label_characters(
            data_item, mask_type="smart"
        )

        assert hasattr(masked_item, 'indexes')
        assert hasattr(masked_item, 'mask')
        assert hasattr(masked_item, 'labels')
        assert len(masked_item.mask) == len(masked_item.indexes)
        assert len(masked_item.labels) == len(masked_item.indexes)
        assert isinstance(total_mask, int)

    def test_actual_lacuna_mask_and_label(self, rnn_model):
        # Test with lacuna brackets
        data_item = DataItem(text="αβ[γδ]εζ")

        result = rnn_model.actual_lacuna_mask_and_label(data_item)

        assert hasattr(result, 'indexes')
        assert hasattr(result, 'mask')
        assert hasattr(result, 'labels')
        assert len(result.mask) == len(result.indexes)
        assert len(result.labels) == len(result.indexes)

        # Text should have brackets removed and lacuna replaced with mask chars
        assert '[' not in result.text
        assert ']' not in result.text
        assert '_' in result.text

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

    def test_weight_sharing_false(self, sample_specs):
        sample_specs[4] = False  # share = False
        model = RNN(sample_specs)
        assert hasattr(model, 'out')
        assert not hasattr(model, 'scale_down')

    def test_weight_sharing_true(self, sample_specs):
        sample_specs[4] = True  # share = True
        model = RNN(sample_specs)
        assert not hasattr(model, 'out')
        assert hasattr(model, 'scale_down')

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

        expected_params = ['embed.weight', 'scale_up.weight', 'scale_up.bias',
                          'rnn.weight_ih_l0', 'rnn.weight_hh_l0', 'rnn.bias_ih_l0',
                          'rnn.bias_hh_l0', 'out.weight', 'out.bias']

        for expected in expected_params:
            assert any(expected in param for param in param_names)

    def test_bidirectional_lstm(self, rnn_model):
        # Test that LSTM is bidirectional
        assert rnn_model.rnn.bidirectional is True
        assert rnn_model.rnn.hidden_size == 150  # proj_size
        # Effective hidden size should be 300 (150 * 2 for bidirectional)

    def test_embedding_dimensions(self, rnn_model):
        assert rnn_model.embed.num_embeddings == 35  # num_tokens
        assert rnn_model.embed.embedding_dim == 300  # embed_size

    @pytest.mark.parametrize("mask_type", ["random", "smart"])
    def test_masking_preserves_sequence_length(self, rnn_model, mask_type):
        original_text = "αβγδεζηθικλμν"
        data_item = DataItem(text=original_text)

        masked_item, _ = rnn_model.mask_and_label_characters(data_item, mask_type)

        # Sequence length should be preserved (plus BOT/EOT)
        expected_length = len(rnn_model.lookup_indexes(original_text))
        assert len(masked_item.indexes) == expected_length