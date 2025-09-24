import pytest
from rnn_code.letter_tokenizer import LetterTokenizer


class TestLetterTokenizer:
    def test_init_default(self):
        tokenizer = LetterTokenizer()
        assert tokenizer.spaces_are_tokens is True
        assert tokenizer.newlines_are_tokens is True

    def test_init_custom(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        assert tokenizer.spaces_are_tokens is False
        assert tokenizer.newlines_are_tokens is False

    def test_basic_tokenization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("αβγ"))
        assert result == ["α", "β", "γ"]

    def test_square_brackets_removed(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("[αβγ]"))
        assert result == ["α", "β", "γ"]

    def test_gap_replacement(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("α<gap/>β"))
        assert result == ["α", "!", "β"]

    def test_special_chars_preserved(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize(".!#_"))
        assert result == [".", "!", "#", "_"]

    def test_iota_subscript_conversion(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        # Unicode iota subscript
        result = list(tokenizer.tokenize("α\u0345"))
        assert "ι" in result

    def test_spaces_as_tokens(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=True,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("α β"))
        assert result == ["α", " ", "β"]

    def test_spaces_ignored(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("α β"))
        assert result == ["α", "β"]

    def test_newlines_as_tokens(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=True)
        result = list(tokenizer.tokenize("α\nβ"))
        assert result == ["α", "\n", "β"]

    def test_newlines_ignored(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("α\nβ"))
        assert result == ["α", "β"]

    def test_sigma_normalization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("σςϲ"))
        assert result == ["ϲ", "ϲ", "ϲ"]

    def test_phi_normalization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("φϕ"))
        assert result == ["φ", "φ"]

    def test_theta_normalization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("θϑ"))
        assert result == ["θ", "θ"]

    def test_stigma_normalization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("Ϛϛ"))
        assert result == ["Ϛ", "Ϛ"]

    def test_digamma_normalization(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("Ϝϝ"))
        assert result == ["Ϝ", "Ϝ"]

    def test_lowercase_conversion(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("ΑΒΓ"))
        assert result == ["α", "β", "γ"]

    def test_valid_greek_chars(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        valid_chars = "αοιεϲντυρμωηκπλδγχβθφξζψ"
        result = list(tokenizer.tokenize(valid_chars))
        assert result == list(valid_chars)

    def test_invalid_greek_char_becomes_question_mark(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        # Use a Greek character not in the valid set
        result = list(tokenizer.tokenize("ϰ"))  # Greek kappa symbol
        assert "?" in result

    def test_non_greek_letter_ignored(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=False,
                                    newlines_are_tokens=False)
        result = list(tokenizer.tokenize("αaβ"))
        # 'a' is not Greek, should be filtered out
        assert result == ["α", "β"]

    def test_empty_string(self):
        tokenizer = LetterTokenizer()
        result = list(tokenizer.tokenize(""))
        assert result == []

    def test_mixed_content(self):
        tokenizer = LetterTokenizer(spaces_are_tokens=True,
                                    newlines_are_tokens=True)
        text = "α[βγ]<gap/>δ ε\n!"
        result = list(tokenizer.tokenize(text))
        expected = ["α", "β", "γ", "!", "δ", " ", "ε", "\n", "!"]
        assert result == expected