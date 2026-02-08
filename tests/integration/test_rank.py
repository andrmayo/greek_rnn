import subprocess
import sys
from pathlib import Path


def test_rank_cli():
    """Test ranking options through CLI"""
    sentence = "ἄνδρες [___] γυναῖκες"
    options = ["και", "γαρ", "τον", "αιδ", "σομ", "πετ", "ιυδ"]
    main_path = Path(__file__).parent.parent.parent / "rnn_code" / "main.py"
    # sys.executable specifies Python interpreter
    test_args = [sys.executable, "-m", "rnn_code.main", "rank", sentence] + options

    result = subprocess.run(test_args, capture_output=True, text=True)
    print(
        f"Using the rank option with {sentence} and {options}. Output:\n{result.stdout}"
    )

    assert result.returncode == 0
    assert "Ranking:" in result.stdout
    assert "τον" in result.stdout
