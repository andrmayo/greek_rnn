import subprocess
import sys
from pathlib import Path


def _check_model_exists() -> tuple[bool, Path]:
    path = Path(__file__).parent.parent.parent / "greek_rnn/models/best"
    if not list(path.glob("*.pth")):
        return False, path
    return True, path


def test_rank_cli():
    """Test ranking options through CLI"""

    ok, model_dir_path = _check_model_exists()
    assert ok, f"model not available to load in {model_dir_path}"

    sentence = "ἄνδρες [___] γυναῖκες"
    options = ["και", "γαρ", "τον", "αιδ", "σομ", "πετ", "ιυδ"]
    # sys.executable specifies Python interpreter
    test_args = [sys.executable, "-m", "greek_rnn.main", "rank", sentence] + options

    result = subprocess.run(test_args, capture_output=True, text=True)
    print(
        f"Using the rank option with {sentence} and {options}. Output:\n{result.stdout}"
    )

    assert result.returncode == 0
    assert "Ranking:" in result.stdout
    assert "τον" in result.stdout
