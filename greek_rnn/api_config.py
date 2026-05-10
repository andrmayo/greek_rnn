from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

SERVED_MODELS_DIR = PROJECT_ROOT / "served_models/"
DEFAULT_MODEL_NAME = "lstm-gru/"
DEFAULT_MODEL_DIR = SERVED_MODELS_DIR / DEFAULT_MODEL_NAME
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / max(
    DEFAULT_MODEL_DIR.glob("*.pth"), key=lambda f: f.stat().st_mtime
)
