#!/usr/bin/env python3

import sys
from pathlib import Path

model_path: Path | None = None

if len(sys.argv) == 1:
    model_path = Path(
        input(
            "Please enter path to .pth file with model.specs "
            "to convert to dict from list:\n"
        )
    )
else:
    model_path = Path(sys.argv[1])

if not model_path.exists():
    print(f"No model found at {model_path}, exiting...")

import torch


def add_dropout(model: torch.nn.Module) -> None:
    if not hasattr(model, "embed_dropout_layer"):
        model.embed_dropout_layer = torch.nn.Dropout(0.0)
    if not hasattr(model, "output_dropout_layer"):
        model.output_dropout_layer = torch.nn.Dropout(0.0)


def has_new_format(model: torch.nn.Module) -> bool:
    return hasattr(model, "embed_dropout_layer") and hasattr(
        model, "output_dropout_layer"
    )


def convert_pth(path: Path, output_path: Path | None = None) -> None:
    if path.is_dir():
        print(f"Conversion error, {path} is a directory")
    if not path.exists():
        print(f"Conversion error, {path} does not exist")
    print(f"Reading model at {path}...")
    cur_model = torch.load(path, map_location="cpu", weights_only=False)

    if has_new_format(cur_model):
        print(f"{path}: already has right format, skipping")
        return

    backup_path = path.parent / f".{path.name}.bak"
    torch.save(cur_model, backup_path)

    if not isinstance(cur_model.specs, dict):
        old_specs = cur_model.specs
        cur_model.specs = {
            "embed_size": old_specs[0],
            "hidden_size": old_specs[1],
            "proj_size": old_specs[2],
            "rnn_nLayers": old_specs[3],
            "share": old_specs[4],
            "dropout_embed": old_specs[5],
            "dropout_encoder": old_specs[5],
            "dropout_output": old_specs[5],
            "masking_proportion": old_specs[6],
            "num_tokens": old_specs[7] if len(old_specs) > 7 else cur_model.num_tokens,
        }

    add_dropout(cur_model)

    print(f"saving model with reformatted specs at {output_path or path}")
    try:
        torch.save(cur_model, output_path or path)
        print("model saved")
    except Exception as e:
        print(f"Encountered error saving model: {e}")

    if output_path is None or path == output_path:
        try:
            _ = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(
                f"Unable to load newly saved model at {path} "
                f"with error {e}, "
                "swapping in backup of original model..."
            )
            backup_path.rename(path)
            print("Backup restored")
        finally:
            backup_path.unlink(missing_ok=True)


if model_path.is_dir():
    print(f"Converting all .pth files in {model_path}")
    accum = 0
    for path in model_path.glob("*.pth"):
        convert_pth(path)
        accum += 1
    if not accum:
        print("No .pth files found")
else:
    convert_pth(model_path)
