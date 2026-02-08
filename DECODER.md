# Autoregressive Decoder Head for Lacuna Filling

## Overview

Add a toggleable autoregressive GRU decoder that generates lacuna fills
sequentially, conditioned on the biLSTM's contextual representations. When
disabled, behavior is identical to current independent per-position prediction.

## Files to Modify

1. `rnn_code/greek_utils.py` — add toggle + decoder hyperparameters
2. `rnn_code/greek_rnn.py` — add `LacunaDecoder` module, `encode()` method,
   `decode_lacuna_regions()` method, `find_contiguous_masked_regions()` utility
3. `rnn_code/greek_char_generator.py` — modify `train_batch()`,
   `predict_chars()`, `predict_top_k()`, `accuracy_evaluation()`
4. `rnn_code/main.py` — pass decoder config to model constructor
5. `tests/unit/test_greek_rnn.py` — add decoder tests

## 1. greek_utils.py — Config Toggle

Add after `specs` (line 78):

```python
decoder_specs = {
    "enabled": False,      # toggle on/off
    "hidden_size": 150,    # match proj_size
    "num_layers": 1,
}
```

Keep separate from `specs` list to avoid breaking saved models and existing
references.

## 2. greek_rnn.py — Decoder Architecture

### 2a. Module-level utility: `find_contiguous_masked_regions(mask) -> list[tuple[int, int]]`

Scans boolean mask, returns `(start, end_exclusive)` tuples for each contiguous
run of `True`.

### 2b. New class: `LacunaDecoder(nn.Module)`

**Architecture:**

- **Input projection:** `nn.Linear(proj_size + embed_size, decoder_hidden_size)`
  — projects concatenated `[biLSTM_context, prev_token_embedding]` (150+300=450
  -> 150)
- **GRU:** `nn.GRU(decoder_hidden_size, decoder_hidden_size, num_layers=1)` —
  single-layer, processes one step at a time
- **Hidden init:** `nn.Linear(proj_size, decoder_hidden_size)` with tanh —
  initializes GRU hidden state from average of boundary encoder outputs
  (positions flanking the lacuna)
- **Output projection:** `nn.Linear(decoder_hidden_size, embed_size)` -> matmul
  with transposed embedding weights (shared with encoder) -> logits over
  vocabulary
- **Start embedding:** `nn.Parameter(embed_size)` — serves as "previous token"
  for first position in a lacuna

**Key method:**
`forward_region(encoder_output, embed_layer, start, end, seq_len, teacher_labels=None) -> (region_length, num_tokens)`
logits

- With `teacher_labels`: teacher forcing (training)
- Without: autoregressive (inference)

~270K added parameters (negligible vs existing model).

### 2c. Modify `RNN.__init__()` — accept optional `decoder_specs` dict

```python
def __init__(self, specs, spaces_are_tokens=False, newlines_are_tokens=True, decoder_specs=None):
```

When `decoder_specs["enabled"]` is True, instantiate
`self.decoder = LacunaDecoder(...)`. Otherwise `self.decoder = None`.

### 2d. Add `RNN.encode(seqs)` method

Refactor from `forward()`: returns `(encoder_output, logits)` where
`encoder_output` is biLSTM output before vocab projection (shape: batch,
seq_len, proj_size).

`forward()` becomes: `return self.encode(seqs)[1]` — preserving backward
compatibility.

### 2e. Add `RNN.decode_lacuna_regions(encoder_output, mask, teacher_labels=None)`

For a single sample, finds contiguous masked regions via the utility, runs
`decoder.forward_region()` on each, returns
`dict[(start, end)] -> logits_tensor`.

## 3. greek_char_generator.py — Training & Inference Changes

### 3a. `train_batch()` (line 72)

When `model.decoder is not None`:

1. Call `model.encode([index_tensor])` to get `encoder_output` and `logits`
2. Find contiguous masked regions
3. For each region: run `model.decode_lacuna_regions()` with teacher forcing,
   compute CrossEntropyLoss on decoder output
4. Sum decoder loss across regions
5. Backward + accumulate as before

When `model.decoder is None`: existing behavior unchanged.

### 3b. `predict_chars()` (line 545)

When decoder enabled:

1. `model.encode()` to get encoder output
2. Get independent argmax predictions for all positions (fallback for
   non-masked)
3. For each contiguous masked region: run `decoder.forward_region()`
   autoregressively (no teacher labels), override predictions at those positions

### 3c. `predict_top_k()` (line 585)

When decoder enabled, replace current beam search with decoder-aware beam
search:

- For each contiguous region, expand beams through decoder steps
- At each step: for each beam, run one GRU step, expand with top-k tokens, prune
  to k beams
- Beams carry `(token_sequence, log_prob, hidden_state, prev_embedding)`
- For multiple lacuna regions: beam search each independently, Cartesian product
  to combine

### 3d. `accuracy_evaluation()` (line 432)

When decoder enabled: use `model.encode()` + decoder for contiguous regions,
same pattern as `predict_chars()`.

### 3e. `fill_masks()` and `rank_reconstructions()` — update similarly

`fill_masks` (line 388): same pattern as predict_chars. `rank_reconstructions`
(line 673): with decoder, teacher-force each proposed option and collect
per-step log probs.

## 4. main.py — Pass Config

Line 146: `model = greek_rnn.RNN(specs, decoder_specs=decoder_specs)` (import
`decoder_specs` from `greek_utils`).

Update logging to include decoder status.

## 5. Backward Compatibility

- `specs` list unchanged — old models load fine
- Code checks `getattr(model, 'decoder', None) is not None` for models saved
  before this change
- `forward()` returns identical output when decoder is disabled

## 6. Tests

Add `tests/unit/test_decoder.py`:

- `test_find_contiguous_masked_regions` — various mask patterns
- `test_decoder_init_toggle` — decoder created when enabled, None when disabled
- `test_forward_region_shapes` — output shape correctness
- `test_teacher_forcing_vs_autoregressive` — both modes produce correct shapes
- `test_encode_backward_compat` — `forward()` unchanged when decoder disabled

## 7. LSTM Decoder Variant

The `LacunaDecoder` should support hot-swapping between GRU and LSTM backends
via the `decoder_specs` config in `greek_utils.py`:

```python
decoder_specs = {
    "gru": {
        "hidden_size": proj_size if proj_size > 0 else hidden_size,
        "num_layers": 1,
    },
    "lstm": {
        "hidden_size": proj_size if proj_size > 0 else hidden_size,
        "num_layers": 1,
    },
}
```

The active decoder is selected via the `--seq` / `--sequence-decoder` CLI
option in `main.py`, which passes the decoder type string (e.g. `"gru"` or
`"lstm"`) through to the model constructor.

### Changes to `LacunaDecoder.__init__()`

Accept a `decoder_type: str` parameter and instantiate the appropriate RNN:

```python
if decoder_type == "gru":
    self.rnn = nn.GRU(
        input_size=decoder_hidden_size,
        hidden_size=decoder_hidden_size,
        num_layers=num_layers,
        batch_first=False,
    )
elif decoder_type == "lstm":
    self.rnn = nn.LSTM(
        input_size=decoder_hidden_size,
        hidden_size=decoder_hidden_size,
        num_layers=num_layers,
        batch_first=False,
    )
```

Store `self.decoder_type = decoder_type` for use in hidden state initialization.

### Changes to hidden state initialization

GRU hidden state is a single tensor `h` of shape `(num_layers, 1, hidden_size)`.
LSTM hidden state is a tuple `(h, c)` where both have shape
`(num_layers, 1, hidden_size)`.

`compute_init_hidden()` must return the correct format:

```python
h0 = torch.tanh(self.hidden_init(boundary_repr)).unsqueeze(0).unsqueeze(0)
if self.decoder_type == "lstm":
    c0 = torch.zeros_like(h0)
    return (h0, c0)
return h0
```

### Changes to `forward_region()`

The GRU step `gru_out, h = self.rnn(input, h)` and LSTM step
`lstm_out, (h, c) = self.rnn(input, (h, c))` have the same output shape for
`gru_out`/`lstm_out`, so the rest of the decoding loop (output projection,
teacher forcing, logits) remains identical. Only the hidden state unpacking
differs.

### No changes needed elsewhere

All other code (`train_batch`, `predict_chars`, `predict_top_k`, etc.)
interacts with `LacunaDecoder` through `forward_region()` and
`decode_lacuna_regions()`, which have the same interface regardless of backend.
The swap is fully encapsulated within `LacunaDecoder`.

## 8. Implementation Order

1. `greek_utils.py` — add config (no dependencies)
2. `greek_rnn.py` — utility function, LacunaDecoder class, RNN modifications
3. `greek_char_generator.py` — training/inference modifications (depends on 2)
4. `main.py` — pass config (depends on 1)
5. Tests (depends on 2-3)

## 9. Verification

1. Run existing tests with decoder disabled — should pass unchanged
2. Toggle decoder on, run `train smart -d` for a few epochs — verify loss
   decreases
3. Run `predict` and `predict-k` with decoder-trained model — verify output
4. Compare eval accuracy with decoder on vs off
