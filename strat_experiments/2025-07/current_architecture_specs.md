# Current RNN Architecture and Hyperparameter Specifications

## Model Architecture

### Core Model (RNN class in `greek_rnn.py`)
- **Model Type**: Bidirectional LSTM
- **Embedding Size**: 300
- **Hidden Size**: 300
- **Projection Size**: 150
- **RNN Layers**: 4
- **Bidirectional**: True (effective hidden size: 300 = 150 * 2)
- **Dropout**: 0.0
- **Weight Sharing**: False (separate output layer)
- **Total Parameters**: 2,350,500

### Model Layers
```python
self.embed = nn.Embedding(35, 300)
self.scale_up = nn.Linear(300, 300)
self.rnn = nn.LSTM(300, 150, num_layers=4, batch_first=True, bidirectional=True)
self.out = nn.Linear(300, 300)
self.dropout = nn.Dropout(p=0.0)
```

### Weight Initialization
- **Method**: Kaiming Normal initialization for parameters with dim > 1
- **No Xavier or other initialization schemes used**

## Training Hyperparameters

### Optimization
- **Optimizer**: AdamW
- **Learning Rate**: 0.0003 (fixed, no scheduling)
- **Weight Decay (L2)**: 0.0
- **Criterion**: CrossEntropyLoss with reduction="sum"

### Batch Configuration
- **Initial Batch Size**: 1
- **Batch Size Multiplier**: 1.4 (increases each epoch)
- **Max Epochs**: 50

### Early Stopping
- **Criteria**: Stop when train_loss < prev_train_loss AND dev_loss > prev_dev_loss
- **Patience**: None (stops immediately on first violation)

## Data Processing

### Tokenization
- **Vocabulary Size**: 35 tokens
- **Character Set**: `"αοιεϲντυρμωηκπλδγχβθφξζψϛϚϜ" + '.?<>\n!' + '_' + '#'`
- **Special Tokens**:
  - BOT: `'<'`
  - EOT: `'>'`
  - MASK: `'_'`
  - USER_MASK: `'#'`
  - UNK: `'?'`

### Masking Strategy
- **Masking Proportion**: 0.15 (15% of tokens)
- **Random Masking**: 80% MASK token, 10% random token, 10% original token
- **Smart Masking**: 1-5 contiguous spans with length distribution (48% single char, 22% two chars, 12% three chars, 18% 4-35 chars)

## Current Performance Results

### Experimental Results (from logs)
1. **Random-Dynamic**: 19.6% accuracy (best performing)
2. **Smart-Dynamic**: 18.4% accuracy
3. **Smart-Once**: 17.9% accuracy
4. **Random-Once**: 18.3% accuracy

### Training Characteristics
- **Training typically stops at epoch 1-3** due to aggressive early stopping
- **Dev accuracy plateaus around 13-19%** across strategies
- **Model shows signs of early stopping being too aggressive**

## Data Specifications

### Dataset
- **Training Set**: 75,500 texts
- **Dev Set**: 9,437 texts
- **Test Set**: 9,438 texts
- **Vocabulary**: Greek papyrus texts (MAAT corpus)
- **Task**: Fill gaps in Greek papyrus texts (lacuna reconstruction)

### Input Format
- **Lacunae represented as**: `[___]` (brackets with underscores)
- **Unknown length lacunae**: `!`
- **Control tokens added**: BOT and EOT tokens wrap each sequence