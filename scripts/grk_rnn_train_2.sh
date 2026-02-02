#!/usr/bin/env bash

/bin/hostname
echo “hello world”
nvidia-smi

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

source .venv/bin/activate

python -c "import multiprocessing as mp; print(f'{mp.cpu_count()} CPUs available')"

mkdir -p log/strat_experiments

python rnn_code/main.py -tr --masking random --masking-strategy dynamic -ev

mkdir -p rnn_code/log
mv rnn_code/log/"$(ls -t rnn_code/log | head -n 1)" log/strat_experiments/rand_dynam_"$(ls -t rnn_code/log | head -n 1)"
mv rnn_code/models/"$(ls -t rnn_code/models | head -n 1)" rnn_code/models/rand_dynam_"$(ls -t rnn_code/models | head -n 1)"
