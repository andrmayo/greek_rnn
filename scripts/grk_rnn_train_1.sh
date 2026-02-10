#!/usr/bin/env/bash

/bin/hostname
echo “hello world”
nvidia-smi

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJ_ROOT"

echo $(pwd)

source .venv/bin/activate

python -c "import multiprocessing as mp; print(f'{mp.cpu_count()} CPUs available')"

mkdir -p log/strat_experiments

python greek_rnn/main.py -tr --masking random --masking-strategy once -p -ev

if [ $? -ne 0 ]; then
  echo -e "\nmain.py failed with code $?\n"
  exit $?
fi

mkdir -p greek_rnn/log
mkdir -p greek_rnn/models
mv greek_rnn/log/"$(ls -t greek_rnn/log | head -n 1)" log/strat_experiments/rand_once_"$(ls -t greek_rnn/log | head -n 1)"
mv greek_rnn/models/"$(ls -t greek_rnn/models | head -n 1)" greek_rnn/models/rand_once_"$(ls -t greek_rnn/models | head -n 1)"
