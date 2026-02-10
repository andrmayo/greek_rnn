#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options, can use account acmayo0

#SBATCH --job-name=greek_rnn_train
#SBATCH --account=acmayo0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=rnn_runs/rnn_training3.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
echo “hello world”
nvidia-smi

source ~/.bashrc

cd ..

source .venv/bin/activate

python -c "import multiprocessing as mp; print(f'{mp.cpu_count()} CPUs available')"

mkdir -p log/strat_experiments

python greek_rnn/main.py -tr --masking smart --masking-strategy once -ev

mkdir -p greek_rnn/log
mv greek_rnn/log/"$(ls -t greek_rnn/log | head -n 1)" log/strat_experiments/smart_once_"$(ls -t greek_rnn/log | head -n 1)"
mv greek_rnn/models/"$(ls -t greek_rnn/models | head -n 1)" greek_rnn/models/smart_once_"$(ls -t greek_rnn/models | head -n 1)"
