#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options, can use account acmayo0

#SBATCH --job-name=greek_rnn_train
#SBATCH --account=acmayo0
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=6000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=rnn_runs/rnn_training1.out

# The application(s) to execute along with its input arguments and options:

/bin/hostname
echo “hello world”
nvidia-smi

source ~/.bashrc

cd ..

source .venv/bin/activate

python -c "import multiprocessing as mp; print(f'{mp.cpu_count()} CPUs available')"

mkdir -p log/strat_experiments

python rnn_code/main.py -tr --masking random --masking-strategy once -p -ev
mv rnn_code/log/"$(ls -t rnn_code/log | head -n 1)" log/strat_experiments/rand_once_"$(ls -t rnn_code/log | head -n 1)"
mv rnn_code/models/"$(ls -t rnn_code/models | head -n 1)" rnn_code/models/rand_once_"$(ls -t rnn_code/models | head -n 1)"
