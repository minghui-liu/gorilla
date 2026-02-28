#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --account=bfpp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --job-name=bfcl_eval_qwen3_base
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --time 4:00:00

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_IB_DISABLE=1
# ulimit -n 2048


set +x

module add python/3.11.9
module add cuda/12.6
# module add cmake

source .venv/bin/activate
which python3
python3 -m site

# Set the project root so results are written to the correct location
export BFCL_PROJECT_ROOT=/u/mliu21/hdd/gorilla/berkeley-function-call-leaderboard
python3 -m bfcl_eval generate \
  --model Qwen/Qwen3-8B \
  --test-category multi_turn \
  --multi-turn-system-prompt-style short \
  --backend vllm \
  --num-gpus 4 \
  --gpu-memory-utilization 0.9
  # --local-model-path /path/to/local/qwen3/model


python3 -m bfcl_eval evaluate \
  --model Qwen/Qwen3-8B \
  --test-category multi_turn


set -x 
