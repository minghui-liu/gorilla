#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --account=bfpp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gres=gpu:4
#SBATCH --job-name=bfcl_sft_qwen3_full
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --time 3:00:00

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

bfcl generate \
  --model Qwen/Qwen3-8B \
  --test-category multi_turn \
  --backend vllm \
  --num-gpus 4 \
  --gpu-memory-utilization 0.9 \
  --local-model-path /u/mliu21/hdd/LLaMA-Factory/saves/qwen3-8b-full-sft/


bfcl evaluate \
  --model Qwen/Qwen3-8B \
  --test-category multi_turn


set -x 