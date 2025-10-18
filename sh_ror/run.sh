#!/bin/bash
#SBATCH --job-name=dpo_qwen_test
#SBATCH --output=logs/dpo_qwen_test_%j.out
#SBATCH --error=logs/dpo_qwen_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus=h100:1
#SBATCH --time=05:00:00                 # 2 hours for quick test
#SBATCH --mem=64G

# Quick test script for Qwen3-8B
# Tests model loading and runs 50 training steps
export HF_HUB_OFFLINE=1


module load cuda/12.2
module load python/3.12.4

source .venv/bin/activate

mkdir -p logs

python scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns


