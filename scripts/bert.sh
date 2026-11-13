#!/bin/bash
#SBATCH --job-name=bert
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err

set -e
mkdir -p logs results/bert

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

echo "=== BERT Baseline Training ==="
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python src/bert_baseline.py \
    --data_dir data/re_docred \
    --output_dir results/bert \
    --model_name distilbert-base-cased \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_docs 500

echo "=== Training complete ==="