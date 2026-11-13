#!/bin/bash
#SBATCH --job-name=llm_llama
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err

set -e
mkdir -p logs results/llm results/langextract

module purge
module load Python/3.12.3-GCCcore-13.3.0
module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0

cd $SLURM_SUBMIT_DIR

if [ ! -d ".venv_llm" ]; then
    python -m venv .venv_llm
    source .venv_llm/bin/activate
    pip install ollama langextract tqdm
else
    source .venv_llm/bin/activate
fi

export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_CONTEXT_LENGTH=8192
ollama serve &
OLLAMA_PID=$!
sleep 10

ollama pull llama3.1:8b

mkdir -p results/llm/$SLURM_JOB_ID results/langextract/$SLURM_JOB_ID

echo "=== LLM Zero-Shot Extraction ==="
python src/llm_extractor.py \
    --data_dir data/re_docred \
    --split dev \
    --model llama3.1:8b \
    --output_dir results/llm/$SLURM_JOB_ID \
    --mode zero_shot

echo "=== LLM Few-Shot Extraction ==="
python src/llm_extractor.py \
    --data_dir data/re_docred \
    --split dev \
    --model llama3.1:8b \
    --output_dir results/llm/$SLURM_JOB_ID \
    --mode few_shot \
    --n_examples 5

echo "=== LangExtract + Ollama Extraction ==="
python src/langextract_extractor.py \
    --data_dir data/re_docred \
    --split dev \
    --backend ollama \
    --model llama3.1:8b \
    --output_dir results/langextract/$SLURM_JOB_ID \
    --n_examples 5

kill $OLLAMA_PID

echo "=== All LLM extraction complete ==="
