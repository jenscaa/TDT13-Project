# Large Language Models for Relation Extraction in Unstructured Texts

This repository is meant as code appendix for the TDT13 project report.

## Run locally (not recommended)

### Clone the repository

```bash
git clone https://github.com/jenscaa/TDT13-Project.git
cd TDT13-Project
```

### Download Re-DocRED

```bash
git clone https://github.com/tan92hl/Re-DocRED.git /tmp/Re-DocRED
mkdir -p data/re_docred
cp /tmp/Re-DocRED/data/{train,dev,test}_revised.json data/re_docred/
```

### Install dependencies

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama serve
```

### Run the experiments

```bash
python src/bert.py --data_dir data/re_docred --output_dir results/bert \
    --model_name distilbert-base-cased --epochs 5 --batch_size 16 --lr 2e-5 --max_docs 500

python src/llm_extractor.py --data_dir data/re_docred --split dev \
    --model llama3.1:8b --output_dir results/llm --mode zero_shot

python src/llm_extractor.py --data_dir data/re_docred --split dev \
    --model llama3.1:8b --output_dir results/llm --mode few_shot --n_examples 5

python src/langextract_extractor.py --data_dir data/re_docred --split dev \
    --backend ollama --model llama3.1:8b --output_dir results/langextract --n_examples 5
```

## Run on Idun

```bash
ssh <username>@idun-login1.hpc.ntnu.no
git clone https://github.com/jenscaa/TDT13-Project.git
cd TDT13-Project
```

Create the environment for the BERT job (once, on a login node):

```bash
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

Submit jobs from the repository root:

```bash
sbatch scripts/bert.sh
sbatch scripts/llm_llama.sh
sbatch scripts/llm_gemma.sh
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/bert_<jobid>.out
tail -f logs/llm_<jobid>.out
scancel <jobid>
```

## Evaluation

```bash
python src/evaluate.py data/re_docred/dev_revised.json <predictions.json>
```
