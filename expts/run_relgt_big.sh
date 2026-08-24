#!/bin/bash
#SBATCH --job-name=relgt-large
#SBATCH --partition=epyc-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --array=0-8%1        # 9 Configs (große Datensätze), max. 1 gleichzeitig laufend
                              # -> %K an die Anzahl der euch auf Licca zur Verfügung
                              #    stehenden GPUs anpassen. Wegen des hohen
                              #    Speicherbedarfs (Amazon/H&M/Stack) hier
                              #    vorsichtshalber niedriger als bei den kleinen
                              #    Experimenten angesetzt.

set -euo pipefail
mkdir -p logs

# Dieses Skript wird aus dem Ordner "expts/" heraus submittet (sbatch run_relgt_big.sh).
# main_node_ddp.py liegt eine Ebene höher im Repo-Root.
SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
 
CONFIG_FILE="${SCRIPT_DIR}/configs_big.txt"

# Kommentarzeilen (#) und Leerzeilen rausfiltern, dann die Zeile für diesen
# Array-Task auswählen (SLURM_ARRAY_TASK_ID ist 0-indiziert, sed ist 1-indiziert)
ARGS=$(grep -v '^#' "$CONFIG_FILE" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

if [ -z "$ARGS" ]; then
    echo "Keine Config-Zeile für SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} gefunden. Abbruch."
    exit 1
fi

echo "[Task ${SLURM_ARRAY_TASK_ID}] Args: ${ARGS}"

eval "$(/hpc/gpfs2/home/u/thomasti/micromamba/bin/micromamba shell hook --shell bash)"
micromamba activate gt

export RELBENCH_CACHE_DIR=/hpc/gpfs2/scratch/u/thomasti/relbench_cache
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled

torchrun --nproc_per_node=1 \
    --master_port=$((29500 + SLURM_ARRAY_TASK_ID)) \
    "${REPO_ROOT}/main_node_ddp.py" \
    --precompute --seed 0 --num_neighbors 300 --channels 512 \
    --num_workers 8 --lr 0.0001 \
    --cache_dir "${RELBENCH_CACHE_DIR}/relbench_examples" \
    --out_dir "${SCRIPT_DIR}/results" \
    $ARGS