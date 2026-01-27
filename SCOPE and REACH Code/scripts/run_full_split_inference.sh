#!/bin/bash
#SBATCH --job-name=ED_ethos_infer_array
#SBATCH --time=24:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --array=0-63
#SBATCH --output=logs/ethos_M2_ed_%A_%a.log

NUM_PARTS=32
TASKS=("hospital_mortality" "icu_admission")

TASK_IDX=$((SLURM_ARRAY_TASK_ID / NUM_PARTS))
PART_IDX=$((SLURM_ARRAY_TASK_ID % NUM_PARTS))
TASK_NAME="${TASKS[$TASK_IDX]}"

echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Task: ${TASK_NAME} (index ${TASK_IDX})"
echo "Part: ${PART_IDX} of ${NUM_PARTS}"
model_variant="recent_model.pt"
dataset_dir="data2/tokenized_meds/test"

if [[ ! -d $dataset_dir ]]; then
    echo "Dataset directory not found: $dataset_dir"
    exit 1
fi

mkdir -p logs

res_name_prefix=""
rep_num=100
for arg in "$@"; do
    if [[ $arg == res_name_prefix=* ]]; then
        res_name_prefix="${arg#res_name_prefix=}_"
    fi
    if [[ $arg == rep_num=* ]]; then
        rep_num="${arg#rep_num=}"
    fi
done

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null

# Use other tmp dir to avoid /tmp filling up and preserve the cache across the runs
export TORCHINDUCTOR_CACHE_DIR=/ethos/torchinductor_cache
"

output_dir="results/${TASK_NAME}_M2_ED"

script_body="
set -e

echo RUNNING: task=${TASK_NAME}, part=${PART_IDX}/${NUM_PARTS}, model_variant=${model_variant}
ethos_infer_split \
    task=${TASK_NAME} \
    model_fp=data2/tokenized_meds/models/layer_6_do_0.3_ED_included/recent_model.pt \
    input_dir=data2/tokenized_meds/test \
    output_dir=${output_dir} \
    output_fn=part_${PART_IDX} \
    rep_num=${rep_num} \
    n_gpus=1 \
    save_logits=false \
    no_compile=true \
    +dataset_kwargs.num_parts=${NUM_PARTS} \
    +dataset_kwargs.part_idx=${PART_IDX}
"

module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    export NUM_GPUS=1
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    NUM_GPUS=1
    export NUM_GPUS
    bash -c "${script_body}"
fi