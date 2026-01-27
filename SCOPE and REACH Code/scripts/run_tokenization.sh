#!/bin/bash
#SBATCH --job-name=ethos_tokenize
#SBATCH --time=6:00:00
#SBATCH --partition=tier3q
#SBATCH --output=ethos_tokenize.log

set -e

input_dir="scripts/meds/ed/data"
output_dir="data2/tokenized_meds"

ethos_tokenize \
    input_dir=$input_dir/train \
    output_dir=$output_dir \
    out_fn=train

ethos_tokenize \
    input_dir=$input_dir/test \
    vocab=$output_dir/train \
    output_dir=$output_dir \
    out_fn=test