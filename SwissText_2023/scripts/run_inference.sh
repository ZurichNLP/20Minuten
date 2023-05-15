#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --partition=volta

#######################################################################
# ACTIVATE ENV
#######################################################################

module purge
module load volta anaconda3 cuda/10.2 gcc/7.4.0
module list

eval "$(conda shell.bash hook)"
# conda init bash/
# source ~/.bashrc
conda deactivate && echo "CONDA ENV: $CONDA_DEFAULT_ENV"
conda activate 20min && echo "CONDA ENV: $CONDA_DEFAULT_ENV"

repo_dir="/net/cephfs/home/tkew/work/projects/20min"
base_model="det5-base"
data="v-splits_220512"

declare -A task_names
task_names[title_generation]=64
task_names[lead_generation]=256
task_names[summary_generation]=256
task_names[caption_generation]=256
task_names[category_prediction]=32
task_names[reading_time_prediction]=4

##############
# single-tasks
##############
for task_name in "${!task_names[@]}"; do
    # retrieve best checkpoint
    checkpoint=$(find "${repo_dir}/results/${base_model}/${data}/${task_name}" -name "checkpoint-*" -and -type d | head -n 1) # | cut -d" " -f 1)
    echo ""
    echo "Task: $task_name"
    echo "Best checkpoint: $checkpoint"
    echo "Max target length: ${task_names[$task_name]}"
    echo ""
    python ${repo_dir}/train.py --do_predict \
        --model_name_or_path "$checkpoint" \
        --output_dir "${repo_dir}/results/${base_model}/${data}/${task_name}" \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --max_target_length "${task_names[$task_name]}" \
        --per_device_eval_batch_size 128 \
        --predict_with_generate \
        --overwrite_cache

    echo ""
    echo "Finished running inference for $task_name"
    echo ""

done

############
# multi-task
############
task_name="multitask"
max_target_length=256
checkpoint=$(find "$repo_dir/results/$base_model/$task_name" -name "checkpoint-*" -and -type d | head -n 1) # | cut -d" " -f 1)
echo ""
echo "Task: $task_name"
echo "Best checkpoint: $checkpoint"
echo "Max target length: $max_target_length"
echo ""

python ${repo_dir}/train.py --do_predict \
    --model_name_or_path "$checkpoint" \
    --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
    --local_dataset_name "twenty_min_datasets.py" \
    --task_names title_generation lead_generation summary_generation caption_generation category_prediction reading_time_prediction \
    --max_target_length $max_target_length \
    --per_device_eval_batch_size 128 \
    --predict_with_generate \
    --overwrite_cache 

echo ""
echo "Finished running inference for $task_name"
echo ""