#!/usr/bin/env bash
# -*- coding: utf-8 -*-


DEBUG_20min() {

    # If running interactively, set the GPU with 
    # export CUDA_VISIBLE_DEVICES=X

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    # base_model="det-large" # per_device_train_batch_size=2
    task_name=$3

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/dummy/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 32 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 10 \
        --evaluation_strategy "steps" \
        --logging_steps 10 --save_steps 10 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --load_best_model_at_end \
        --early_stopping --early_stopping_patience 5 \
        --overwrite_cache \
        --max_train_samples 100 --max_eval_samples 50 --max_predict_samples 50

}

DEBUG_20min_multi() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    # base_model="det-large" # per_device_train_batch_size=2

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/dummy/multi" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 128 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names title_generation category_prediction reading_time_prediction \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 10 \
        --evaluation_strategy "steps" \
        --logging_steps 10 --save_steps 10 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --max_train_samples 100 --max_eval_samples 50 --max_predict_samples 50


}


train_20min_title() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="title_generation"

    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 32 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"
}


train_20min_lead() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="lead_generation"

    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 64 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --load_best_model_at_end \
        --early_stopping --early_stopping_patience 5 \
        --report_to "wandb"
}

train_20min_summary() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="summary_generation"

    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 128 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"

}


train_20min_caption() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="caption_generation"
    
    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 256 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"
}


train_20min_categories() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="category_prediction"

    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 32 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"
}


train_20min_reading_time() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    task_name="reading_time_prediction"

    mkdir -p "${repo_dir}/results/${base_model}/${task_name}"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/${task_name}" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 4 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names "${task_name}" \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 2 --gradient_checkpointing \
        --max_steps 4000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"
}


train_20min_multi() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    # base_model="det-large" # per_device_train_batch_size=2

    mkdir -p "${repo_dir}/results/${base_model}/multitask"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/multitask" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 256 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names title_generation lead_generation summary_generation caption_generation category_prediction reading_time_prediction \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 4 --gradient_checkpointing \
        --max_steps 8000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 \
        --load_best_model_at_end \
        --report_to "wandb"
}

cont_train_20min_multi() {

    repo_dir=$1
    base_model=$2 # "det5-base" # per_device_train_batch_size=8
    # base_model="det-large" # per_device_train_batch_size=2

    mkdir -p "${repo_dir}/results/${base_model}/v-splits_220512/multitask_cont"

    python train.py \
        --model_name_or_path "${repo_dir}/models/${base_model}" \
        --do_train --do_eval --do_predict \
        --output_dir "${repo_dir}/results/${base_model}/v-splits_220512/multitask_cont" \
        --resume_from_checkpoint "${repo_dir}/results/${base_model}/v-splits_220512/multitask/checkpoint-3400/" \
        --overwrite_output_dir \
        --max_source_length 512 \
        --max_target_length 256 \
        --predict_with_generate --num_beams 4 \
        --local_dataset_name "twenty_min_datasets.py" \
        --task_names title_generation lead_generation summary_generation caption_generation category_prediction reading_time_prediction \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 4 --gradient_checkpointing \
        --max_steps 8000 \
        --evaluation_strategy "steps" \
        --logging_steps 200 --save_steps 200 --save_total_limit 1 \
        --metric_for_best_model "loss" \
        --early_stopping --early_stopping_patience 5 --early_stopping_threshold 0.00001 \
        --load_best_model_at_end \
        --report_to "wandb"
}
