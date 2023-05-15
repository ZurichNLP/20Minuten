#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --partition=volta

# Author: Nicolas Spring / T. Kew
# sbatch jobs/run_finetune.sh -t 20min_lead

#######################################################################
# HANDLING COMMAND LINE ARGUMENTS
#######################################################################

# cwd="$(dirname $(readlink -fm $0))"
# echo "$cwd"
# base=$cwd/..
# cd "$base"
# echo "$base"

base=""
task=""

# arguments that are not supported
print_usage() {
    script=$(basename "$0")
    >&2 echo "Usage: "
    >&2 echo "$script -t [task] -b [base]"
}

# missing arguments that are required
print_missing_arg() {
    missing_arg=$1
    message=$2
    >&2 echo "Missing: $missing_arg"
    >&2 echo "Please provide: $message"
}

# argument parser
while getopts "b:t:" flag; do
  case "${flag}" in
    b) base="$OPTARG" ;;
    t) task="$OPTARG" ;;
    *) print_usage
       exit 1 ;;
  esac
done

# checking required arguments
if [[ -z $task ]]; then
    print_missing_arg "[-t task]" "task to finetune on ( see functions in ../experiments.sh )"
    exit 1
fi
if [[ -z $base ]]; then
    print_missing_arg "[-b base]" "Base directory of the repository"
    exit 1
fi


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

#######################################################################
# IMPORTING FUNCTIONS
#######################################################################

source $base/experiments.sh

#######################################################################
# LAUNCH EXPERIMENT
#######################################################################

case "$task" in
    # "imdb_debug") 
    #     echo "Submitting imdb_debug finetuning" && exit
    #     ;;
    # "cnn") 
    #     echo "Submitting CNN finetuning" && train_cnn_dailymail
    #     ;;
    "title_generation") 
        echo "Submitting 20min title finetuning" && train_20min_title "$base" "det5-base"
        ;;
    "lead_generation") 
        echo "Submitting 20min lead finetuning" && train_20min_lead "$base" "det5-base"
        ;;
    "summary_generation") 
        echo "Submitting 20min summary finetuning" && train_20min_summary "$base" "det5-base"
        ;;
    "caption_generation") 
        echo "Submitting 20min caption finetuning" && train_20min_caption "$base" "det5-base"
        ;;
    "category_prediction")
        echo "Submitting 20min category finetuning" && train_20min_categories "$base" "det5-base"
        ;;
    "reading_time_prediction")
        echo "Submitting 20min reading time finetuning" && train_20min_reading_time "$base" "det5-base"
        ;;
    "multitask")
        echo "Submitting 20min multitask finetuning" && train_20min_multi "$base" "det5-base"
        ;;
    "cont_multitask")
        echo "Submitting 20min continue training..." && cont_train_20min_multi "$base" "det5-base"
        ;;
    *) 
        echo "No job submitted for task: $task" && exit 1 
        ;;
esac
