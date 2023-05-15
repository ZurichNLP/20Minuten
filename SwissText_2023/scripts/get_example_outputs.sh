#!/usr/bin/env bash
# -*- coding: utf-8 -*-

results="../results/det5-base/v-splits_220512"

# iterate through multitask results

# category_prediction/predict_dataset.json


ids=$(grep -Po '"id":"\d+"' $results/multitask/predict_dataset.json | sort | uniq -c | grep -P '6 "id"' | grep -Po '\d\d+' | shuf | head -n 1) 

# https://stackoverflow.com/questions/24628076/convert-multiline-string-to-array
SAVEIFS=$IFS    # Save current IFS (Internal Field Separator)
IFS=$'\n'       # Change IFS to newline char
ids=($ids)      # split the '\n'-string into an array by the same name
# IFS=$SAVEIFS    # Restore original IFS

for (( i=0; i<${#ids[@]}; i++ ))
do
    echo "$i: ${ids[$i]}"

    # get line numbers from prediction dataset (these should match line nums in generated_predicitions.txt)
    mline_nums=$(grep -n "${ids[$i]}" "$results/multitask/predict_dataset.json" | grep -Po "^\d+")
    mline_nums=($mline_nums)

    for task in caption_generation category_prediction lead_generation reading_time_prediction summary_generation title_generation
    do
        echo "SINGLE: $task"
        line_num=$(grep -n "${ids[$i]}" "$results/$task/predict_dataset.json" | grep -Po "^\d+")
        sed -n "${line_num}p" "$results/$task/predict_dataset.json"
        sed -n "${line_num}p" "$results/$task/generated_predictions.txt"
        echo ""
    done

    for (( j=0; j<${#mline_nums[@]}; j++ ))
    do
        echo "MULTI:"
        # echo "${mline_nums[$j]}"
        sed -n "${mline_nums[$j]}p" "$results/multitask/predict_dataset.json"
        sed -n "${mline_nums[$j]}p" "$results/multitask/generated_predictions.txt"
        echo ""
    done

done