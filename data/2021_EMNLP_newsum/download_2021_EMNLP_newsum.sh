#!/bin/bash


script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1; pwd -P )"


bucket_name="20min-simplified-language-uzh"
object_key="EMNLP_newsum_2021_A_New_Dataset_TS_DE.zip"
outfile="$script_path/$object_key"
outdir="${outfile%*.zip}"


python3 download_s3_object.py \
    --bucket-name "$bucket_name" \
    --object-key "$object_key" \
    --output-file "$outfile"


mkdir -p "$outdir"
unzip -q "$outfile" -d "$outdir"
