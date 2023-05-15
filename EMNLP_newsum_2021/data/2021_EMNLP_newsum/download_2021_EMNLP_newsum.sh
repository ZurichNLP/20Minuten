#!/bin/bash


script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1; pwd -P )"


bucket_name="20min-simplified-language-uzh"
object_key="EMNLP_newsum_2021_A_New_Dataset_TS_DE.zip"
download_url="https://${bucket_name}.s3.eu-central-1.amazonaws.com/${object_key}"
outfile="$script_path/$object_key"
outdir="${outfile%*.zip}"

wget -O "$outfile" "$download_url"

mkdir -p "$outdir"
unzip -q "$outfile" -d "$outdir"
