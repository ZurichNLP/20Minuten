#!/bin/bash


# script_path="$( cd "$(dirname "$0")" >/dev/null 2>&1 || exit 1; pwd -P )"

data_dir="resources/data/"
mkdir -p "$data_dir"

bucket_name="20min-simplified-language-uzh"
object_key="SwissText_2023_TS_DE.zip"
download_url="https://${bucket_name}.s3.eu-central-1.amazonaws.com/${object_key}"
outfile="$data_dir/$object_key"
outdir="${outfile%*.zip}"

wget -O "$outfile" "$download_url"

mkdir -p "$outdir"
unzip -q "$outfile" -d "$outdir"

outdir="$outdir/20min_0_00_000_220512/"

echo "outdir: $outdir"

# unzip each split file
for f in "$outdir"*.zip; do
    echo "unzipping $f ..."
    unzip -q "$f" -d "$outdir"
done

# also fetch the mc4 overlap data for filtering
object_key="SwissText_2023_TS_DE_articles_in_mc4.zip"
download_url="https://${bucket_name}.s3.eu-central-1.amazonaws.com/${object_key}"
outfile="$data_dir/$object_key"

wget -O "$outfile" "$download_url"

unzip -q "$outfile" -d "$outdir"

echo "Done."
