# 20Minuten

This repository contains scripts and instructions for downloading the 20 Minuten ("20 Minutes") dataset.

## AWS Bucket and Credentials

- Bucket name: `20min-simplified-language-uzh`
- Access key id: `AKIA2I5XTVWVKRZDTZ4J`
- Secret access key: `TEkUXE11y7evphjde7bJ/vxDPQJlodZ7KwAnDfg+`

To configure your credentials, follow the [quickstart guide for Boto 3](https://boto3.amazonaws.com/v1/documentation/api/1.9.42/guide/quickstart.html).

## Setup

The scripts use [Boto 3](https://boto3.amazonaws.com/v1/documentation/api/1.9.42/index.html) to download files from Amazon S3. Use the following commands to clone the repository and to install the required packages (preferably in a virtual environment):

```bash
git clone https://github.com/ZurichNLP/20Minuten
cd 20Minuten

pip3 install -r requirements.txt
```

## Data Sets

### EMNLP newsum 2021

This section describes downloading the dataset described in the paper ["A New Dataset and Efficient Baselines for Document-level Text Simplification in German"](https://aclanthology.org/2021.newsum-1.16/), presented at the *Third Workshop on New Frontiers in Summarization* at EMNLP 2021.

The object key for the data is `EMNLP_newsum_2021_A_New_Dataset_TS_DE.zip`.

To download and extract the data, use the provided shell script:

```bash
bash data/2021_EMNLP_newsum/download_2021_EMNLP_newsum.sh
```

If you did not set up the [AWS credentials](#aws-bucket-and-credentials) before (or prefer not to), you can adapt the shell script. The python script `download_s3_object.py` accepts two cli arguments, `--access-key-id` and `--secret-key`, to provide the access key id and the secret access key, respectively.