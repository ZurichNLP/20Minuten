#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse

import boto3
from botocore.client import Config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket-name",
        type=str,
        help="Name of the S3 bucket.",
        metavar="BUCKET",
        required=True,
    )
    parser.add_argument(
        "--object-key",
        type=str,
        help="Key of the S3 object.",
        metavar="KEY",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help="Output file for the donwloaded object.",
        metavar="PATH",
        required=True,
    )
    parser.add_argument(
        "--access-key-id",
        type=str,
        help="Access key (AWS credentials).",
        metavar="STRING",
        required=True,
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        help="Secret key (AWS credentials).",
        metavar="STRING",
        required=True,
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    s3 = boto3.resource(
        "s3",
        aws_access_key_id=args.access_key_id,
        aws_secret_access_key=args.secret_key,
        config=Config(signature_version='s3v4'),
    )

    s3.Bucket(args.bucket_name).download_file(args.object_key, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
