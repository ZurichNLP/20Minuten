#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""

20 Minuten articles appear in the mC4 dataset used to train mT5.

This script aims to remove compromised articles from the existing test set to ensure minimal data leakage.

It also checks for overlap between training, test and validation splits and removes any compromised items from the 

python data_cleaning.py \
    -i ../resources/data/20min/split/ \
    -m ../resources/data/20min_articles_in_mc4.jsonl \
    -k title

"""

import argparse
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json
import shutil
from collections import Counter
from datasets import load_dataset


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--orig_splits', type=Path, required=True, default=None, help='Path to original test set directory containing articles in JSON file format.')
    ap.add_argument('-m', '--mc4_articles', type=Path, required=False, default=None, help='Path to file containing compromised data examples from the mC4 corpus. If not provided, will iterate through mC4 to collect them on the fly, which is very time consuming.')
    ap.add_argument('-o', '--output_path', type=Path, required=False, default=None, help='Path to newly filtered dataset.')
    ap.add_argument('-d', '--date_threshold', type=str, required=False, default='2019-10', help='Y%-m% formatted date string indicating earliest data points in 20 Minuten corpus')
    ap.add_argument('-k', '--key', type=str, required=False, default='title', help='data key to use for checking overlap, e.g. title, url, etc.')
    return ap.parse_args()

def parse_date(date_str):
    """
    Extracts the y-m-d of a given date string. 
    
    Expected format:
        - `2008-09-03T20:56:35.450686Z` (anything after 'T' is ignored!)
    """
    try: # parse date
        date = date_str.split('T')[0] # only interested in day of publication
        date = datetime.strptime(date, '%Y-%m-%d')
    except Exception as e:
        print(f"{e}: Failed to parse date from {date_str}")
        date = None
    return date

def parse_article_url(url_str, key):
    """
    Extracts the url stem (title and article ID) from urls stored in 20 Minuten and mC4. 
    
    Expected format: 
        - /story/hess-will-hunde-gegen-demonstranten-einsetzen-876119369576
        - https://www.20min.ch/story/messi-oder-ronaldo-viele-experten-sind-sich-einig-928036379459
 
    """
    url = url_str.split('/')[-1]
    title_and_id = url.split('-')
    title = '-'.join(title_and_id[:-1])
    id_ = title_and_id[-1]
    
    if key == 'title':
        return title.lower()
    elif key == 'url':
        return url.lower()
    else:
        raise NotImplementedError('Unknown key specified')
    
def fetch_mc4_articles(mc4_articles, date_threshold):
    """
    Iterates over the entire mC4 dataset and extracts any article taken from 20min.ch.
    Note, given the sheer size of mC4, this takes a long time (e.g. 32 hours!).
    """

    mc4_data = load_dataset('mc4', 'de', streaming=True)
    c = 0

    date_threshold = datetime.strptime(date_threshold, '%Y-%m') # convert string to datetime obj

    with open(mc4_articles, 'w', encoding='utf8') as outf:
        for i, ex in tqdm(enumerate(mc4_data['train'])):
            if '20min.ch' in ex['url']:
                ex_date = parse_date(ex['timestamp'])
                if ex_date and ex_date > date_threshold:
                    outf.write(f'{json.dumps(ex, ensure_ascii=False, indent=None)}\n')
                    c += 1

    print(f'Found {c} potentially overlapping articles in mc4')

    return mc4_articles

def collect_mc4_article_info(mc4_articles, key):
    """
    Collects date and url or title info from mC4 articles collected from 20min.ch
    """
    
    date_counter = Counter()
    unique_items = set()

    with open(mc4_articles, 'r', encoding='utf8') as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            ex_date = parse_date(data['timestamp'])
            if ex_date:
                date_counter[ex_date] += 1
            item = parse_article_url(data['url'], key)
            if item:
                unique_items.add(item)

    return date_counter, unique_items

def collect_20min_split_article_info(split_dir, key):
    """
    Collects date and url info from a 20 Minuten split
    """

    date_counter = Counter()
    unique_items = set()

    for json_file in tqdm(sorted(split_dir.iterdir())):
        if json_file.suffix == '.json':
            with open(json_file) as f:
                data = json.load(f)
                pub_date = parse_date(data['datePublished'])
                if pub_date:
                    date_counter[pub_date] += 1
                
                item = parse_article_url(data['url'], key)
                if item:
                    unique_items.add(item)

    return date_counter, unique_items

def collect_article_info(fp, key):

    if fp.is_dir():    
        date_counter, article_items = collect_20min_split_article_info(fp, key)
            
    elif fp.is_file():
        date_counter, article_items = collect_mc4_article_info(fp, key)
    
    print(f'Dataset: {fp}')
    print(f'\tOldest article: {min(date_counter.keys())}')
    print(f'\tNewest article: {max(date_counter.keys())}')

    return date_counter, article_items


def filter_dataset_based_on_key(orig_dir_path, new_dir_path, bad_items, key):
    """
    Compares the urls of articles in a given split with the set of 'bad_urls' found in mC4 data.
    """
    if new_dir_path.exists():
        print(f'removing exsiting directory at {new_dir_path}')
        # new_dir_path.rmdir()
        shutil.rmtree(new_dir_path)
    new_dir_path.mkdir(parents=True)
    
    invalid_articles = set()
    
    for json_file in tqdm(sorted(orig_dir_path.iterdir())):
        if json_file.suffix == '.json':
            with open(json_file) as f:
                data = json.load(f)
                item = parse_article_url(data['url'], key)
                if item in bad_items:
                    invalid_articles.add(json_file)                    
                
    print(f'Number of compromised test set articles (by {key.upper()}): {len(invalid_articles)}')

    # copy files in a separate loop to ensure files are properly closed before copying
    c = 0
    for i, json_file in enumerate(tqdm(sorted(orig_dir_path.iterdir()))):
        if json_file not in invalid_articles:
            shutil.copy(str(json_file), str(new_dir_path))
            c += 1
    
    print(f'Copied {c} (out of an original {i}) test set articles to {new_dir_path}')

    return

def check_overlap_between_splits(twenty_min_data):
    """
    Looks for overlapping items between train, test and dev splits, by article title and ID.

    :twenty_min_data (dict): 
        
        dictionary with the following structure:
            {
                'train': (set of publication dates, set of urls),
                'test': (set of publication dates, set of urls),
                'dev': (set of publication dates, set of urls),
            }
    :key (string):
        
        'title' or 'url'
    """

    
    train = twenty_min_data['train'][1]
    test = twenty_min_data['test'][1]
    dev = twenty_min_data['dev'][1]
    
    train_test_ol = train.intersection(test)
    if len(train_test_ol) > 0:
        print(f'[!] Found {len(train_test_ol)} overlapping items between train and test splits.')
        print(train_test_ol)

    train_dev_ol = train.intersection(dev)
    if len(train_dev_ol) > 0:
        print(f'[!] Found {len(train_dev_ol)} overlapping items between train and validation splits.')
        print(train_dev_ol)

    dev_test_ol = dev.intersection(test)
    if len(dev_test_ol) > 0:
        print(f'[!] Found {len(dev_test_ol)} overlapping items between validation and test splits.')
        print(dev_test_ol)

    return train_test_ol, train_dev_ol, dev_test_ol


def main():
    
    args = set_args()
    
    mc4_articles = args.mc4_articles
    if mc4_articles is not None:
        if not mc4_articles.exists():
            print('mC4 articles not provided. These will be collected on the fly. This takes a while!')
            mc4_articles = fetch_mc4_articles(mc4_articles, args.date_threshold)
    
        mc4_dates, mc4_items = collect_article_info(mc4_articles, args.key)
    
    twenty_min_data = {}
    for split in ['train', 'test', 'dev']:
        twenty_min_data[split] = collect_article_info(args.orig_splits / f'20min_all_{split}', args.key)
    
    train_test_ol, train_dev_ol, dev_test_ol = check_overlap_between_splits(twenty_min_data)
    
    bad_items = {*mc4_items, *train_test_ol, *dev_test_ol, *train_dev_ol}

    filter_dataset_based_on_key(
        args.orig_splits / f'20min_all_test', 
        args.orig_splits / f'20min_all_filt_by_{args.key}_test' if args.output_path is None else args.output_path,
        bad_items,
        args.key
        )

    filter_dataset_based_on_key(
        args.orig_splits / f'20min_all_dev', 
        args.orig_splits / f'20min_all_filt_by_{args.key}_dev' if args.output_path is None else args.output_path,
        bad_items,
        args.key
        )

if __name__ == "__main__":
    main()

    

# invalid since timestamps in mC4 indicate the common crawl date, not the article!
# def filter_dataset_based_on_date_thresholds(orig_dir_path, new_dir_path, min_date, max_date):
#     """
#     """
#     if new_dir_path.exists():
#         print(f'removing exsiting directory at {new_dir_path}')
#         # new_dir_path.rmdir()
#         shutil.rmtree(new_dir_path)
#     new_dir_path.mkdir(parents=True)
    
#     invalid_articles = set()
    
#     for json_file in tqdm(sorted(orig_dir_path.iterdir())):
#         if json_file.suffix == '.json':
#             with open(json_file) as f:
#                 data = json.load(f)
#                 pub_date = parse_date(data['datePublished'])
                
#                 if min_date < pub_date > max_date:
#                     invalid_articles.add(json_file)

#     print(f'Number of compromised test set articles (by date): {len(invalid_articles)}')

#     # copy files in a separate loop to ensure files are properly closed
#     c = 0
#     for i, json_file in enumerate(tqdm(sorted(orig_dir_path.iterdir()))):
#         if json_file not in invalid_articles:
#             shutil.copy(str(json_file), str(new_dir_path))
#             c += 1
    
#     print(f'Copied {c} (out of an original {i}) test set articles to {new_dir_path}')

#     return


