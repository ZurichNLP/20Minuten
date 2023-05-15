# %%

import spacy
import os
from tqdm import tqdm
from statistics import mean
import numpy as np



path = '/home/deephost/Documents/PHD/codes/20_min_paper/EMNLP_newsum_2021_A_New_Dataset_TS_DE/2021_ANewDatasetandEfficientBaselinesforDocument-levelTextSimplificationinGerman/data/dedup'

src_file_pattern_name = 'src.no_tag.de'
trg_file_pattern_name = 'trg.no_tag.simpde'
file_types = ['train', 'test', 'dev']
file_patterns = [src_file_pattern_name, trg_file_pattern_name]
nlp = spacy.load('de_core_news_lg')
nlp.disable_pipes('ner','lemmatizer','morphologizer','attribute_ruler' )



def underline(text):
    return '{:s}'.format('\u0332'.join(text))

def get_lines_from_file(type, file_pattern):
    file = open(os.path.join(path, type+'.'+file_pattern), 'r', encoding='utf-8')
    return file.readlines()

def get_documents(file_pattern):
    lines = []
    for type in file_types:
        lines.extend(get_lines_from_file(type, file_pattern))
    return list(tqdm(nlp.pipe(lines, n_process=2, batch_size=1000)))


def count_documents(type, file_pattern):
    lines = get_lines_from_file(type, file_pattern)
    print("Type: {} \t Count: {}".format(type, len(lines)))

def count_tokens(docs):
    return len(docs)

def get_tokens(docs):
    if isinstance(docs, list):
        return [[i.text.lower() for i in doc]  for doc  in docs]
    else:
        return [i.text.lower() for i in docs]

def count_sentences(docs):
    return sum(1 for sent in  docs.sents)

def n_grams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range (len(tokens)-n+1)]

def get_unique_ngrams(src_docs, trg_docs, n=1):
    out = 0
    unq_src_ngram = set(n_grams(src_docs, n=n))
    unq_trg_ngram = set(n_grams(trg_docs, n=n))
    if len(unq_trg_ngram) >0:
       out = len(unq_trg_ngram - unq_src_ngram) / len(unq_trg_ngram)*100
    return out

for type in file_types:
    print(underline("Number of documents"))
    count_documents(type, src_file_pattern_name)

src_docs = get_documents(src_file_pattern_name)
trg_docs = get_documents(trg_file_pattern_name)

stats = {
    "tokens_src":np.zeros(len(src_docs)),
    "tokens_trg":np.zeros(len(src_docs)),
    "sents_src":np.zeros(len(src_docs)),
    "sents_trg":np.zeros(len(src_docs)),
    "unigrams":np.zeros(len(src_docs)),
    "bigrams":np.zeros(len(src_docs)),
    "trigrams":np.zeros(len(src_docs)), 
    "compresion":np.zeros(len(src_docs))
}

#for i, (src_tok, trg_tok) in tqdm(enumerate(zip(get_tokens(src_docs), get_tokens(trg_docs)))):
for i, (src_doc, trg_doc) in tqdm(enumerate(zip(src_docs, trg_docs))):
    stats['sents_src'][i]= count_sentences(src_doc)
    stats['sents_trg'][i]= count_sentences(trg_doc) 
    stats['tokens_src'][i]= count_tokens(src_doc)
    stats['tokens_trg'][i]= count_tokens(trg_doc)
    src_tok = get_tokens(src_doc) 
    trg_tok = get_tokens(trg_doc)
    stats['unigrams'][i] = get_unique_ngrams(src_tok, trg_tok, n=1)
    stats['bigrams'][i] = get_unique_ngrams(src_tok, trg_tok, n=2)
    stats['trigrams'][i] = get_unique_ngrams(src_tok, trg_tok, n=3) 
    stats['compresion'][i] = stats['tokens_trg'][i] / stats['tokens_src'][i]



for k in list(stats.keys()):
    stats[k] = stats[k].mean()


for k in list(stats.keys()):
    print(f'{k} -> {stats[k]:.2f}')


