import os
import pandas as pd
import numpy as np
import re
import pickle
import json
import csv
import time
import glob
import shutil
from collections import Counter
import argparse

from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB

# os.chdir('../')
# print(os.getcwd())
# print(os.listdir(os.getcwd()))
# from local_processors import mv_files
# os.chdir('./2_data_processing')

def mv_files(subdir_name,outerdir_name):
    """Moves contents of subdir_name (usually smaller batches) to outerdir_name."""
    print('Moving contents of {} to {}...'.format(subdir_name,outerdir_name))
    print('Size of outerdir:',len(os.listdir(outerdir_name)))
    inner_fs = os.listdir(os.path.join(outerdir_name,subdir_name))
    print('Size of subdir:',len(inner_fs))
    for f in inner_fs:
        os.rename(os.path.join(outerdir_name,subdir_name,f),os.path.join(outerdir_name,f))
    print('New size of outerdir:',len(os.listdir(outerdir_name)))
    shutil.rmtree(os.path.join(outerdir_name,subdir_name))

nlp = spacy.load('en')
lemmatizer = nlp.vocab.morphology.lemmatizer
ps = PorterStemmer()

config = json.load(open('../config.json', 'r'))
BASE_DIR = config["BASE_DIR"]
QUOTES_DIR = os.path.join(BASE_DIR,'2_data_processing','url_quotes')

householder_verbs = pd.read_pickle(open('householder_verbs.pkl','rb'))
with open('pronouns.txt','r') as f:
    PRONOUNS = set(f.read().splitlines())

def read_stem_str(s):
    """Reads string formatted set of stems into an iterable list."""
    return s[2:-2].split("', '")

def stem(s):
    return [ps.stem(w) for w in word_tokenize(s.lower())]

def read_quote_json(url_guid,quotes_dir):
    """Reads json file containing quote objects associated with url_guid."""
    with open(os.path.join(quotes_dir,'{}.json'.format(url_guid)),'r') as f:
        contents = f.read()
        if len(contents) > 0:
            return json.loads(contents)
        return None

def get_householder_main_v_quotes(quote_tag_dict_list):
    """Given a labeled sentence, returns the Quotes in the sentence with a Householder verb main verb."""
    good_quotes = [(q_no,q_dict) for q_no,q_dict in enumerate(quote_tag_dict_list['quotes'])]
    #for q_no,q_dict in enumerate(quote_tag_dict_list['quotes']):
        #main_v_indices = q_dict['main_v']#[x for x in q_dict if q_dict[x] == 'main_v']
        #main_v_lemmas = [quote_tag_dict_list['idx2lemma'][x] for x in sorted(main_v_indices,key=lambda x:int(x))]
        #main_v_lemmas = [lemmatizer(tok.lower(),VERB)[0] for tok in main_v_toks]
        #main_v_lemmas.append('_'.join(main_v_lemmas))
        #main_v_lemmas = set(main_v_lemmas)
        #if len(main_v_lemmas.intersection(householder_verbs)) > 0:
        #good_quotes.append((q_no,q_dict))

    return good_quotes

with open('filtering_keywords.txt','r') as f:
    lines = f.read().splitlines()
    KEYWORDS = set([l.split('\t')[0] for l in lines])
    KEYWORD_STEMS = set([l.split('\t')[1] for l in lines])

def contains_keyword(stem_set):
    """Returns True if stem_set contains a CC keyword stem"""
    return len(set(stem_set).intersection(KEYWORD_STEMS)) > 0


def main(df_path,output_dir,quotes_dir):
    """Writes complement clauses of quotes (filtered down to those with a Householder stem as the quoting verb) to 'all_quote_comps.csv'.
    Then, finds stems of all tokens in filtered complement clauses. Finally, removes indirect questions and filters comp. clauses again
    by keyword stems and writes to `keyword_filtered_comp_clauses.tsv`."""

    # Load data
    df = pd.read_csv(df_path,sep='\t',header=0)

    # Set-up CSV with quote comp clauses
    fieldnames = ['guid', 'sent_no', 'quote_no', 'quote_text', 'coref']
    with open('./{}/all_quote_comps.csv'.format(output_dir), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for n,url_guid in enumerate(df.guid):

        # Read in parsed results
        obj = read_quote_json(url_guid,quotes_dir)
        if obj is None:
            pass
        else:
            # Find sents containing extracted quotes
            sents_with_quotes = [s_no for s_no in obj['quote_tags'] if len(obj['quote_tags'][s_no]['quotes']) > 0]
            if len(sents_with_quotes) == 0:
                pass

            good_v_quotes = []
            for sent_no in sents_with_quotes:
                # Get quotes with main verb that is one of interest (in Householder list)
                #householder_main_v_quotes = get_householder_main_v_quotes(obj['quote_tags'][sent_no])
                quote_tag_dict_list = obj['quote_tags'][sent_no]
                #good_v_quotes.extend(list(zip([sent_no]*len(householder_main_v_quotes),householder_main_v_quotes)))
                good_v_quotes.extend([(sent_no,q_no,q_dict) for q_no,q_dict in enumerate(quote_tag_dict_list['quotes'])])

            # Now, get the quote text to classify stance, with url_guid + sent_no, so I can recover context
            good_v_quote_texts = [(sent_no,q_no,[(obj['quote_tags'][sent_no]['idx2text'][str(idx)],obj['coref_tags'][str(idx)])
                                            for idx in sorted(q_dict['q'],key=lambda x: int(x))]) for sent_no,q_no,q_dict in good_v_quotes]


            with open('./{}/all_quote_comps.csv'.format(output_dir), 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for sent_no,q_no,q_text in good_v_quote_texts:
                    writer.writerow({'guid': url_guid, 'sent_no': sent_no, 'quote_no': q_no,
                                     'coref': ' '.join([x[1] if x[0].lower() in PRONOUNS and x[1] is not None
                                                           else x[0] for x in q_text]),
                                    'quote_text': ' '.join([x[0] for x in q_text])})

        if n % 1000 == 0:
            print(n,url_guid)
    print('Finished writing all filtered comp. clauses!\n')

    print('Adding stems to filtered comp. clauses...')
    quotes_df = pd.read_csv('./{}/all_quote_comps.csv'.format(output_dir),sep=',',header=0)
    quotes_df['quote_stems'] = quotes_df['quote_text'].apply(stem)
    quotes_df['quote_stems_coref'] = quotes_df['coref'].apply(stem)
    print('Done! Saving...\n')
    quotes_df.to_csv('./{}/all_quote_comps_with_stems.csv'.format(output_dir),sep=',',header=True)

    quotes_df = pd.read_csv('./{}/all_quote_comps_with_stems.csv'.format(output_dir),sep=',',header=0)
    print('Filtering out indirect questions...')
    QUESTION_WORDS = set(['what','who','where','which'])
    quotes_df = quotes_df.loc[quotes_df['quote_text'].apply(lambda x: x.split()[0].lower() not in QUESTION_WORDS)]
    print('Found {} comp. clauses that are not indirect questions.\n'.format(len(quotes_df)))

    print('Filtering comp. clauses by keywords...')
    quotes_df['quote_stem_list'] = quotes_df['quote_stems'].apply(read_stem_str)
    quotes_df['quote_stem_list_coref'] = quotes_df['quote_stems_coref'].apply(read_stem_str)
    keyword_coref_quotes_df = quotes_df.loc[quotes_df['quote_stem_list_coref'].apply(contains_keyword)].copy()
    print('Found {} comp. clauses with keywords.\n'.format(len(keyword_coref_quotes_df)))
    print('Saving...')
    keyword_coref_quotes_df.to_csv('./{}/keyword_filtered_comp_clauses.tsv'.format(output_dir),sep='\t',header=True)
    print('Done!\n')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--path_to_df', type=str, default=None,
                      help='/path/to/df')
    arg_parser.add_argument('--output_dir', type=str, default=None,
                      help='where to write batched output')
    arg_parser.add_argument('--quotes_dir', type=str, default=None,
                      help='where to source fulltext')

    args = arg_parser.parse_args()

    # Move files from inside batches
    print('Moving the following quote extraction batches to {}:\n'.format(args.quotes_dir),
             glob.glob(os.path.join(args.quotes_dir,'extracted_quotes_*')))
    for subdir_path in glob.glob(os.path.join(args.quotes_dir,'extracted_quotes_*')):
        mv_files(subdir_path.split('/')[-1],args.quotes_dir)

    main(args.path_to_df,args.output_dir,args.quotes_dir)
