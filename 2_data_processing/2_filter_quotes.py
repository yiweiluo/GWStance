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

from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

import spacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB

nlp = spacy.load('en')
lemmatizer = nlp.vocab.morphology.lemmatizer
ps = PorterStemmer()

config = json.load(open('../config.json', 'r'))
QUOTES_DIR = '/Users/yiweiluo/scientific-debates/2_data_processing/url_quotes'

householder_verbs = pd.read_pickle(open('householder_verbs.pkl','rb'))
with open('pronouns.txt','r') as f:
    PRONOUNS = set(f.read().splitlines())

def stem(s):
    return set([ps.stem(w) for w in word_tokenize(s.lower())])

def read_stem_str(s):
    """Reads string formatted set of stems into an iterable list."""
    return s[2:-2].split("', '")

def read_quote_json(url_guid):
    """Reads json file containing quote objects associated with url_guid."""
    with open(os.path.join(QUOTES_DIR,'{}.json'.format(url_guid)),'r') as f:
        contents = f.read()
        if len(contents) > 0:
            return json.loads(contents)
        return None

def get_householder_main_v_quotes(quote_tag_dict_list):
    """Given a labeled sentence, returns the Quotes in the sentence with a Householder verb main verb."""
    good_quotes = []
    for q_dict in quote_tag_dict_list['quotes']:
        main_v_indices = [x for x in q_dict if q_dict[x] == 'main_v']
        main_v_toks = [quote_tag_dict_list['idx2text'][x] for x in sorted(main_v_indices)]
        main_v_lemmas = [lemmatizer(tok.lower(),VERB)[0] for tok in main_v_toks]
        main_v_lemmas.append('_'.join(main_v_lemmas))
        main_v_lemmas = set(main_v_lemmas)
        if len(main_v_lemmas.intersection(householder_verbs)) > 0:
            good_quotes.append(q_dict)

    return good_quotes

with open('filtering_keywords.txt','r') as f:
    lines = f.read().splitlines()
    KEYWORDS = set([l.split('\t')[0] for l in lines])
    KEYWORD_STEMS = set([l.split('\t')[1] for l in lines])

def contains_keyword(stem_set):
    """Returns True if stem_set contains a CC keyword stem"""
    return len(set(stem_set).intersection(KEYWORD_STEMS)) > 0

def prettify(clause):
    """Clean comp. clause for classification."""
    clause = re.sub('[^a-zA-Z0-9’“”"\'% \n\.]', '', clause)

    # remove initial that, add periods, capitalize first word
    split_clause = clause.split()
    if split_clause[0] == 'that':
        split_clause = split_clause[1:]
    split_clause[0] = split_clause[0].capitalize()
    clause_str = " ".join(split_clause).strip()

    if clause_str[-1] != '.':
        clause_str += "."

    return clause_str


def main():
    """Writes complement clauses of quotes (filtered down to those with a Householder stem as the quoting verb) to 'all_quote_comps.csv'.
    Then, finds stems of all tokens in filtered complement clauses. Finally, removes indirect questions and filters comp. clauses again
    by keyword stems."""

    # Load data
    df = pd.read_csv('../1_data_scraping/dedup_df.tsv',sep='\t',header=0)

    # Set-up CSV with quote comp clauses
    fieldnames = ['guid', 'sent_no', 'quote_text', 'coref']
    with open('all_quote_comps.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for n,url_guid in enumerate(df.guid):
        # Read in parsed results
        obj = read_quote_json(url_guid)
        if obj is not None:
            # Find sents containing extracted quotes
            sents_with_quotes = [s_no for s_no in obj['quote_tags'] if len(obj['quote_tags'][s_no]['quotes']) > 0]
            if len(sents_with_quotes) > 0:
                good_v_quotes = []
                for sent_no in sents_with_quotes:
                    # Get quotes with main verb that is one of interest (in Householder list)
                    householder_main_v_quotes = get_householder_main_v_quotes(obj['quote_tags'][sent_no])
                    good_v_quotes.extend(list(zip([sent_no]*len(householder_main_v_quotes),householder_main_v_quotes)))

                # Now, get the quote text to classify stance, with url_guid + sent_no, so I can recover context
                good_v_quote_texts = [(sent_no,[(obj['quote_tags'][sent_no]['idx2text'][idx],obj['coref_tags'][idx])
                                                for (idx,label) in sorted(q_dict.items(),key=lambda x: x[0])
                                                if q_dict[idx] == 'q']) for sent_no,q_dict in good_v_quotes]

                # Write to csv file
                with open('all_quote_comps.csv', 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    for tup in good_v_quote_texts:
                        writer.writerow({'guid': url_guid, 'sent_no': tup[0],
                                         'coref': ' '.join([x[1] if x[0].lower() in PRONOUNS and x[1] is not None else x[0] for x in tup[1]]),
                                        'quote_text': ' '.join([x[0] for x in tup[1]])})

        if n % 1000 == 0:
            print('Finished writing comp. clauses from URL no. {}, {}'.format(n,url_guid))
    print('Finished writing all filtered comp. clauses!\n')

    print('Adding stems to filtered comp. clauses...')
    quotes_df = pd.read_csv('all_quote_comps.csv',sep=',',header=0)
    quotes_df['quote_stems'] = quotes_df['quote_text'].apply(stem)
    quotes_df['quote_stems_coref'] = quotes_df['coref'].apply(stem)
    quotes_df = pd.read_csv('all_quote_comps_with_stems.csv',sep=',',header=0)
    quotes_df['quote_stem_list'] = quotes_df['quote_stems'].apply(read_stem_str)
    quotes_df['quote_stem_list_coref'] = quotes_df['quote_stems_coref'].apply(read_stem_str)
    print('Done! Saving...\n')
    quotes_df.to_csv('all_quote_comps_with_stems.csv',sep=',',header=True)

    print('Filtering out indirect questions...')
    QUESTION_WORDS = set(['what','who','where','which'])
    quotes_df = quotes_df.loc[quotes_df['quote_text'].apply(lambda x: x.split()[0].lower() not in QUESTION_WORDS)]
    print('Found {} comp. clauses that are not indirect questions.\n'.format(len(quotes_df)))

    print('Filtering comp. clauses by keywords...')
    #keyword_quotes_df = quotes_df.loc[quotes_df['quote_stem_list'].apply(contains_keyword)]
    keyword_coref_quotes_df = quotes_df.loc[quotes_df['quote_stem_list_coref'].apply(contains_keyword)]
    print('Found {} comp. clauses with keywords.\n'.format(len(keyword_coref_quotes_df)))

    print('Cleaning comp. clauses for classification...')
    keyword_coref_quotes_df['clean_quote'] = keyword_coref_quotes_df['quote_text'].apply(prettify)
    print('Saving...')
    keyword_coref_quotes_df[['guid','sent_no','quote_text','coref','clean_quote']].to_csv('keyword_filtered_comp_clauses.tsv'
                                                                            ,sep='\t',header=True)
    print('Done!\n')

if __name__ == "__main__":

    # Move files from inside batches
    print('Moving the following quote extraction batches to {}:\n'.format(QUOTES_DIR),
             glob.glob(os.path.join(QUOTES_DIR,'extracted_quotes_*')))
    for subdir_path in glob.glob(os.path.join(QUOTES_DIR,'extracted_quotes_*')):
        mv_files(subdir_path.split('/')[-1],QUOTES_DIR)
        shutil.rmtree(subdir_path)

    main()
