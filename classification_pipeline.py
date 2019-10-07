#!/usr/bin/env python
# coding: utf-8

import pickle
import os
from nltk.tokenize import sent_tokenize, word_tokenize, MWETokenizer
tokenizer = MWETokenizer()

# SET TO LOCATION OF REPO
HOME_DIR = '/Users/yiweiluo/scientific-debates-test/'

# ASSUMES THAT YOU HAVE all_urls_meta_and_fulltext_df.pkl DOWNLOADED TO THE DATA_DIR
DATA_DIR = HOME_DIR+'data/'
os.mkdir(DATA_DIR+'filtered_sents/')
FILTERED_DIR = DATA_DIR+'filtered_sents/'

# Load urls and text
all_url_df = pd.read_pickle(DATA_DIR+"all_urls_meta_and_fulltext_df.pkl")
all_url_df_ixs = []
for ix_url,url in enumerate(list(all_url_df['url'])):
    all_url_df_ixs.append(ix_url)
#all_url_df_url_to_ix_dict = dict(zip(list(all_url_df['url']),all_url_df_ixs))
all_urls = list(all_url_df['url'])
all_url_to_fulltext = dict(zip(list(all_url_df['url']),list(all_url_df['fulltext'])))
all_url_df_is_bad_nyt = list(all_url_df['bad NYT'])
all_url_to_topic = dict(zip(list(all_url_df['url']),list(all_url_df['topic'])))
all_url_to_stance = dict(zip(list(all_url_df['url']),list(all_url_df['stance'])))
#print(len(all_urls))

# Set lists of verbs to use for each category
with open(HOME_DIR+'verb_noun_cats/factives.txt') as File_obj:
    factives = File_obj.readlines()
with open(HOME_DIR+'verb_noun_cats/nonfactives.txt') as File_obj:
    nonfactives = File_obj.readlines()
FACTIVES = set([f.strip() for f in factives])
NONFACTIVES = set([f.strip() for f in nonfactives])
COMPLEMENT_VERBS = FACTIVES_SET | NONFACTIVES_SET

for f in FACTIVES_SET:
    tokenizer.add_mwe(f.split('_'))


all_sents = {'vax':{'pro':[],'anti':[]},'cc':{'pro':[],'anti':[]}}
for ix,url in enumerate(all_urls):
    is_bad = all_url_df_is_bad_nyt[ix]
    if not is_bad:
        ft = all_url_to_fulltext[url]
        if ft is not None and len(ft) > 0:
            tokenized_ft = sent_tokenize(ft)
            url_topic = all_url_to_topic[url]
            url_stance = all_url_to_stance[url]
            all_sents[url_topic][url_stance].extend(tokenized_ft)
    if ix % 100 == 0:
        print(ix)

print(len(all_sents['vax']['pro']))
print(len(all_sents['vax']['anti']))
print(len(all_sents['cc']['pro']))
print(len(all_sents['cc']['anti']))

all_sents_tokenized = {'vax':{'pro':[tokenizer.tokenize(sent.split()) for sent in all_sents['vax']['pro']],
                              'anti':[tokenizer.tokenize(sent.split()) for sent in all_sents['vax']['anti']]},
                       'cc':{'pro':[tokenizer.tokenize(sent.split()) for sent in all_sents['cc']['pro']],
                             'anti':[tokenizer.tokenize(sent.split()) for sent in all_sents['cc']['anti']]}}

sents_with_complement_verbs = {'vax':{'pro':[],'anti':[]},'cc':{'pro':[],'anti':[]}}

for topic in ['vax','cc']:
    for stance in ['anti','pro']:
        tokenized_sents = all_sents_tokenized[topic][stance]
        for ix,s in enumerate(tokenized_sents):
            #s_toks = set(s)
            s_toks_complement_verbs = [ix_t for ix_t,t in enumerate(s) if t in COMPLEMENT_VERBS]
            if len(s_toks_complement_verbs) > 0:
                sents_with_complement_verbs[topic][stance].append((s,s_toks_complement_verbs))

print(len(sents_with_complement_verbs['vax']['pro']))
print(len(sents_with_complement_verbs['vax']['anti']))
print(len(sents_with_complement_verbs['cc']['pro']))
print(len(sents_with_complement_verbs['cc']['anti']))

pickle.dump(sents_with_complement_verbs,open(FILTERED_DIR+'sents_with_complement_verbs.pkl','wb'))
