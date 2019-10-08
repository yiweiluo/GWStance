#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from nltk.tokenize import MWETokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
tokenizer = MWETokenizer()
for f in FACTIVES_SET:
    tokenizer.add_mwe(f.split('_'))

# SET TO LOCATION OF REPO
HOME_DIR = '/Users/yiweiluo/scientific-debates-test/'

# ASSUMES THAT YOU HAVE all_urls_meta_and_fulltext_df.pkl DOWNLOADED TO THE DATA_DIR
DATA_DIR = HOME_DIR+'data/'
os.mkdir(DATA_DIR+'processed_sents/')
PROCESSED_DIR = DATA_DIR+'processed_sents/'

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

with open(HOME_DIR+'verb_noun_cats/factives.txt') as File_obj:
    factives = File_obj.readlines()
with open(HOME_DIR+'verb_noun_cats/nonfactives.txt') as File_obj:
    nonfactives = File_obj.readlines()
FACTIVES = set([f.strip() for f in factives])
NONFACTIVES = set([f.strip() for f in nonfactives])
print(NONFACTIVES)
COMPLEMENT_VERBS = FACTIVES_SET | NONFACTIVES_SET

def nltk_pipe(orig_sent):
    toks = word_tokenize(orig_sent)
    lemmas = [lemmatizer.lemmatize(t,'v') for t in toks]
    lemmatized_sent = ' '.join(lemmas)
    mwe_toks = tokenizer.tokenize(lemmatized_sent.split())
    return (mwe_toks,orig_sent)

import spacy
nlp = spacy.load('en_core_web_sm')

def spacy_dep_parse_pipe(sent):
    doc = nlp(sent)
    index_to_token_dict = {ix_t:t for ix_t,t in enumerate(doc)}
    token_to_index_dict = {t:ix_t for ix_t,t in enumerate(doc)}

    verbs_and_ccs = []

    for ix_token,token in enumerate(doc):
        if token.dep_ == 'ccomp':
            matrix_verb = token.head
            matrix_verb_children_and_indices_queue = [(token_to_index_dict[c],c) for c in matrix_verb.children]
            matrix_verb_and_modifiers = [x for x in matrix_verb_children_and_indices_queue
                                     if x[0] <= token_to_index_dict[matrix_verb]]
            sorted_matrix_verb_and_modifiers = sorted(matrix_verb_and_modifiers,key=lambda x:x[0])
            sorted_matrix_verb_and_modifiers_text_and_labels = [(x[1].text, x[1].dep_)
                                                                for x in sorted_matrix_verb_and_modifiers]
            #out ADP prt find []
            matrix_verb_text = matrix_verb.text
            matrix_verb_ix = token_to_index_dict[matrix_verb]
            if matrix_verb_ix+1 in index_to_token_dict and\
            index_to_token_dict[matrix_verb_ix+1].dep_ == 'prt' and \
            index_to_token_dict[matrix_verb_ix+1].head == matrix_verb:
                matrix_verb_text += '_{}'.format(index_to_token_dict[matrix_verb_ix+1].text)
            #matrix_verb_modifiers = ' '.join(sorted_matrix_verb_and_modifiers_text)

            comp_clause = [(ix_token,token.text)]
            children_and_indices_queue = [(token_to_index_dict[c],c) for c in token.children]
            #print(children_and_indices_queue)

            while len(children_and_indices_queue) > 0:
                for item in children_and_indices_queue:
                    c = item[1]
                    #print('current child:',c.text)
                    comp_clause.append((token_to_index_dict[c],c.text))
                    #print('new comp clause:',comp_clause)
                    children_and_indices_queue.remove(item)
                    children_and_indices_queue.extend([(token_to_index_dict[c2],c2) for c2 in c.children])
                    #print('new queue:',children_queue)

            in_order_comp_clause = sorted(comp_clause,key=lambda x:x[0],reverse=False)
            comp_clause_text = ' '.join([x[1] for x in in_order_comp_clause])

            verbs_and_ccs.append({'mat verb modifiers':sorted_matrix_verb_and_modifiers_text_and_labels,
                   'mat verb':matrix_verb_text,
                  'comp_clause':comp_clause_text})

    return verbs_and_ccs

def do_pipeline(all_data=True,data=all_urls):
    if not all_data:
        # Load existing data instead of initializing to empty predicted_results
        all_sents_mwe_tokenized = pickle.load(open(PROCESSED_DIR+'all_sents_mwe_tokenized.pkl','rb'))
        sents_with_complement_verbs = pickle.load(open(PROCESSED_DIR+'sents_with_complement_verbs.pkl','rb'))
        sents_with_complement_clauses = pickle.load(open(PROCESSED_DIR+'sents_with_complement_clauses.pkl','rb'))
    else:
        all_sents_mwe_tokenized = {'vax':{'pro':[],'anti':[]},'cc':{'pro':[],'anti':[]}}
        sents_with_complement_verbs = {'vax':{'pro':[],'anti':[]},'cc':{'pro':[],'anti':[]}}
        sents_with_complement_clauses = {'vax':{'pro':[],'anti':[]},'cc':{'pro':[],'anti':[]}}

    print('Tokenizing sentences, lemmatizing tokens, doing MWE...')
    for ix,url in enumerate(all_urls):
        is_bad = all_url_df_is_bad_nyt[ix]
        if not is_bad:
            ft = all_url_to_fulltext[url]
            orig_url = url
            #print('original url:',orig_url)
            if ft is not None and len(ft) > 0:
                tokenized_ft = sent_tokenize(ft)
                url_topic = all_url_to_topic[url]
                url_stance = all_url_to_stance[url]
                for ix_sent,sent in enumerate(tokenized_ft):
                    #print('sentence token:',sent)
                    res = nltk_pipe(sent)
                    #print('nltk pipe result:',res)
                    all_sents_mwe_tokenized[url_topic][url_stance].append((res,'sent_no_{} of {}'.format(ix_sent,orig_url)))
    for topic in ['vax','cc']:
        for stance in ['anti','pro']:
            print('Number of {}-{} sentences:'.format(stance,topic),len(all_sents_mwe_tokenized[topic][stance]))
    pickle.dump(all_sents_mwe_tokenized,open(PROCESSED_DIR+'all_sents_mwe_tokenized.pkl','wb'))

    print('Filtering to sentences with complement verbs...')
    for topic in ['vax','cc']:
        for stance in ['anti','pro']:
            tokenized_sents = all_sents_mwe_tokenized[topic][stance]
            for ix,s in enumerate(tokenized_sents):
                s_toks = s[0][0]
                s_toks_complement_verbs = [ix_t for ix_t,t in enumerate(s_toks) if t in COMPLEMENT_VERBS]
                if len(s_toks_complement_verbs) > 0:
                    sents_with_complement_verbs[topic][stance].append((s,s_toks_complement_verbs))
            print('Number of filtered {}-{} sentences:'.format(stance,topic),len(sents_with_complement_verbs[topic][stance]))
    pickle.dump(sents_with_complement_verbs,open(PROCESSED_DIR+'sents_with_complement_verbs.pkl','wb'))

    print('Finding complement clauses to sentences...')
    for topic in ['vax','cc']:
        print('Doing '+topic)
        for stance in ['pro','anti']:
            print('Stance:{}'.format(stance))
            for ix_sent in range(0,len(sents_with_complement_verbs[topic][stance])):#,sent in enumerate(sents_with_complement_verbs[topic][stance]):
                sent = sents_with_complement_verbs[topic][stance][ix_sent]
                sent_text = sent[0][0][1]
                parsed_res = spacy_dep_parse_pipe(sent_text)
                if len(parsed_res) > 0:
                    sents_with_complement_clauses[topic][stance].append((sent,parsed_res))
                if ix_sent % 100 == 0:
                    print(ix_sent)
            print('Number of {}-{} sentences w/ complement clauses:'.format(stance,topic),len(sents_with_complement_clauses[topic][stance]))
    pickle.dump(sents_with_complement_clauses,open(PROCESSED_DIR+'sents_with_complement_clauses.pkl','wb'))
