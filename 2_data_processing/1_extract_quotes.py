#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import json
import time
import shutil
from nltk.tokenize import sent_tokenize
import pickle
from  collections import defaultdict
import argparse

import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

config = json.load(open('../config.json', 'r'))
REMOTE_PREPRO_DIR = config['REMOTE_PREPRO_DIR']
REMOTE_SCRAPE_DIR = config['REMOTE_SCRAPE_DIR']


def get_fulltext(url_guid,fulltext_dir):
    """Reads and returns file contents of text corresponding to a URL guid."""
    with open(os.path.join(fulltext_dir,url_guid+'.txt'),'r') as f:
        lines = f.readlines()
    if len(lines) > 0:
        return lines[0]
    return ""

def is_good_verb_dep(dep):
    return dep[:3] == 'aux' or dep[:3] == 'adv' or dep == 'det' or dep == 'rel' or dep == 'prep' or \
        dep[-3:] == 'obj' or dep[-3:] == 'mod' or (dep[-4:] == 'comp' and dep != 'ccomp')

def is_verb_prt(dep):
    return dep == 'prt'

def is_good_subj_dep(dep):
    return dep != 'ccomp'

def is_ROOT(tok):
    return tok.dep_ == 'ROOT' or tok.dep_[-2:] == 'cl' or tok.dep_ == 'ccomp' or tok.dep_ == 'pcomp' or\
            (tok.dep_ == 'conj' and tok.head.dep_ == 'ROOT')

REL_PRONOUNS = set(['who', 'whom', 'whose', 'which', 'that'])
def is_rel_pronoun(tok):
    tok = tok.lower().strip()
    return tok in REL_PRONOUNS

PRONOUNS = set(['he','she','it','they','him','her','them','his','its','their',
               'himself','herself','itself','theirselves','themselves','hers','theirs'])
def is_pronoun(tok):
    tok = tok.lower().strip()
    return tok in PRONOUNS


def spacy_pipe(text,verbose=False):
    # Step 0. Run pipeline.
    doc = nlp(text)

    # Step 1. Figure out which tokens to tag w/ coreferring token
    to_coref = {}
    for clust in doc._.coref_clusters:
        for mention in clust.mentions:
            start,end = mention.start,mention.end
            if start+1==end:
                to_coref[start] = True
            else:
                to_coref[start] = True
                for ix in range(start+1,end):
                    to_coref[ix] = False

    # Step 2. Annotate those tokens with coreferring tokens
    corefed_tokens = {}
    for token in doc:
        if token.i in to_coref:
            if to_coref[token.i]:
                corefed_tokens[token.i] = token._.coref_clusters[0].main.text
            else:
                corefed_tokens[token.i] = ''
        else:
            corefed_tokens[token.i] = None

    # Step 3. Go through each sentence in the doc, annotating relevant parts
    labeled_sents = defaultdict(dict) # To fill with list of doc.sents, with annotated versions of tokens

    for sent_no,sent in enumerate(doc.sents):

        labeled_sents[sent_no]["idx2text"] = {tok.i: tok.text for tok in sent}
        labeled_sents[sent_no]["idx2lemma"] = {tok.i: tok.lemma_ for tok in sent}
        labeled_sents[sent_no]["quotes"] = [] # We will add dicts with {tok.idx: tok.label} (key, value) pairs.

        # Step A. Get verbs that are in complement clauses, then filter to Householder stems--
        # these represent the main verbs in embedded sentences.
        VERBS = set([token.head for token in sent if token.dep_ == 'ccomp'])
        VERBS = [v for v in VERBS if v.lemma_ in householder_stems]

        # Extract everything else for each VERB
        for VERB in VERBS:

            # Step C. Extract the rest of the quoting verb
            verb_deps = [x for x in VERB.children if is_good_verb_dep(x.dep_)]
            verb_prts = [x for x in VERB.children if is_verb_prt(x.dep_)]

            for x in verb_prts:
                new_children = [c for c in x.children]
                verb_prts.extend(new_children)

            for x in verb_deps:
                new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                verb_deps.extend(new_children)

            # If verb's head is not itself (i.e., verb is not the ROOT),
            # recursively trace back to ROOT, then add all children of ROOT
            ROOT = VERB
            if is_ROOT(ROOT):
                if ROOT.dep_ == 'relcl' and ROOT.head.lemma_ in householder_stems:
                    ROOT = ROOT.head
                    verb_deps.append(ROOT)
            else:
                while not is_ROOT(ROOT):
                    ROOT = ROOT.head
                if ROOT.dep_ == 'relcl' and ROOT.head.lemma_ in householder_stems:
                    ROOT = ROOT.head
                verb_deps.append(ROOT)

            root_deps = [x for x in ROOT.children if is_good_verb_dep(x.dep_)]
            for x in root_deps:
                new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                root_deps.extend(new_children)
            verb_deps.extend([x for x in root_deps if x != VERB and x not in verb_deps])

            # Step D. Get the subject and negation on main verbs.
            NEG,subj_NEG,neg_children,subj_NEG_children = None,None,None,None
            SUBJECT,subj_children = None,None

            for child in ROOT.children:

                # First pass at finding subject of verb
                if child.dep_[:5] == 'nsubj' or child.dep_ == 'expl':
                    SUBJECT = child
                    if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                        SUBJECT = SUBJECT.head.head

                # Find negation on verb
                if child.dep_[:3] == 'neg':
                    NEG = child
                    neg_children = [c for c in NEG.children if c != ROOT]
                    for x in neg_children:
                        new_children = [c for c in x.children]
                        neg_children.extend(new_children)

            # Second pass at finding subject
            if (SUBJECT is None) and ROOT.dep_[-2:] == 'cl':
                SUBJECT = ROOT.head if ROOT.head.pos_ == 'NOUN' or ROOT.head.pos_ == 'PROPN' else None

            # Third pass: If VERB is conjoined to another verb, subject is shared
            if (SUBJECT is None) and (VERB.dep_ == 'conj' and VERB.head.dep_ == 'ROOT'):
                for c in VERB.head.children:
                    if c.dep_[:5] == 'nsubj' or c.dep_ == 'expl':
                        SUBJECT = c
                        if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                            SUBJECT = SUBJECT.head.head

            # Step E. Get rest of subject tokens and negation on subject
            if SUBJECT is not None:
                subj_children = [c for c in SUBJECT.children if is_good_subj_dep(c.dep_)]
                subj_NEG = [c for c in subj_children if c.dep_ == 'neg']
                if len(subj_NEG) > 0:
                    subj_NEG = subj_NEG[0]
                    subj_children.remove(subj_NEG)
                    subj_NEG_children = [c for c in subj_NEG.children if c != ROOT]
                    for x in subj_NEG_children:
                        new_children = [c for c in x.children]
                        subj_NEG_children.extend(new_children)
                else:
                    subj_NEG = None
                for x in subj_children:
                    new_children = [c for c in x.children if is_good_subj_dep(c.dep_)]
                    subj_children.extend(new_children)

            # Step F. Find embedded comp. clause.
            emb_main_verbs = [c for c in VERB.children if c.dep_ == 'ccomp']

            for emb_main_verb in emb_main_verbs:
                children_queue = [x for x in emb_main_verb.children]
                for x in children_queue:
                    new_children = [c for c in x.children]
                    children_queue.extend(new_children)

                # Step G. Group indices by Quote component
                quote_indices = [c.i for c in children_queue+[emb_main_verb]]
                verb_indices = [c.i for c in verb_deps+[VERB]]
                verb_prt_indices = [c.i for c in verb_prts]
                main_verb_indices = [VERB.i]
                subj_indices = [c.i for c in subj_children+[SUBJECT]] if SUBJECT is not None else []
                main_subj_indices = [SUBJECT.i] if SUBJECT is not None else []
                neg_verb_indices = [c.i for c in neg_children+[NEG]] if NEG is not None else []
                main_neg_verb_indices = [NEG.i] if NEG is not None else []
                neg_subj_indices = [c.i for c in subj_NEG_children+[subj_NEG]] if subj_NEG is not None else []
                main_neg_subj_indices = [subj_NEG.i] if subj_NEG is not None else []

                indices_per_label = {
                                     'neg_s':neg_subj_indices,
                                     'main_neg_s':main_neg_subj_indices,
                                    's':subj_indices,
                                    'main_s':main_subj_indices,
                                    'neg_v':neg_verb_indices,
                                    'main_neg_v':main_neg_verb_indices,
                                    'v': verb_indices,
                                     'v_prt':verb_prt_indices,
                                    'main_v':main_verb_indices,
                                    'q': quote_indices}

                labeled_sents[sent_no]["quotes"].append(indices_per_label)

        if verbose:
            sample_output = ""
            sample_output += 'Original sentence: '+' '.join([tok.text for tok in sent])
            sample_output += '\n'
            sample_output += 'Corefed sentence: '+' '.join([corefed_tokens[tok.i]
                                                if corefed_tokens[tok.i] is not None
                                                else tok.text for tok in sent])
            id2text = labeled_sents[sent_no]["idx2text"]
            quotes = labeled_sents[sent_no]["quotes"]
            for quote in quotes:
                sample_output += '\n***** new quote ******'
                for key in quote:
                    if 's' in key:
                        sample_output += '{}:\t'.format(key)+' '.join([corefed_tokens[i]
                                                           if corefed_tokens[i] is not None
                                                           else id2text[i]
                                                           for i in sorted(quote[key])])
                        print('{}:\t'.format(key)+' '.join([corefed_tokens[i]
                                                           if corefed_tokens[i] is not None
                                                           else id2text[i]
                                                           for i in sorted(quote[key])]))
                    else:
                        sample_output += '{}:\t'.format(key)+' '.join([id2text[i]
                                                            for i in sorted(quote[key])])
                        print('{}:\t'.format(key)+' '.join([id2text[i]
                                                            for i in sorted(quote[key])]))
            return labeled_sents,corefed_tokens,sample_output

    return labeled_sents,corefed_tokens



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--debug', action="store_true", default=None,
                      help='whether to test run on smaller sample first')
    arg_parser.add_argument('--input_df_filename', type=str, default=None,
                      help='/path/to/df')
    arg_parser.add_argument('--output_dir', type=str, default=None,
                      help='where to write batched output')
    arg_parser.add_argument('--fulltext_dir', type=str, default=None,
                      help='where to source fulltext')
    args = arg_parser.parse_args()

    print('Getting Householder stems for verb filtering...')
    with open('householder_stems.txt','r') as f:
        householder_stems = f.read().splitlines()
    print('\tNumber of Householder stems:',len(householder_stems))
    print('\tSample stems:',householder_stems[:3]+householder_stems[-3:])

    df = pd.read_pickle(os.path.join(REMOTE_SCRAPE_DIR,args.input_df_filename))#,sep='\t',header=0,index_col=0)
    print('Length of df:',len(df))

    if args.debug:
        end_ix = 5
    else:
        end_ix = len(df)

    batch_no = 0
    if not os.path.exists(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no))):
        os.makedirs(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no)))

    start_time = time.time()
    for ix in range(end_ix):
        row_ix = df.index[ix]
        row = df.loc[row_ix]
        guid = row['guid']
        text = get_fulltext(guid,args.fulltext_dir)
        print(text)
        save_name = '{}.json'.format(guid)
        if len(text) > 0:
            if args.debug:
                labeled_sents,corefed_tokens,sample_output = spacy_pipe(text,verbose=True)
                with open(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'sample_output_{}.txt'.format(guid)),'w') as f:
                    f.write(sample_output)
            else:
                labeled_sents,corefed_tokens = spacy_pipe(text,verbose=verbose)
            j = json.dumps({"quote_tags":labeled_sents,
                   "coref_tags":corefed_tokens})
        else:
            j = ""

        with open(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}/{}'.format(batch_no,save_name)),'w') as f:
            f.write(j)

        #shutil.rmtree(REMOTE_SCRAPE_DIR+'/url_texts/{}.txt'.format(guid))

        if ix % 5000 == 0:
            print(ix,guid)
            print('Elapsed time in minutes:',(time.time()-start_time)/60.)

            # Divide into batches of 5000
            batch_no = ix
            if not os.path.exists(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no))):
                os.mkdir(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no)))
