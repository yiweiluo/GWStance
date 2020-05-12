import os
import pandas as pd
import numpy as np
import json
import time
import shutil
from nltk.tokenize import sent_tokenize
import pickle
from  collections import defaultdict

import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

config = json.load(open('../config.json', 'r'))
REMOTE_PREPRO_DIR = config['REMOTE_PREPRO_DIR']
REMOTE_SCRAPE_DIR = config['REMOTE_SCRAPE_DIR']
fulltext_dir = os.path.join(REMOTE_SCRAPE_DIR,'url_texts')

with open('householder_stems.txt','r') as f:
    householder_stems = f.read().splitlines()
print('Length of householder stems:',len(householder_stems))
print(householder_stems[:3]+householder_stems[-3:])

def get_fulltext(url_guid):
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

PRONOUNS = set(['he','she','it','they','him','her','them','his','her','its','their',
               'himself','herself','itself','theirselves','themselves','hers','theirs'])
def is_pronoun(tok):
    tok = tok.lower().strip()
    return tok in PRONOUNS


def spacy_pipe(text):
    # Step 0. Run pipeline.
    doc = nlp(text)

    # Step 1. Figure out which tokens to tag w/ coreferring token
    to_coref = {}
    for clust in doc._.coref_clusters:
        for mention in clust.mentions:
            start = mention.start
            end = mention.end
            if start+1==end:
                to_coref[start] = True
            else:
                to_coref[start] = True
                for ix in range(start+1,end):
                    to_coref[ix] = False

    # Step 2. Annotate those tokens with coreferring tokens
    corefed_tokens = {}
    for token in doc:
        #print(token.i,token.text,token.head,token.dep_)
        if token.i in to_coref and to_coref[token.i]:
            #print('Main span:',token._.coref_clusters[0].main)
            #print('Main span start token_ix, end token_ix:',token._.coref_clusters[0].main.start,
            #     token._.coref_clusters[0].main.end)
            corefed_tokens[token.i] = token._.coref_clusters[0].main.text

        else:
            corefed_tokens[token.i] = None#token

    # Step 3. Go through each sentence in the doc, tag relevant parts as Q, V, S, or N
    labeled_sents = defaultdict(dict) # To fill with list of doc.sents, with tagged versions of tokens
    for sent_no,sent in enumerate(doc.sents):
        labeled_sents[sent_no]["idx2text"] = {tok.i: tok.text for tok in sent}
        labeled_sents[sent_no]["quotes"] = [] # We will add dicts with {tok.idx: tok.label} (key, value) pairs.

        VERBS = list(np.unique([token.head for token in sent if token.dep_ == 'ccomp']))
        VERBS = [v for v in VERBS if v.lemma_ in householder_stems]

        # Extract everything else for each VERB
        for VERB in VERBS:
            # Extract the rest of the quoting verb
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
                    #print('CL DEP:',ROOT.dep_)
                    ROOT = ROOT.head
                    verb_deps.append(ROOT)
                else:
                    pass
            else:
                while not is_ROOT(ROOT):
                    ROOT = ROOT.head
                assert is_ROOT(ROOT)
                if ROOT.dep_ == 'relcl' and ROOT.head.lemma_ in householder_stems:
                    ROOT = ROOT.head
                verb_deps.append(ROOT)

                root_deps = [x for x in ROOT.children if is_good_verb_dep(x.dep_)]
                for x in root_deps:
                    new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                    root_deps.extend(new_children)

                verb_deps.extend([x for x in root_deps if x != VERB and x not in verb_deps])

            NEG,IS_NEG,neg_children = None,None,None
            SUBJECT,subj_children = None,None

            for child in ROOT.children:
                if child.dep_[:5] == 'nsubj' or child.dep_ == 'expl':
                    SUBJECT = child
                    if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                        SUBJECT = SUBJECT.head.head

                if child.dep_[:3] == 'neg':
                    NEG = child
                    verb_deps.append(NEG)
                    neg_children = [c for c in NEG.children if c != ROOT]
                    for x in neg_children:
                        new_children = [c for c in x.children]
                        neg_children.extend(new_children)
                    verb_deps.extend(neg_children)
                    IS_NEG = ROOT in NEG.head.children or ROOT == NEG.head

            if (SUBJECT is None) and ROOT.dep_[-2:] == 'cl':
                SUBJECT = ROOT.head if ROOT.head.pos_ == 'NOUN' or ROOT.head.pos_ == 'PROPN' else None

            # Get rest of subject tokens
            if SUBJECT is not None:
                subj_children = [c for c in SUBJECT.children if is_good_subj_dep(c.dep_)]
                for x in subj_children:
                    new_children = [c for c in x.children if is_good_subj_dep(c.dep_)]
                    subj_children.extend(new_children)

            emb_main_verbs = [c for c in VERB.children if c.dep_ == 'ccomp']
            # ^what if change to ROOT.children??????

            for emb_main_verb in emb_main_verbs:
                # Recursively get all children of main verb of embedded clause
                children_queue = [x for x in emb_main_verb.children]
                for x in children_queue:
                    new_children = [c for c in x.children]
                    children_queue.extend(new_children)

                # Group indices by Quote component
                quote_indices = set([c.i for c in children_queue+[emb_main_verb]])
                verb_indices = set([c.i for c in verb_deps+[VERB]])
                verb_prt_indices = set([c.i for c in verb_prts+[VERB]])
                main_verb_indices = {ROOT.i}
                subj_indices = set([c.i for c in subj_children+[SUBJECT]]) if SUBJECT is not None else set()
                main_subj_indices = {SUBJECT.i} if SUBJECT is not None else set()
                neg_indices = set([c.i for c in neg_children+[NEG]]) if NEG is not None else set()
                main_neg_indices = {NEG.i} if NEG is not None else set()

                indices_per_label = {'s':subj_indices,
                                    'main_s':main_subj_indices,
                                    'n':neg_indices,
                                    'main_n':main_neg_indices,
                                    'v': verb_indices,
                                     'v_prt':verb_prt_indices,
                                    'main_v':main_verb_indices,
                                    'q': quote_indices}

                # Tagging of tokens in sent happens here
                tagged_tokens = {}
                for label in indices_per_label:
                    tagged_tokens.update({tok_index: label for tok_index in indices_per_label[label]})

                labeled_sents[sent_no]["quotes"].append(tagged_tokens)

    return labeled_sents,corefed_tokens



if __name__ == "__main__":
    df = pd.read_pickle(REMOTE_SCRAPE_DIR+'/temp_combined_df_with_ft_date_title_dedup.pkl')
    print('Length of df:',len(df))


    batch_no = 0
    if not os.path.exists(REMOTE_PREPRO_DIR+'/extracted_quotes_{}'.format(batch_no)):
        os.mkdir(REMOTE_PREPRO_DIR+'/extracted_quotes_{}'.format(batch_no))


    start_time = time.time()
    for ix in range(5000,len(df)):
        row_ix = df.index[ix]
        row = df.loc[row_ix]
        guid = row['guid']
        #print(ix,row_ix,guid)
        text = get_fulltext(guid)
        save_name = '{}.json'.format(guid)
        if len(text) > 0:
            labeled_sents,corefed_tokens = spacy_pipe(text)
            j = json.dumps({"quote_tags":labeled_sents,
                   "coref_tags":corefed_tokens})
        else:
            j = ""

        with open(REMOTE_PREPRO_DIR+'/extracted_quotes_{}/{}'.format(batch_no,save_name),'w') as f:
            f.write(j)

        #shutil.rmtree(REMOTE_SCRAPE_DIR+'/url_texts/{}.txt'.format(guid))

        if ix % 5000 == 0:
            print(ix,guid)
            print('Elapsed time in minutes:',(time.time()-start_time)/60.)

            # Divide into batches of 5000
            batch_no = ix
            if not os.path.exists(os.path.join(REMOTE_PREPRO_DIR,'extracted_quotes_{}'.format(batch_no))):
                os.mkdir(os.path.join(REMOTE_PREPRO_DIR+'/extracted_quotes_{}'.format(batch_no)))
