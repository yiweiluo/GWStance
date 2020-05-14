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
fulltext_dir = os.path.join(REMOTE_SCRAPE_DIR,'url_texts')

with open('householder_stems.txt','r') as f:
    householder_stems = f.read().splitlines()
print('Length of householder stems:',len(householder_stems))
print(householder_stems[:3]+householder_stems[-3:])

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
        if token.i in to_coref and to_coref[token.i]:
            corefed_tokens[token.i] = token._.coref_clusters[0].main.text
        else:
            corefed_tokens[token.i] = None

    # Step 3. Go through each sentence in the doc, tag relevant parts as Q, V, S, or N
    labeled_sents = defaultdict(dict) # To fill with list of doc.sents, with tagged versions of tokens
    for sent_no,sent in enumerate(doc.sents):
        labeled_sents[sent_no]["idx2text"] = {tok.i: tok.text for tok in sent}
        labeled_sents[sent_no]["idx2lemma"] = {tok.i: tok.lemma_ for tok in sent}
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

            NEG,subj_NEG,IS_NEG,neg_children,subj_NEG_children = None,None,None,None,None
            SUBJECT,subj_children = None,None

            for child in ROOT.children:
                if child.dep_[:5] == 'nsubj' or child.dep_ == 'expl':
                    SUBJECT = child
                    if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                        SUBJECT = SUBJECT.head.head

                if child.dep_[:3] == 'neg':
                    NEG = child
                    neg_children = [c for c in NEG.children if c != ROOT]
                    for x in neg_children:
                        new_children = [c for c in x.children]
                        neg_children.extend(new_children)
                    IS_NEG = ROOT in NEG.head.children or ROOT == NEG.head

            if (SUBJECT is None) and ROOT.dep_[-2:] == 'cl':
                SUBJECT = ROOT.head if ROOT.head.pos_ == 'NOUN' or ROOT.head.pos_ == 'PROPN' else None

            # Get rest of subject tokens
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

            # Find embedded comp. clause
            emb_main_verbs = [c for c in VERB.children if c.dep_ == 'ccomp']

            for emb_main_verb in emb_main_verbs:
                # Recursively get all children of main verb of embedded clause
                children_queue = [x for x in emb_main_verb.children]
                for x in children_queue:
                    new_children = [c for c in x.children]
                    children_queue.extend(new_children)

                # Group indices by Quote component
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

    return labeled_sents,corefed_tokens



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--debug', action="store_true", default=None,
                      help='whether to test run on smaller sample first')
    arg_parser.add_argument('--path_to_df', type=str, default=None,
                      help='/path/to/df')
    arg_parser.add_argument('--output_dir', type=str, default=None,
                      help='where to write batched output')
    arg_parser.add_argument('--fulltext_dir', type=str, default=None,
                      help='where to source fulltext')

    args = arg_parser.parse_args()

    df = pd.read_pickle(os.path.join(REMOTE_SCRAPE_DIR,args.path_to_df))#,sep='\t',header=0,index_col=0)
    print('Length of df:',len(df))

    if args.debug:
        end_ix = 5
    else:
        end_ix = len(df)


    batch_no = 0
    if not os.path.exists(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no))):
        os.mkdir(os.path.join(REMOTE_PREPRO_DIR,args.output_dir,'extracted_quotes_{}'.format(batch_no)))


    start_time = time.time()
    for ix in range(end_ix):
        row_ix = df.index[ix]
        row = df.loc[row_ix]
        guid = row['guid']
        #print(ix,row_ix,guid)
        text = get_fulltext(guid,args.fulltext_dir)
        save_name = '{}.json'.format(guid)
        if len(text) > 0:
            labeled_sents,corefed_tokens = spacy_pipe(text)
            #print('corefed_tokens type:',type(corefed_tokens))
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
