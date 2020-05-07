import os
import pandas as pd
import numpy as np
import json
import time
from nltk.tokenize import sent_tokenize
import pickle
import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

config = json.load(open('../config.json', 'r'))
REMOTE_PREPRO_DIR = config['REMOTE_PROPRO_DIR']
REMOTE_SCRAPE_DIR = config['REMOTE_SCRAPE_DIR']
fulltext_dir = os.path.join(SCRAPE_DIR,'url_texts')

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
    """Runs full preprocessing and extraction pipeline on raw text."""
    # Step 0. Run pipeline.
    doc = nlp(text)

    # Step 1. Figure out which tokens to tag w/ coreferring token
    to_annotate = {}
    for clust in doc._.coref_clusters:
        for mention in clust.mentions:
            start = mention.start
            end = mention.end
            if start+1==end:
                to_annotate[start] = True
            else:
                to_annotate[start] = True
                for ix in range(start+1,end):
                    to_annotate[ix] = False

    # Step 2. Annotate those tokens with coreferring tokens
    annotated_tokens = {}
    for token in doc:
        #print(token.i,token.text,token.head,token.dep_)
        if token.i in to_annotate and to_annotate[token.i]:
            #print('Main span:',token._.coref_clusters[0].main)
            #print('Main span start token_ix, end token_ix:',token._.coref_clusters[0].main.start,
            #     token._.coref_clusters[0].main.end)
            annotated_tokens[token.i] = token._.coref_clusters[0].main

        else:
            annotated_tokens[token.i] = None

    # Step 3. Go through each sentence in the doc, extract up to 1 quote from each sentence
    quote_objs = []
    for sent in doc.sents:
        print(sent)

        VERBS = list(np.unique([token.head for token in sent if token.dep_ == 'ccomp']))
        VERBS = [v for v in VERBS if v.lemma_ in householder_stems]
        print('\nHouseholder stems present:',VERBS)

        # Extract everything else for each VERB
        for VERB in VERBS:
            print("\nCcomp dependency found! For quoting verb '{}'".format(VERB))
            # Extract the rest of the quoting verb
            verb_deps = [x for x in VERB.children if is_good_verb_dep(x.dep_)]
            verb_prts = [x for x in VERB.children if is_verb_prt(x.dep_)]

            print('\tFound verb particle(s):',verb_prts)
            for x in verb_prts:
                new_children = [c for c in x.children]
                print("\tAdding children of {}:".format(x.text),new_children)
                verb_prts.extend(new_children)
                print("\tUpdated verb particles:",verb_prts)

            print("\ntFound children of VERB:",verb_deps)
            for x in verb_deps:
                new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                print("\tAdding children of {}:".format(x.text),new_children)
                verb_deps.extend(new_children)
                print("\tUpdated verb deps:",verb_deps)

            # If verb's head is not itself (i.e., verb is not the ROOT),
            # recursively trace back to ROOT, then add all children of ROOT
            ROOT = VERB
            if is_ROOT(ROOT):
                if ROOT.dep_ == 'relcl' and ROOT.head.lemma_ in householder_verbs:
                    #print('CL DEP:',ROOT.dep_)
                    print("\n\tHead of '{}' is the true ROOT:".format(ROOT),ROOT.head)
                    ROOT = ROOT.head
                    verb_deps.append(ROOT)
                else:
                    #pass
                    print("\n\tVerb '{}' is the ROOT.".format(ROOT))
                #verb_deps.append(ROOT)
            else:
                print('\n\tFinding ROOT...')
                while not is_ROOT(ROOT):
                    ROOT = ROOT.head
                    print('\t\tCurrent root:',ROOT)
                assert is_ROOT(ROOT)
                if ROOT.dep_ == 'relcl' and ROOT.head.lemma_ in householder_verbs:
                    ROOT = ROOT.head
                print('\n\tROOT found!:',ROOT)
                verb_deps.append(ROOT)

                print("\tAdding children of ROOT...")
                root_deps = [x for x in ROOT.children if is_good_verb_dep(x.dep_)]
                print("\tFound ROOT deps:",root_deps)
                for x in root_deps:
                    new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                    print("\tAdding children of {}:".format(x.text),new_children)
                    root_deps.extend(new_children)
                    print("\tUpdated ROOT deps:",root_deps)

                print("\tAdding ROOT deps to verb deps...")
                verb_deps.extend([x for x in root_deps if x != VERB and x not in verb_deps])
                print("\tUpdated verb deps:",verb_deps)

            NEG,IS_NEG,neg_children = None,None,None
            SUBJECT,subj_children = None,None

            print("\nLooking for SUBJECT and NEGATION(s) among ROOT's children:",[c.text for c in ROOT.children])
            for child in ROOT.children:
                if child.dep_[:5] == 'nsubj' or child.dep_ == 'expl':
                    SUBJECT = child
                    print("\tFound SUBJECT:",SUBJECT)
                    if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                        print("\tFound quote inside a relative clause. Finding antecedent subject...")
                        SUBJECT = SUBJECT.head.head
                        print("\tTrue subject:",SUBJECT)
                    #print("Subject token '{}' is in a coref cluster:".format(SUBJECT),SUBJECT._.in_coref)

                if child.dep_[:3] == 'neg':
                    NEG = child
                    print('\tFound NEG:',NEG)
                    verb_deps.append(NEG)
                    neg_children = [c for c in NEG.children if c != ROOT]
                    print("n\tAdding new NEG children:",neg_children)
                    for x in neg_children:
                        new_children = [c for c in x.children]
                        print("\tNew NEG grandchildren:",new_children)
                        neg_children.extend(new_children)
                        print("\tUpdated neg_children:",neg_children)
                    verb_deps.extend(neg_children)
                    print("\tUpdated verb_deps:",verb_deps)
                    IS_NEG = ROOT in NEG.head.children or ROOT == NEG.head

            print('\nROOT dep label:',ROOT.dep_)
            if (SUBJECT is None) and ROOT.dep_[-2:] == 'cl':
                print('\nRoot is a clausal modifier, modify-ee may be subject...')
                SUBJECT = ROOT.head if ROOT.head.pos_ == 'NOUN' or ROOT.head.pos_ == 'PROPN' else None
                print('\n\tFound SUBJECT:',SUBJECT)
#                 SUBJS = [c for c in ROOT.children if c.dep_[:5] == 'nsubj' or c.dep_ == 'expl']
#                 SUBJECT = SUBJS[0] if len(SUBJS) > 0 else None

            # Get rest of subject tokens
            if SUBJECT is not None:
                print("\nFound SUBJECT:",SUBJECT)
                print("\tAdding children of SUBJECT...")
                subj_children = [c for c in SUBJECT.children if is_good_subj_dep(c.dep_)]
                print("\tFound children:",subj_children)
                for x in subj_children:
                    new_children = [c for c in x.children if is_good_subj_dep(c.dep_)]
                    print("\tAdding children of child {}:".format(x.text),new_children)
                    subj_children.extend(new_children)
                    print("\tUpdated subject children:",subj_children)

            print("\nFinding quote introduced by '{}'...".format(VERB))
            emb_main_verbs = [c for c in VERB.children if c.dep_ == 'ccomp']
            # ^what if change to ROOT.children??????

            print("\tMain verbs of embedded clause:",emb_main_verbs)
            #assert len(emb_main_verbs) <= 2
            for emb_main_verb in emb_main_verbs:
                print("\tMain *verb*:",emb_main_verb)

                # Recursively get all children of main verb of embedded clause
                #print("\nRecursively getting children of main verb of embedded clause...\n")
                #print("*"*50)
                children_queue = [x for x in emb_main_verb.children]
                #print("\tAdding children of main verb:",children_queue)
                for x in children_queue:
                    new_children = [c for c in x.children]
                    #print("\tAdding children of {}:".format(x.text),new_children)
                    children_queue.extend(new_children)
                    #print("\tNew children queue:",children_queue)

                # Sort children and matrix verb to be in correct order
                # Replace w/ coreferring text as necessary
                children_and_indices_and_coref = [(c,c.i,annotated_tokens[c.i]) for c in children_queue+[emb_main_verb]]
                sorted_ = sorted(children_and_indices_and_coref,key=lambda x:x[1])
                sorted_verb_tokens = sorted([(c,c.i,annotated_tokens[c.i]) for c in verb_deps+[VERB]],key=lambda x:x[1])
                #print("\n\tSorted verb tokens:",sorted_verb_tokens)
                sorted_verb_particles = sorted([(c,c.i,annotated_tokens[c.i]) for c in verb_prts+[VERB]],
                                              key=lambda x:x[1])
                if SUBJECT is not None:
                    sorted_subj_tokens = sorted([(c,c.i,annotated_tokens[c.i]) for c in subj_children+[SUBJECT]],key=lambda x:x[1])
                    #print("\tSorted subject tokens:",sorted_subj_tokens)
                else:
                    sorted_subj_tokens = None
                if NEG is not None:
                    sorted_neg_tokens = sorted([(c,c.i,annotated_tokens[c.i]) for c in neg_children+[NEG]],key=lambda x:x[1])
                else:
                    sorted_neg_tokens = None

                dict_ = {'quote tokens':[tup[0] for tup in sorted_],
                        'quote tokens coref':[tup[0] if tup[-1] is None else tup[-1] for tup in sorted_],
                        'verb tokens':[tup[0] for tup in sorted_verb_tokens],
                        'verb tokens coref':[tup[0] if tup[-1] is None else tup[-1] for tup in sorted_verb_tokens],
                         'verb prts':[tup[0] for tup in sorted_verb_particles],
                         'verb prts coref':[tup[0] if tup[-1] is None else tup[-1] for tup in sorted_verb_particles],
                        'main verb':ROOT,
                        'main verb coref':annotated_tokens[ROOT.i] if ROOT.i in annotated_tokens and
                                   annotated_tokens[ROOT.i] else ROOT,
                       'subject tokens':[tup[0] for tup in sorted_subj_tokens] if sorted_subj_tokens is not None else None,
                        'subject tokens coref':([tup[0] if tup[-1] is None else tup[-1] for tup in sorted_subj_tokens])
                                   if sorted_subj_tokens is not None else None,
                       'main subject':SUBJECT if SUBJECT is not None else None,
                        'main subject coref':annotated_tokens[SUBJECT.i] if SUBJECT is not None and SUBJECT.i in annotated_tokens and
                                   annotated_tokens[SUBJECT.i] else
                                   (SUBJECT if SUBJECT is not None else None),
                       'neg tokens':[tup[0] for tup in sorted_neg_tokens] if sorted_neg_tokens is not None else None,
                       'main neg':NEG if NEG is not None else None,
                       'is neg':IS_NEG}

                res_ = {}
                res_['quote lemmas'] = [x.lemma_ for x in dict_['quote tokens coref']]
                res_['verb lemmas'] = [x.lemma_ for x in dict_['verb tokens coref']]
                res_['verb prt lemmas'] = [x.lemma_ for x in dict_['verb prts coref']]
                res_['main verb lemma'] = dict_['main verb coref'].lemma_
                res_['subject lemmas'] = [x.lemma_ for x in dict_['subject tokens coref']] if dict_['subject tokens coref'] is not None else None
                res_['main subject lemma'] = dict_['main subject'].lemma_ if dict_['main subject'] is not None else None
                res_['main subject lemma coref'] = dict_['main subject coref'].lemma_ if dict_['main subject coref'] is not None else None
                res_['neg lemmas'] = [x.lemma_ for x in dict_['neg tokens']] if dict_['neg tokens'] is not None else None
                res_['main neg lemma'] = dict_['main neg'].lemma_ if dict_['main neg'] is not None else None
                res_['quote text'] = [x.text for x in dict_['quote tokens']]
                res_['verb text'] = [x.text for x in dict_['verb tokens']]
                res_['main verb text'] = dict_['main verb'].text
                res_['subject text'] = [x.text for x in dict_['subject tokens']] if dict_['subject tokens'] is not None else None
                res_['main subject text'] = dict_['main subject'].text if dict_['main subject'] is not None else None
                res_['is neg'] = dict_['is neg']

                quote_objs.append(res_)

    return quote_objs



if __name__ == "__main__":
    df = pd.read_pickle(REMOTE_SCRAPE_DIR+'/temp_combined_df_with_ft_date_title_dedup_df.pkl')
    print('Length of df:',len(df))

    batch_no = 0
    if not os.path.exists(REMOTE_PREPRO_DIR+'/extracted_quotes_{}'.format(batch_no)):
        os.mkdir(REMOTE_PREPRO_DIR+'/extracted_quotes_{}'.format(batch_no))

    for ix in range(len(df)):
        row = df.loc[ix]
        guid = row['guid']
        text = get_fulltext(guid)
        save_name = '{}.jl'.format(guid)
        if len(text) > 0:
            quotes = get_quotes(text)
        else:
            quotes = []
        with open(REMOTE_PREPRO_DIR+'/extracted_quotes_{}/{}'.format(batch_no,save_name),'w') as f:
            for res in quotes:
                json.dump(res, f)
                f.write('\n')

        if ix % 5000 == 0:
            print(ix,guid)
            print('Elapsed time in minutes:',(time.time()-start_time)/60.)

            # Divide into batches of 5000
            batch_no = ix
            if not os.path.exists(REMOTE_SCRAPE_DIR+'extracted_quotes_{}'.format(batch_no)):
                os.mkdir(REMOTE_SCRAPE_DIR+'extracted_quotes_{}'.format(batch_no))
