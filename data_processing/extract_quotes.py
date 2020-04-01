import os
import pandas as pd
import numpy as np
import json
import time
from nltk.tokenize import sent_tokenize

fulltext_dir='/juice/u/yiweil/sci-debates/scraping/fulltexts/'
fnames = set(os.listdir(fulltext_dir))
print(len(fnames))

def fulltext_exists(url,fulltext_dir=fulltext_dir):
    fname = url.replace('/','[SEP]')
    return fname+'.txt' in fnames or fname[:90]+'.txt' in fnames

def get_fname(url,fulltext_dir=fulltext_dir):
    fname = url.replace('/','[SEP]')
    if fname+'.txt' in fnames:
        return fname
    else:
        return fname[:90]

def get_fulltext(url,fulltext_dir=fulltext_dir):
    fname = url.replace('/','[SEP]')
    if fname+'.txt' in fnames or fname[:90]+'.txt' in fnames:
        try:
            with open(fulltext_dir+fname+'.txt','r') as f:
                lines = f.readlines()
        except OSError:
            with open(fulltext_dir+fname[:90]+'.txt','r') as f:
                lines = f.readlines()

        return lines
    return ""

import spacy
nlp = spacy.load('en_core_web_sm')
import neuralcoref
neuralcoref.add_to_pipe(nlp)

def is_good_verb_dep(dep):
    return dep[:3] == 'aux' or dep[:3] == 'adv' or dep == 'det' or dep == 'rel' or dep == 'prep' or dep[-3:] == 'obj' or dep[-3:] == 'mod' or dep == 'prt' or (dep[-4:] == 'comp' and dep != 'ccomp')

def is_good_subj_dep(dep):
    return dep != 'ccomp'

def is_ROOT(tok):
    return tok.dep_ == 'ROOT' or tok.dep_[-2:] == 'cl' or tok.dep_ == 'ccomp' or \
            (tok.dep_ == 'conj' and tok.head.dep_ == 'ROOT')

REL_PRONOUNS = set(['who', 'whom', 'whose', 'which', 'that'])
def is_rel_pronoun(tok):
    tok = tok.lower().strip()
    return tok in REL_PRONOUNS

def get_quotes(text):
    # Do coref resolution
    doc = nlp(text)
    text = doc._.coref_resolved

    quote_objs = []

    for sent in sent_tokenize(text):#d.sents:
        #print(sent)
        sent = nlp(sent)
        # Go through entire sentence, looking for verbs embedding a complement clause
        VERBS = np.unique([token.head for token in sent if token.dep_ == 'ccomp'])
        #print(VERBS)

        # Extract everything else for each VERB
        for VERB in VERBS:
            #print("\nCcomp dependency found! For quoting verb '{}'".format(VERB))
            # Extract the rest of the quoting verb
            verb_deps = [x for x in VERB.children if is_good_verb_dep(x.dep_)]
            #print("\tFound children:",verb_deps)
            for x in verb_deps:
                new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                #print("\tAdding children of {}:".format(x.text),new_children)
                verb_deps.extend(new_children)
                #print("\tUpdated verb deps:",verb_deps)

            # If verb's head is not itself (i.e., verb is not the ROOT),
            # recursively trace back to ROOT, then add all children of ROOT
            ROOT = VERB
            while not is_ROOT(ROOT):
                ROOT = ROOT.head
                #print('\t\tCurrent root:',ROOT)
            if VERB is not ROOT:
                verb_deps.append(ROOT)

            #print("\tAdding children of ROOT...")
            root_deps = [x for x in ROOT.children if is_good_verb_dep(x.dep_)]
            #print("\tFound ROOT deps:",root_deps)
            for x in root_deps:
                new_children = [c for c in x.children if is_good_verb_dep(c.dep_)]
                #print("\tAdding children of {}:".format(x.text),new_children)
                root_deps.extend(new_children)
                #print("\tUpdated ROOT deps:",root_deps)

            #print("\tAdding ROOT deps to verb deps...")
            verb_deps.extend([x for x in root_deps if x != VERB and x not in verb_deps])
            #print("\tUpdated verb deps:",verb_deps)

            NEG,IS_NEG,neg_children = None,None,None
            SUBJECT,subj_children = None,None

            #print("\nLooking for SUBJECT and NEGATION(s)...")
            for child in ROOT.children:
                if child.dep_[:5] == 'nsubj' or child.dep_ == 'expl':
                    SUBJECT = child
                    #print("\tFound SUBJECT:",SUBJECT)
                    if SUBJECT.head.dep_ == 'relcl' and is_rel_pronoun(SUBJECT.text): # we're dealing with the subject of a rel clause
                        #print("\tFound quote inside a relative clause. Finding antecedent subject...")
                        SUBJECT = SUBJECT.head.head
                        #print("\tTrue subject:",SUBJECT)
                    #print("Subject token '{}' is in a coref cluster:".format(SUBJECT),SUBJECT._.in_coref)

                if child.dep_[:3] == 'neg':
                    NEG = child
                    verb_deps.append(NEG)
                    neg_children = [c for c in NEG.children if c != VERB]
                    #print("n\tAdding new NEG children:",neg_children)
                    for x in neg_children:
                        new_children = [c for c in x.children]
                        #print("\tNew NEG grandchildren:",new_children)
                        neg_children.extend(new_children)
                        #print("\tUpdated neg_children:",neg_children)
                    verb_deps.extend(neg_children)
                    #print("\tUpdated verb_deps:",verb_deps)
                    IS_NEG = VERB in NEG.head.children or VERB == NEG.head

            if SUBJECT is None and (ROOT.dep_ == 'acl' or ROOT.dep_ == 'advcl'):
                main_verb = ROOT.head
                #print([(c.text,c.dep_) for c in main_verb.children])
                SUBJS = [c for c in main_verb.children if c.dep_[:5] == 'nsubj' or c.dep_ == 'expl']
                SUBJECT = SUBJS[0] if len(SUBJS) > 0 else None

            # Get rest of subject tokens
            if SUBJECT is not None:
                #print("\nFound SUBJECT:",SUBJECT)
                #print("\tAdding children of SUBJECT...")
                subj_children = [c for c in SUBJECT.children if is_good_subj_dep(c.dep_)]
                #print("\tFound children:",subj_children)
                for x in subj_children:
                    new_children = [c for c in x.children if is_good_subj_dep(c.dep_)]
                    #print("\tAdding children of child {}:".format(x.text),new_children)
                    subj_children.extend(new_children)
                    #print("\tUpdated subject children:",subj_children)

            sorted_verb_tokens = sorted([(c,c.i) for c in verb_deps+[VERB]],key=lambda x:x[1])
            #print("\n\tSorted verb tokens:",sorted_verb_tokens)
            if SUBJECT is not None:
                sorted_subj_tokens = sorted([(c,c.i) for c in subj_children+[SUBJECT]],key=lambda x:x[1])
                #print("\tSorted subject tokens:",sorted_subj_tokens)
            else:
                sorted_subj_tokens = None
            if NEG is not None:
                sorted_neg_tokens = sorted([(c,c.i) for c in neg_children+[NEG]],key=lambda x:x[1])
            else:
                sorted_neg_tokens = None

            #print("\nFinding quote introduced by '{}'...".format(VERB))
            emb_main_verbs = [c for c in VERB.children if c.dep_ == 'ccomp']
            #print("\tMain verbs of embedded clause:",emb_main_verbs)
            #assert len(emb_main_verbs) <= 2
            for emb_main_verb in emb_main_verbs:
                #print("\tMain *verb*:",emb_main_verb)

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
                #print("\nSorting tokens in quote...")
                children_and_indices = [(c,c.i) for c in children_queue+[emb_main_verb]]
                sorted_ = sorted(children_and_indices,key=lambda x:x[1])
                quote_objs.append({'quote':[tup[0].text for tup in sorted_],
                        'verb tokens':[tup[0].text for tup in sorted_verb_tokens],
                        'main verb':VERB.text,
                       'subject tokens':[tup[0].text for tup in sorted_subj_tokens] if sorted_subj_tokens is not None else None,
                       'main subject':SUBJECT.text if SUBJECT is not None else None,
                       'neg tokens':[tup[0].text for tup in sorted_neg_tokens] if sorted_neg_tokens is not None else None,
                       'main neg':NEG.text if NEG is not None else None,
                       'is neg':IS_NEG})

    return quote_objs



if __name__ == "__main__":
    df = pd.read_pickle('../scraping/dedup_combined_df.pkl')
    df.head(10)

    start_time = time.time()
    for url_ix in range(7433,len(df)):
        curr_url = df.url.values[url_ix]
        quotes = get_quotes(get_fulltext(curr_url)[0])
        #fname = get_fname(curr_url)
        with open('./extracted_quotes_2/url_no_{}.jsonlist'.format(url_ix),'w+') as f:
            for res in quotes:
                json.dump(res, f)
                f.write('\n')
        if url_ix % 50 == 0:
            print(url_ix,curr_url)
    print('{}\tElapsed time in minutes:'.format(fname),(time.time()-start_time)/60.)
