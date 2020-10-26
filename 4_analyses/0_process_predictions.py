#!/usr/bin/env python
import glob
import os
import pandas as pd
import json
import nltk
from nltk.tokenize import MWETokenizer

from utils import int2str_label, get_rel_stance_label, get_pronouns, get_fulltext, read_quote_json, mv_files

PRONOUNS = get_pronouns()

# Create MWE Tokenizer for special words
tk = MWETokenizer()
tk.add_mwe(('peer', 'review'))
tk.add_mwe(('non', 'peer', 'review'))
tk.add_mwe(('climate', 'scientist'))
tk.add_mwe(('climate', 'scientist'))
tk.add_mwe(('medium', 'outlet'))
tk.add_mwe(('nobel', 'prize', 'win'))
tk.add_mwe(('nobel', 'win'))
tk.add_mwe(('nobel', 'peace', 'prize'))
tk.add_mwe(('nobel', 'laureate'))
tk.add_mwe(('nobel', 'laureates'))
tk.add_mwe(('prize', 'win'))
tk.add_mwe(('ocasio', 'cortez'))

def get_sorted_indices(q_dict):
    """Helper function to `get_context()`.
    Sorts indices of every Quote component ((main) subj., (main) verb, (main) neg.) within a Quote obj."""
    context_dict_ = {}
    for context_type in q_dict:
        context_dict_[context_type] = sorted([int(x) for x in q_dict[context_type]])
    return context_dict_

def get_inorder_text(context_indices,idx2text,idx2lemma,idx2coref):
    """Helper function to `get_context()`."""
    text_dict_ = {}
    for context_type in context_indices:
        text_dict_[context_type] = [idx2coref[str(x)] if idx2text[str(x)].lower() in PRONOUNS
                                    and idx2coref[str(x)] is not None
                                    else idx2text[str(x)]
                                    for x in context_indices[context_type]]
        text_dict_[context_type+'_lemmas'] = [idx2coref[str(x)] if idx2lemma[str(x)].lower() in PRONOUNS
                                    and idx2coref[str(x)] is not None
                                    else idx2lemma[str(x)]
                                    for x in context_indices[context_type]]
    return text_dict_

def get_context(src_guid,src_sent_no,src_q_no,q_dir):
    j = read_quote_json(src_guid,q_dir)
    q_dict = j['quote_tags'][src_sent_no]['quotes'][src_q_no]
    context_indices = get_sorted_indices(q_dict)
    idx2text = j['quote_tags'][src_sent_no]['idx2text']
    idx2lemma = j['quote_tags'][src_sent_no]['idx2lemma']
    idx2coref = j['coref_tags']
    text_dict = get_inorder_text(context_indices,idx2text,idx2lemma,idx2coref)
    return text_dict

def get_verb_tense(verb_text):
    return nltk.pos_tag([verb_text])[0][1]

def get_verb_mods(verb_list):
    """Given a list of verbs ('v_lemmas' in a Quote obj), groups into other verbs, adverbs, modals."""
    out = nltk.pos_tag(' '.join(verb_list).split())
    return {pos: [x[0] for x in out if x[1][:2] == pos] for pos in ['RB','MD','VB']}

with open('implicatives.txt','r') as f:
    IMPLICATIVES = set(f.read().splitlines())
print('Using implicatives set:',IMPLICATIVES)

def has_neg(context_dict):
    """Determines whether a comp. clause has been negated in its subject or verb, with a negative or implicative."""
    has_subj_neg = len(context_dict['main_neg_s']) > 0
    has_verb_neg = len(context_dict['main_neg_v']) > 0
    has_verb_impl = set(context_dict['v_lemmas']).intersection(IMPLICATIVES)
    has_verb_NEG = has_verb_neg or has_verb_impl
    if has_subj_neg and has_verb_NEG:
        return 'subj_verb_neg'
    elif has_subj_neg:
        return 'subj_neg'
    elif has_verb_neg:
        return 'verb_neg'
    elif has_verb_impl:
        return 'verb_impl'
    else:
        return 'no_neg'

if __name__=="__main__":

    articles_df = pd.read_pickle('../1_data_scraping/output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl')
    src_guid2attrs = dict()
    for attr in ['stance','domain','date','is_AP']:
        src_guid2attrs[attr] = dict(zip(articles_df['guid'],articles_df[attr]))

    def get_src_attrs(src_guid):
        """Function for getting attributes of source article given article guid"""
        return {attr: src_guid2attrs[attr][src_guid] for attr in src_guid2attrs}

    orig = pd.read_csv('../2_data_processing/output/keyword_filtered_comp_clauses.tsv',
                      sep='\t',header=0,index_col=0)
    orig.reset_index(drop=True,inplace=True)
    print(orig.columns)

    print("Reading in batched BERT predictions...")
    PRED_DIR = "../3_cc_stance/2_Stance_model/model_preds"
    pred_files = glob.glob(os.path.join(PRED_DIR,'batch_*_pred.tsv'))
    print('\tFound {} prediction files.'.format(len(pred_files)))
    all_preds = [pd.read_csv(pred_file,sep='\t',header=0) for pred_file in pred_files]
    all_preds = pd.concat([x for x in all_preds],ignore_index=True)
    print('\tRead in {} predictions.'.format(len(all_preds)))

    print("Processing predictions...")
    all_preds['quote_text'] = orig['quote_text'].copy()
    all_preds['src_guid'] = orig['guid'].copy()
    all_preds['src_sent_no'] = orig['sent_no'].copy().apply(lambda x: str(int(x)))
    all_preds['src_quote_no'] = orig['quote_no'].copy().apply(lambda x: str(int(x)))
    all_preds['quote_guid'] = all_preds['src_guid'].apply(lambda x: x+"_") + \
                              all_preds['src_sent_no'].apply(lambda x: str(int(x))+"_") + \
                              all_preds['src_quote_no']
    all_preds['predicted_label'] = all_preds['predicted'].apply(int2str_label)
    all_preds['max_prob'] = all_preds[['0', '1','2']].max(axis=1)

    print("Creating main analysis dataframe...")
    src_guid_,src_sent_no_,src_q_no_,med_stance_,pub_date_,domain_,is_AP_,\
    quote_,cc_abs_stance_,cc_rel_stance_,neg_type_,subjs_,main_subj_lemma_,\
    verb_lemma_,verb_tense_,verb_advs_,verb_mds_,verb_other_ = [],[],[],[],[],[],[],[],[],\
                                                               [],[],[],[],[],[],[],[],[]
    for pred_item_ix,row in all_preds.iterrows():
        quote_.append(row['quote_text'])
        pred_label = row['predicted']
        src_guid = row['src_guid']
        src_sent_no = str(row['src_sent_no'])
        src_q_no = row['src_quote_no']

        src_media_attrs = get_src_attrs(src_guid)
        stance,domain,pub_date,is_AP = src_media_attrs['stance'],src_media_attrs['domain'],\
                                       src_media_attrs['date'],src_media_attrs['is_AP']
        rel_label = get_rel_stance_label(pred_label,stance)

        med_stance_.append(stance)
        domain_.append(domain)
        pub_date_.append(pub_date)
        is_AP_.append(is_AP)
        cc_abs_stance_.append(pred_label)
        cc_rel_stance_.append(rel_label)
        src_guid_.append(src_guid)
        src_sent_no_.append(src_sent_no)
        src_q_no_.append(src_q_no)

        context = get_context(src_guid,src_sent_no,src_q_no,
                              '../2_data_processing/url_quotes/')
        main_v_text = context['main_v'][0]
        verb_tense_.append(get_verb_tense(main_v_text))

        main_v_lemma = context['main_v_lemmas'][0]
        all_v = context['v_lemmas']
        other_v = all_v.copy()
        other_v.remove(main_v_lemma)
        verb_mods = get_verb_mods(other_v)
        verb_advs_.append(json.dumps(verb_mods['RB']))
        verb_mds_.append(json.dumps(verb_mods['MD']))
        verb_other_.append(json.dumps(verb_mods['VB']))

        main_s_lemma = context['main_s_lemmas']
        if len(main_s_lemma) > 0:
            main_subj_lemma_.append(main_s_lemma[0])
        else:
            main_subj_lemma_.append(None)
        subjs_.append(json.dumps(context['s_lemmas']))

        # Append particles
        if len(context['v_prt']) > 0:
            good_v_prts = list(set(context['v_prt']).intersection({'out'}))
            if len(good_v_prts) > 0:
                main_v_lemma += '_'+good_v_prts[0]

        verb_lemma_.append(main_v_lemma)

        neg_type = has_neg(context)
        neg_type_.append(neg_type)

        if pred_item_ix % 1000 == 0:
            print('Processed row:',pred_item_ix)

    main_df = pd.DataFrame({
        "guid":src_guid_,"sent_no":src_sent_no_,"quote_no":src_q_no_,
        "outlet_stance":med_stance_,"domain":domain_,"is_AP":is_AP_,"pub_date":pub_date_,
        "neg_type":neg_type_,"main_s_lemma":main_subj_lemma_,"s_lemmas":subjs_,
        "main_v_lemma":verb_lemma_,"tense":verb_tense_,"advs":verb_advs_,"modals":verb_mds_,"other_verbs":verb_other_,
        "quote":quote_,"rel_quote_stance":cc_rel_stance_,"abs_quote_stance":cc_abs_stance_
    })
    main_df['joined_s_lemmas'] = main_df['s_lemmas'].apply(lambda x: ' '.join(json.loads(x)).strip().replace(' - ',' ').lower())
    main_df['mwe_tok_s_lemmas'] = main_df['joined_s_lemmas'].apply(lambda x: json.dumps(tk.tokenize(x.split())))

    output_dir = 'test_output_dir'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    main_df.to_pickle(os.path.join(output_dir,'quote_analysis_df.pkl'))
    print("Saving to {}...".format(os.path.join(output_dir,'quote_analysis_df.pkl')))
    print("Done!")
