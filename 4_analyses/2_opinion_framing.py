#!/usr/bin/env python
import os
import argparse
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter,defaultdict
from scipy import stats
import statsmodels.api as sm
import statsmodels as sm
from scipy.stats import chisquare

import seaborn as sns
sns.set(font_scale=1.8)
sns.set_style("ticks")
import matplotlib.pyplot as plt

from utils import log_odds_ratio,log_odds

config = json.load(open('../config.json', 'r'))


def load_lexicons():
    """Returns a dict with (lexicon category, list of category framing devices) key, value pairs."""

    WORD_CATS = {}
    for word_cat in ['AFFIRM','DOUBT','VERBS']:
        with open('lexicons/{}_words.txt'.format(word_cat),'r') as f:
            WORD_CATS[word_cat] = f.read().splitlines()

    return WORD_CATS

def get_coverage_proportions(affirm_words,doubt_words,df_,df_desc,verbose=False):
    """
    Generates barplots showing self-affirmation vs. opponent-doubt coverage across RL and LL.

    :param affirm_words: set of strs with affirming devices to use in finding self-affirming discourse
    :param doubt_words: set of strs with doubting devices to use in finding opponent-doubting discourse
    :df_: dataframe with labeled opinions, outlet source, and framing context
    :param df_desc: str description of df_ for figure naming purposes
    """

    framed_df = df_.loc[df_['main_v_lemma'].isin(affirm_words | doubt_words)]
    self_affirm_df = framed_df.loc[(framed_df['main_v_lemma'].isin(affirm_words)) &
                                   (framed_df['rel_quote_stance']=='own')]
    opponent_doubt_df = framed_df.loc[(framed_df['main_v_lemma'].isin(doubt_words)) &
                                      (framed_df['rel_quote_stance']=='opposing')]

    N_LL = len(framed_df.loc[framed_df.outlet_stance=='pro'])
    N_RL = len(framed_df.loc[framed_df.outlet_stance=='anti'])

    if verbose:
        print(framed_df.shape,self_affirm_df.shape,opponent_doubt_df.shape)
        print("N_LL, N_RL:",N_LL,N_RL)

    coverage_proportion_df = pd.DataFrame(
    {
        "med_slant":['LL','LL','RL','RL'],
        "prop":[len(self_affirm_df.loc[self_affirm_df.outlet_stance=='pro'])/N_LL,
                len(opponent_doubt_df.loc[opponent_doubt_df.outlet_stance=='pro'])/N_LL,
                len(self_affirm_df.loc[self_affirm_df.outlet_stance=='anti'])/N_RL,
                len(opponent_doubt_df.loc[opponent_doubt_df.outlet_stance=='anti'])/N_RL],
        "coverage_type":['Self-Affirming','Opponent-Doubt','Self-Affirming','Opponent-Doubt']
    })

    fig,ax = plt.subplots(figsize=(10,5))
    sns.barplot(x="prop",y="med_slant",data=coverage_proportion_df,hue="coverage_type",orient='h',ax=ax)
    sns.despine()
    ax.set_xlabel('')
    ax.set_ylabel('')
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.21,0.1),fontsize=32)
    leg.get_frame().set_linewidth(3)
    ax.set_title('Proportions of coverage types across media',fontsize=33)
    plt.yticks(fontsize=36)
    plt.tight_layout()
    fig.savefig(os.path.join(curr_output_dir,'figs','coverage_asym_{}.pdf'.format(df_desc)))


def log_odds_w_s_M(w,w_type,s,s_type,M,df_,verbose=False):
    """
    Computes log odds of device 𝑤 framing an Opinion with stance 𝑠 (in media with slant 𝑀).
    Formula: log(c/n-c), where c is the count of (the event of) 𝑤 framing 𝑠 in 𝑀, and
    n is the total count of 𝑠 in 𝑀.

    :param w: the set of str lemmas of interest (a main verb, a set of adjs)
    :param w_type: the str fieldname of w, e.g. 'main_v_lemma', 'mwe_tok_s_lemmas'
    :param s: the str stance of interest, e.g. 'own', 'opposing'; 'pro', 'anti'
    :param s_type: the str fieldname of w, e.g. 'abs_quote_stance', 'rel_quote_stance'
    :param M: the str media leaning, e.g. 'pro', 'anti'
    :param df_: the dataframe to query
    """

    M_df = df_.loc[df_.outlet_stance==M]
    M_s_df = M_df.loc[M_df[s_type]==s]

    if w_type == 'main_v_lemma':
        M_s_w_df = M_s_df.loc[M_s_df[w_type].isin(w)]
    elif w_type == 'mwe_tok_s_lemmas':
        M_s_w_df = M_s_df.loc[M_s_df[w_type].apply(lambda x: len(x.intersection(w)) > 0)]
    else:
        M_s_w_df = None
        print("Unknown word type")

    c, n = len(M_s_w_df), len(M_s_df)
    if c == 0:
        c +=  0.05
        n += 0.05

    if verbose:
        print(w,c,n)

    return log_odds(c, n)

def log_odds_ratio_w_s_M(w,w_type,s_type,M,df_,verbose=False):
    """
    Computes the log odds ratio of device 𝑤 framing an Opinion with stance 𝑠1 vs. 𝑠2 (in media with slant 𝑀).
    Formula: log(c1/n1-c1) - log(c2/n2-c2), where c1 is the count of (the event of) 𝑤 framing 𝑠1 in 𝑀,
    n1 is the total count of 𝑠1 in 𝑀, c2 is the count of (the event of) 𝑤 framing 𝑠2 in 𝑀, and
    n2 is the total count of 𝑠2 in 𝑀.

    :param w: the set of str lemmas of interest (a main verb, a set of adjs)
    :param w_type: the str fieldname of w, e.g. 'main_v_lemma', 'mwe_tok_s_lemmas'
    :param s_type: the str fieldname of w, e.g. 'abs_quote_stance', 'rel_quote_stance'.
                   If s_type == 'abs_quote_stance', s1 is set to GW-agree and s2 set to GW-disagree.
                   If s_type == 'rel_quote_stance', s1 is set to own and s2 set to opposing.
    :param M: the str media leaning, e.g. 'pro', 'anti'
    :param df_: the dataframe to query
    """

    M_df = df_.loc[df_.outlet_stance==M]

    if s_type == 'abs_quote_stance':
        s1, s2 = 'pro', 'anti'
    else:
        s1, s2 = 'own', 'opposing'
    M_s1_df = M_df.loc[M_df[s_type]==s1]
    M_s2_df = M_df.loc[M_df[s_type]==s2]

    if w_type == 'main_v_lemma':
        M_s1_w_df = M_s1_df.loc[M_s1_df[w_type].isin(w)]
        M_s2_w_df = M_s2_df.loc[M_s2_df[w_type].isin(w)]
    elif w_type == 'mwe_tok_s_lemmas':
        M_s1_w_df = M_s1_df.loc[M_s1_df[w_type].apply(lambda x: len(x.intersection(w)) > 0)]
        M_s2_w_df = M_s2_df.loc[M_s2_df[w_type].apply(lambda x: len(x.intersection(w)) > 0)]
    else:
        M_s1_w_df, M_s2_w_df = None, None
        print("Unknown word type")

    c1, n1 = len(M_s1_w_df), len(M_s1_df)
    c2, n2 = len(M_s2_w_df), len(M_s2_df)
    if c1 == 0 or c2 == 0:
        c1 +=  0.05
        n1 += 0.05
        c2 +=  0.05
        n2 += 0.05

    if verbose:
        print(w,c1,n1,c2,n2)

    return log_odds_ratio(c1, n1, c2, n2)

def weighted_chisquare(w,w_type,s_type,M,df_):
    """
    Does a chi-squared test on whether observed frequencies of a device, w, with a given stance
    differ significantly from expected among media from side M.

    :param w: the str device of interest
    :param w_type: the str fieldname of w, e.g. 'main_v_lemma', 'mwe_tok_s_lemmas'
    :param s_type: the str fieldname of w, e.g. 'abs_quote_stance', 'rel_quote_stance'.
                   If s_type == 'abs_quote_stance', test is done for w occurring with GW-agree opinions
                   If s_type == 'rel_quote_stance', test is done for w occurring with own-side opinions
    :param M: the str media leaning, e.g. 'pro', 'anti'
    :param df_: the dataframe to query
    """

    M_df = df_.loc[df_.outlet_stance==M]

    # Find probability of encountering w
    if w_type == 'main_v_lemma':
        obs_w = M_df.loc[M_df[w_type]==w]
    elif w_type == 'mwe_tok_s_lemmas':
        obs_w = M_df.loc[M_df[w_type].apply(lambda x: w in x)]
    else:
        return
    P_w = len(obs_w)/len(M_df)

    # Compute expected freqs of w
    label_counts = M_df[s_type].value_counts()
    if s_type == 'abs_quote_stance':
        N_A, N_D = label_counts['pro'], label_counts['anti']
    else:
        N_A, N_D = label_counts['own'], label_counts['opposing']
    E_A, E_D = N_A*P_w, N_D*P_w
    exp = [E_A, E_D]

    # Get observed freqs of w
    obs_counts = defaultdict(int)
    obs_counts.update(obs_w[s_type].value_counts())
    if s_type == 'abs_quote_stance':
        if 'pro' not in obs_counts or 'anti' not in obs_counts:
            obs_counts['pro'] += 0.5
            obs_counts['anti'] += 0.5
        obs_A, obs_D = obs_counts['pro'],obs_counts['anti']
    if s_type == 'rel_quote_stance':
        if 'pro' not in obs_counts or 'anti' not in obs_counts:
            obs_counts['own'] += 0.5
            obs_counts['opposing'] += 0.5
        obs_A, obs_D = obs_counts['own'],obs_counts['opposing']
    obs = [obs_A,obs_D]

    return chisquare(obs,exp)

def get_high_freq_devices(device_type,affirm_words,doubt_words,verb_words,df_,verbose=False):
    """
    Returns the set of high frequency framing devices occurring at least 10 times in RL and LL.

    :param device_type: str equal to either "pred" or "mod"--if "pred", returns high frequency verbal predicates;
                        if "mod", returns high frequency source modifiers
    :param affirm_words: set of strs with affirming devices
    :param doubt_words: set of strs with doubting devices
    :param verb_words: set of strs with verbal devices
    :param df_: dataframe containing labeled opinions with framing context and source outlet stance
    """
    if device_type == 'pred':
        device = 'main_v_lemma'
        LL_counts = df_.loc[df_.outlet_stance=='pro'][device].value_counts()
        RL_counts = df_.loc[df_.outlet_stance=='anti'][device].value_counts()
        if verbose:
            print('Vocab size of {}s in LL, RL:'.format(device_type),len(LL_counts),len(RL_counts))

        high_freq_devices = {}
        for cat in ['affirm','doubt']:
            cat_verbs = affirm_words.intersection(verb_words) if cat == 'affirm' else \
                        doubt_words.intersection(verb_words)
            high_freq_cat_verbs = [v for v in cat_verbs if v in LL_counts and
                              v in RL_counts and RL_counts[v] >= 10 and
                             LL_counts[v] >= 10]
            high_freq_devices[cat] = high_freq_cat_verbs
            if verbose:
                print('Number of {} verbs, high freq. {} verbs:'.format(cat,cat),
                  len(cat_verbs),len(high_freq_cat_verbs))

    elif device_type == 'mod':
        device = 'mwe_tok_s_lemmas'

        # Flatten mwe_tok_s_lemmas to get counts
        LL_adjs = list(df_.loc[df_.outlet_stance=='pro'][device])
        LL_adjs = [item for sublist in LL_adjs for item in sublist]
        RL_adjs = list(df_.loc[df_.outlet_stance=='anti'][device])
        RL_adjs = [item for sublist in RL_adjs for item in sublist]
        LL_counts = Counter(LL_adjs)
        RL_counts = Counter(RL_adjs)
        if verbose:
            print('Vocab size of {}s in LL, RL:'.format(device_type),len(LL_counts),len(RL_counts))

        high_freq_devices = {}
        for cat in ['affirm','doubt']:
            cat_adjs = affirm_words.difference(verb_words) if cat == 'affirm' else \
                       doubt_words.difference(verb_words)
            high_freq_cat_adjs = [v for v in cat_adjs if v in LL_counts and
                              v in RL_counts and RL_counts[v] >= 1 and
                             LL_counts[v] >= 1]
            high_freq_devices[cat] = high_freq_cat_adjs
            if verbose:
                print('Number of {} modifiers, high freq. {} modifiers:'.format(cat,cat),
                  len(cat_adjs),len(high_freq_cat_adjs))
    else:
        return None

    return high_freq_devices

def compute_and_plot_device_biases(opinion_type,df_,df_desc,word_cats,fdr=0.1,alpha=0.05):
    """
    Computes each framing device's tendency to frame an opinion vs. the opposite across LL and RL.
    Creates aggregated boxplots and individual barplots.

    :param opinion_type: str equal to either 'abs_quote_stance' or 'rel_quote_stance'; if 'abs_quote_stance',
        bias is computed for framing a GW-agree vs. GW-disagree opinion; if 'rel_quote_stance', bias is computed
        for framing an own-side vs. opposing-side opinion
    :param df_: dataframe containing labeled opinions with framing context and source outlet stance
    :param df_desc: str label attached to dataframe for figure filename
    :param fdr: the false discovery rate to use for correcting p-values of multiple tests, default 0.1
    :param alpha: the parameter for rejecting the null hypothesis, by default 0.05
    """
    # First restrict to high frequency devices
    high_freq_verbs = get_high_freq_devices('pred',set(word_cats['AFFIRM']),set(word_cats['DOUBT']),
                                            set(word_cats['VERBS']),df_)
    high_freq_adjs = get_high_freq_devices('mod',set(word_cats['AFFIRM']),set(word_cats['DOUBT']),
                                            set(word_cats['VERBS']),df_)
    hf_affirm_verbs, hf_doubt_verbs = sorted(high_freq_verbs['affirm']), sorted(high_freq_verbs['doubt'])
    hf_affirm_adjs, hf_doubt_adjs = sorted(high_freq_adjs['affirm']), sorted(high_freq_adjs['doubt'])
    verbs_df_ = hf_affirm_verbs + hf_doubt_verbs
    adjs_df_ = hf_affirm_adjs + hf_doubt_adjs

    # Get counts for frequency-scaled plotting
    counts_per_verb = Counter(df_['main_v_lemma'])
    counts_per_mod = Counter([item for sublist in df_['mwe_tok_s_lemmas'] for item in sublist])

    log_odds_pro_cc_quote_verbs_df = pd.DataFrame(
    {
        "verb":verbs_df_*2, # each verb, once for each media slant
        "log odds":[log_odds_ratio_w_s_M({w},'main_v_lemma',opinion_type,'pro',df_)
                        for w in hf_affirm_verbs]+\
                   [log_odds_ratio_w_s_M({w},'main_v_lemma',opinion_type,'pro',df_)
                        for w in hf_doubt_verbs]+\
                   [log_odds_ratio_w_s_M({w},'main_v_lemma',opinion_type,'anti',df_)
                        for w in hf_affirm_verbs]+\
                   [log_odds_ratio_w_s_M({w},'main_v_lemma',opinion_type,'anti',df_)
                        for w in hf_doubt_verbs],
        "verb category":['Affirming']*len(hf_affirm_verbs)+['Doubting']*len(hf_doubt_verbs)+\
                        ['Affirming']*len(hf_affirm_verbs)+['Doubting']*len(hf_doubt_verbs),
        "Media slant":['LL']*(len(hf_affirm_verbs)+len(hf_doubt_verbs))+\
                      ['RL']*(len(hf_affirm_verbs)+len(hf_doubt_verbs)),
        "chisquare":[weighted_chisquare(w,'main_v_lemma',opinion_type,'pro',df_)
                        for w in hf_affirm_verbs]+\
                    [weighted_chisquare(w,'main_v_lemma',opinion_type,'pro',df_)
                        for w in hf_doubt_verbs]+\
                    [weighted_chisquare(w,'main_v_lemma',opinion_type,'anti',df_)
                        for w in hf_affirm_verbs]+\
                    [weighted_chisquare(w,'main_v_lemma',opinion_type,'anti',df_)
                        for w in hf_doubt_verbs],
        "count":[counts_per_verb[w] for w in hf_affirm_verbs]+\
                [counts_per_verb[w] for w in hf_doubt_verbs]+\
                [counts_per_verb[w] for w in hf_affirm_verbs]+\
                [counts_per_verb[w] for w in hf_doubt_verbs]
    })

    log_odds_pro_cc_quote_adjs_df = pd.DataFrame(
    {
        "mod":adjs_df_*2,
        "log odds":[log_odds_ratio_w_s_M({w},'mwe_tok_s_lemmas',opinion_type,'pro',df_)
                        for w in hf_affirm_adjs]+\
                   [log_odds_ratio_w_s_M({w},'mwe_tok_s_lemmas',opinion_type,'pro',df_)
                        for w in hf_doubt_adjs]+\
                   [log_odds_ratio_w_s_M({w},'mwe_tok_s_lemmas',opinion_type,'anti',df_)
                        for w in hf_affirm_adjs]+\
                   [log_odds_ratio_w_s_M({w},'mwe_tok_s_lemmas',opinion_type,'anti',df_)
                        for w in hf_doubt_adjs],
        "mod category":['Affirming']*len(hf_affirm_adjs)+['Doubting']*len(hf_doubt_adjs)+\
                       ['Affirming']*len(hf_affirm_adjs)+['Doubting']*len(hf_doubt_adjs),
        "Media slant":['LL']*(len(hf_affirm_adjs)+len(hf_doubt_adjs))+\
                      ['RL']*(len(hf_affirm_adjs)+len(hf_doubt_adjs)),
        "chisquare":[weighted_chisquare(w,'mwe_tok_s_lemmas',opinion_type,'pro',df_)
                         for w in hf_affirm_adjs]+\
                    [weighted_chisquare(w,'mwe_tok_s_lemmas',opinion_type,'pro',df_)
                         for w in hf_doubt_adjs]+\
                    [weighted_chisquare(w,'mwe_tok_s_lemmas',opinion_type,'anti',df_)
                         for w in hf_affirm_adjs]+\
                    [weighted_chisquare(w,'mwe_tok_s_lemmas',opinion_type,'anti',df_)
                         for w in hf_doubt_adjs],
        "count":[counts_per_mod[w] for w in hf_affirm_adjs]+\
                [counts_per_mod[w] for w in hf_doubt_adjs]+\
                [counts_per_mod[w] for w in hf_affirm_adjs]+\
                [counts_per_mod[w] for w in hf_doubt_adjs]
    })

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(8,8),sharey=False)
    swarm = sns.stripplot(x="verb category",y="log odds",data=log_odds_pro_cc_quote_verbs_df,hue='Media slant',
                          ax=ax1,palette={'LL':'b','RL':'r'},dodge=True,linewidth=1,
                         **{'s': np.log(np.sqrt(log_odds_pro_cc_quote_verbs_df['count']))*3})
    for t, l in zip(swarm.legend_.texts, ["",""]): t.set_text(l)
    sns.boxplot(x="verb category",y="log odds",data=log_odds_pro_cc_quote_verbs_df,hue='Media slant',ax=ax1,
               palette={'LL':'b','RL':'r'})
    ax1.set_xlabel("")
    ax1.set_ylabel("Log odds of ascribing a GW-agree opinion")
    ax1.get_legend().set_visible(False)

    sns.stripplot(x="mod category",y="log odds",data=log_odds_pro_cc_quote_adjs_df,hue='Media slant',ax=ax2,
                 palette={'LL':'b','RL':'r'},dodge=True,linewidth=1,
                 **{'s': np.log(np.sqrt(log_odds_pro_cc_quote_adjs_df['count']))*3})
    sns.boxplot(x="mod category",y="log odds",data=log_odds_pro_cc_quote_adjs_df,hue='Media slant',ax=ax2,
               palette={'LL':'b','RL':'r'})
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    handles,labels=ax2.get_legend_handles_labels()
    leg = ax2.legend(handles[:2],labels[:2],title='Media slant',loc='best', bbox_to_anchor=(0.4, 0.5, 0.3, 0.5),
                    edgecolor='slategray')
    leg.get_frame().set_linewidth(3)

    import string
    for n, ax in enumerate([ax1,ax2]):
        if n == 0:
            ax.text(-0.105, 1.03, "A)   Verbal Predicates", transform=ax.transAxes,
                size=20, weight='bold')
        else:
            ax.text(-0.105, 1.03, "B)   Source Modifiers", transform=ax.transAxes,
                size=20, weight='bold')

    fig.savefig(os.path.join(curr_output_dir,'figs','pred_mod_boxplots_{}.pdf'.format(df_desc)))

    for base_for_temp in ['verbs','adjs']:
        if base_for_temp == 'adjs':
            y_var = 'mod'
            temp_df = log_odds_pro_cc_quote_adjs_df.loc[log_odds_pro_cc_quote_adjs_df['mod'].isin(word_cats['AFFIRM'])].\
                        sort_values(['mod category','log odds'],ascending=False)
            temp_df = pd.concat([temp_df,log_odds_pro_cc_quote_adjs_df.loc[
                                     log_odds_pro_cc_quote_adjs_df['mod'].isin(WORD_CATS['DOUBT'])].\
                                 sort_values(['mod category','log odds'],ascending=False)])
        else:
            y_var = 'verb'
            temp_df = log_odds_pro_cc_quote_verbs_df.loc[log_odds_pro_cc_quote_verbs_df['verb'].isin(word_cats['AFFIRM'])].\
                        sort_values(['verb category','log odds'],ascending=False)
            temp_df = pd.concat([temp_df,log_odds_pro_cc_quote_verbs_df.loc[
                                     log_odds_pro_cc_quote_verbs_df['verb'].isin(WORD_CATS['DOUBT'])].\
                                 sort_values(['verb category','log odds'],ascending=False)])
        multitest_res = sm.stats.multitest.multipletests([x[1] for x in temp_df['chisquare']],
                                                        alpha=fdr,method='fdr_bh')
        temp_df['is_sig'], temp_df['corrected_pvals'] = multitest_res[0], multitest_res[1]
        pro_sigs = temp_df.loc[(temp_df['Media slant']=='LL') & (temp_df.is_sig)][y_var].values
        anti_sigs = temp_df.loc[(temp_df['Media slant']=='RL') & (temp_df.is_sig)][y_var].values
        double_sigs = set(pro_sigs).intersection(set(anti_sigs))
        single_sigs = [x for x in list(pro_sigs) + list(anti_sigs) if x not in double_sigs]

        LL_order = temp_df.loc[temp_df['Media slant']=='LL'].sort_values('log odds',ascending=False)[y_var]

        fig,ax = plt.subplots(figsize=(10,18))
        sns.barplot(y=y_var,x='log odds',data=temp_df,orient='h',
                    ax=ax,hue='Media slant',ci=95,palette={'LL':'b','RL':'r'},order=LL_order)
        ax.set_xlabel("")
        ylab = ax.set_ylabel('Log odds of ascribing a GW-agree opinion',fontsize=40,labelpad=35)
        ylab.set_position((0.01,0.5))
        leg = ax.legend(handles[:2],labels[:2],title='Media slant',loc='center left',bbox_to_anchor=(0.55,0.54),
                            edgecolor='gray',title_fontsize='large')
        leg.get_frame().set_linewidth(3)
        label_texts = [t.get_text() for t in ax.get_yticklabels()]
        label_texts = ['$\mathbf{'+x.replace('_','\ ')+'}$' if x.replace('*','') in word_cats['AFFIRM']
                       else '$\it{'+x+'}$' for x in label_texts]
        label_texts = [x+'**' if x.split('{')[1].split('}')[0] in double_sigs else x for x in label_texts]
        label_texts = [x+'*' if x.split('{')[1].split('}')[0] in single_sigs else x for x in label_texts]
        ax.set_yticklabels(labels=label_texts,rotation=90,fontsize=24)
        plt.yticks(rotation=360)
        plt.tight_layout()
        fig.savefig(os.path.join(curr_output_dir,'figs','all_{}_bars_{}.pdf'.format(y_var,df_desc)))

    return {'preds':log_odds_pro_cc_quote_verbs_df,
            'mods':log_odds_pro_cc_quote_adjs_df}


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--path_to_input', type=str,
        default=os.path.join('curr_output','quote_analysis_df.pkl'))
    arg_parser.add_argument('--do_subsampled', action="store_true", default=None,
        help="whether to perform suite of opinion-framing studies on subsampled dataset")
    arg_parser.add_argument('--do_nonwire_dataset', action="store_true", default=None,
        help="whether to perform suite of opinion-framing studies on dataset w/o wire articles")
    args = arg_parser.parse_args()

    curr_output_dir = 'curr_output'
    if not os.path.exists(curr_output_dir):
        os.mkdir(curr_output_dir)
    if not os.path.exists(os.path.join(curr_output_dir,'figs')):
        os.mkdir(os.path.join(curr_output_dir,'figs'))

    # Load and prepare quote analysis dfs
    print("\nCreating dataframes for analysis...")
    analysis_dfs = []
    main_df = pd.read_pickle(args.path_to_input)
    main_df['s_lemmas'] = main_df['s_lemmas'].apply(lambda x: json.loads(x))
    main_df['advs'] = main_df['advs'].apply(lambda x: json.loads(x))
    main_df['modals'] = main_df['modals'].apply(lambda x: json.loads(x))
    main_df['other_verbs'] = main_df['other_verbs'].apply(lambda x: json.loads(x))
    main_df['mwe_tok_s_lemmas'] = main_df['mwe_tok_s_lemmas'].apply(lambda x: set(json.loads(x)))
    main_df['abs_quote_stance'] = main_df['abs_quote_stance'].apply(lambda x: {0: 'anti', 1: 'neutral', 2: 'pro'}[x])

    # Named Entity analyses
    fuzzy_match_df = pd.read_csv('fuzzy_match_for_refactor.tsv',sep='\t',index_col=0)
    subj2canonical = dict(zip(fuzzy_match_df['fuzzy_match'],fuzzy_match_df['canonical']))

    NE_stance_dict = pickle.load(open('NE_stance_dict.pkl','rb'))
    PRO_CC_ENTS = [x for x in NE_stance_dict if 'cc' in NE_stance_dict[x] and
                         NE_stance_dict[x]['cc'] == 'pro']
    ANTI_CC_ENTS = [x for x in NE_stance_dict if 'cc' in NE_stance_dict[x] and
                            NE_stance_dict[x]['cc'] == 'anti']
    ent2stance = {}
    ent2stance.update({ent: 'pro' for ent in PRO_CC_ENTS})
    ent2stance.update({ent: 'anti' for ent in ANTI_CC_ENTS})
    main_df['main_s_resolved'] = main_df['main_s_lemma'].apply(lambda x: subj2canonical[x.lower()]
                                                          if type(x)==str and x.lower() in subj2canonical else x)
    main_df['main_s_stance'] = main_df['main_s_resolved'].apply(lambda x: ent2stance[x]
                                                                if x in ent2stance else None)

    print("Limiting analysis dataframe to Opinions not in the scope of negation...")
    analysis_df = main_df.loc[main_df.neg_type=='no_neg'].copy()
    print("New shape:",analysis_df.shape)
    print("Further limiting analysis dataframe to Opinions with non-neutral stance...")
    analysis_df = analysis_df.loc[analysis_df.abs_quote_stance!='neutral'].copy()
    print("New shape:",analysis_df.shape)
    analysis_dfs.append((analysis_df,'full'))



    if args.do_subsampled:
        print("Creating subsampled analysis df excluding top 5 LL/RL outlets...")
        most_common_L = sorted(Counter(analysis_df.loc[analysis_df['outlet_stance']=='pro']['domain']).items(),
                       key=lambda x: x[1],reverse=True)
        most_common_R = sorted(Counter(analysis_df.loc[analysis_df['outlet_stance']=='anti']['domain']).items(),
                               key=lambda x: x[1],reverse=True)
        top_10_domains = set([x[0] for x in most_common_R[:5]]+[x[0] for x in most_common_L[:5]])
        print("\tTop 10 outlets:",top_10_domains)
        subsampled_df = analysis_df.loc[~analysis_df['domain'].isin(top_10_domains)]
        print("\tSubsampled df shape:",subsampled_df.shape)
        analysis_dfs.append((subsampled_df,'subsampled'))

    if args.do_nonwire_dataset:
        articles_df = pd.read_pickle(os.path.join(config["SCRAPE_DIR"],'output',
            'filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl'))
        non_wire_guids = set(articles_df.loc[articles_df['is_AP']==False]['guid'].values)
        print("Creating non-wire df excluding news wire articles...")
        nonwire_df = analysis_df.loc[analysis_df['guid'].isin(non_wire_guids)]
        print('\tNon-wire df shape:',nonwire_df.shape)
        analysis_dfs.append((nonwire_df,'nonwire'))

    # Load lexicons
    WORD_CATS = load_lexicons()

    for analysis_df_,analysis_df_name in analysis_dfs:
        print("\nAnalyzing {} dataset...".format(analysis_df_name))
        print("\nGenerating plots of opinion-framing patterns...")
        get_coverage_proportions(set(WORD_CATS['AFFIRM']),set(WORD_CATS['DOUBT']),analysis_df_,analysis_df_name,verbose=False)
        res = compute_and_plot_device_biases('abs_quote_stance',analysis_df_,analysis_df_name,WORD_CATS)

        print("\nGetting sub-frame with Opinions attributed to entity with a stance label...")
        study2_analysis_df = analysis_df_.loc[~pd.isnull(analysis_df_['main_s_stance'])].copy()
        print("\tshape:",study2_analysis_df.shape)

        study2_analysis_df['main_s_stance_int'] = study2_analysis_df['main_s_stance'].apply(
            lambda x: {'pro':1,'anti':-1}[x] if x == 'pro' or x == 'anti' else 0)
        study2_analysis_df['abs_quote_stance_int'] = study2_analysis_df['abs_quote_stance'].apply(
            lambda x: {'pro':1,'anti':-1}[x] if x == 'pro' or x == 'anti' else 0)
        faithful_df = study2_analysis_df.loc[study2_analysis_df['main_s_stance']==
                                        study2_analysis_df['abs_quote_stance']]
        unfaithful_df = study2_analysis_df.loc[(study2_analysis_df['main_s_stance_int']==
                                        study2_analysis_df['abs_quote_stance_int']*-1) &
                                          (study2_analysis_df['main_s_stance_int']!=0)]
        faithful_df_outlet_counts = faithful_df['outlet_stance'].value_counts()
        unfaithful_df_outlet_counts = unfaithful_df['outlet_stance'].value_counts()
        n_faithful_LL, n_faithful_RL = faithful_df_outlet_counts['pro'], faithful_df_outlet_counts['anti']
        n_unfaithful_LL, n_unfaithful_RL = unfaithful_df_outlet_counts['pro'], unfaithful_df_outlet_counts['anti']

        print('Within Opinions attributed to known stance Source:')
        print('\tPercentage of unfaithfully attributed Opinions in LL:',
              round(n_unfaithful_LL/(n_faithful_LL+n_unfaithful_LL),2))
        print('\tPercentage of unfaithfully attributed Opinions in RL:',
              round(n_unfaithful_RL/(n_faithful_RL+n_unfaithful_RL),2))
