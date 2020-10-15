#!/usr/bin/env python

import pickle
import pandas as pd
import numpy as np
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import string
from dateutil.parser import parse
import argparse
from tabulate import tabulate

def prettify_domain(x):
    if x == 'nyt':
        return 'New York Times'
    elif x == 'mj':
        return 'Mother Jones'
    elif x == 'cs_monitor':
        return 'Christian Science Monitor'
    elif x == 'guardian_us':
        return 'Guardian (US)'
    elif x == 'wapo':
        return 'Washington Post'
    elif x == 'https://shotofprevention/':
        return 'Shot of Prevention'
    elif x == 'nat_review':
        return 'National Review'
    elif x == 'chd':
        return "Children's Health Defense"
    elif x == 'pj':
        return 'Pajamas Media'
    elif x == 'nation':
        return 'The Nation'
    elif x == 'dem_now':
        return 'Democracy Now'
    elif x == 'usa_today':
        return 'USA Today'
    elif x == 'https://www.voicesforvaccines.org/blog/':
        return 'Voices for Vaccines'
    elif x == 'cns':
        return 'CNS'
    elif x == 'nbc':
        return 'NBC'
    elif x == 'https://www.citizens.org/':
        return 'Citizens'
    elif x == 'inthesetimes':
        return 'In These Times'
    elif x == 'activistpost':
        return 'Activist Post'
    elif x == 'https://adultvaccinesnow.org/blog/':
        return 'Adult Vaccines Now'
    elif x == 'newswithviews':
        return 'News With Views'
    elif x == 'sonsoflibertymedia':
        return 'Sons of Liberty Media'
    elif x == 'libertyunyielding':
        return 'Liberty Unyielding'
    elif x == 'conservativedailynews':
        return 'Conservative Daily News'
    elif x == 'https://immunizationevidence.org/featured_issues/':
        return 'Immunization Evidence'
    elif x == 'conservativefiringline':
        return 'Conservative Firing Line'
    elif x == 'progressivestoday':
        return 'Progressives Today'
    elif x == 'independentsentinel':
        return 'Independent Sentinel'
    elif x == 'https://physiciansforinformedconsent.org/':
        return 'Physicians for Informed Consent'
    elif x == 'charismanews':
        return 'Charisma News'
    elif x == 'cbn':
        return 'CBN'
    elif x == 'www.washingtonexaminer':
        return 'Washington Examiner'
    elif x == 'www.thetruthaboutguns':
        return 'The Truth About Guns'
    elif x == 'www.thenewamerican':
        return 'The New American'
    elif x == 'www.campusreform.org':
        return 'Campus Reform'
    elif x == 'www.investors':
        return 'Investors'
    elif x == 'www.hurriyetdailynews':
        return "Hurriyet Daily News"
    elif x == 'www.weaselzippers.us':
        return 'Weasel Zippers'
    elif x == 'www.beliefnet':
        return 'Belief Net'
    elif x == 'www.dailywire':
        return 'Daily Wire'
    elif x == 'www.libertyheadlines':
        return 'Liberty Headlines'
    elif x == 'www.weeklystandard':
        return 'Weekly Standard'
    elif x == 'www.aei.org':
        return 'American Enterprise Institute'
    elif x == 'www.thecollegefix':
        return 'The College Fix'
    elif x == 'www.ammoland':
        return 'Ammoland'
    elif x == 'www.academia.org':
        return 'Academia'
    elif x == 'www.ronpaulinstitute.org':
        return 'Ron Paul Institute'
    elif x == 'www.christianpost':
        return 'Christian Post'
    elif x == 'www.bizpacreview':
        return 'Biz Pac Review'
    elif x == 'www.mercatornet':
        return 'Mercator Net'
    elif x == 'www.aim.org':
        return "Accuracy in Media"
    elif x == 'www.lifezette':
        return 'Lifezette'
    elif x == 'www.theepochtimes':
        return 'The Epoch Times'
    elif x == 'www.heritage.org':
        return 'Heritage Foundation'
    elif x == 'www.libertynation':
        return 'Liberty Nation'
    elif x == 'www.hudson.org':
        return 'Hudson'
    elif x == 'www.lifenews':
        return 'Life News'
    elif x == 'www.jewishworldreview':
        return 'Jewish World Review'
    elif x == 'www.ff.org':
        return 'Frontiers of Freedom'
    elif x == 'www.trtworld':
        return 'TRT World'
    elif x == 'www.freedomworks.org':
        return 'Freedom Works'
    elif x == 'www.manhattan-institute.org':
        return 'Manhattan Institute'
    elif x == 'www.nraila.org':
        return 'NRA-ILA'
    elif x == 'therealnews':
        return 'The Real News'
    elif x == 'www.getreligion.org':
        return 'Get Religion'
    elif x == 'www.illinoispolicy.org':
        return 'Illinois Policy'
    elif x == 'www.armstrongeconomics':
        return 'Armstrong Economics'
    elif x == 'www.gopusa':
        return 'GOP USA'
    elif x == 'www.alec.org':
        return 'Alec'
    elif x == 'www.intellectualtakeout.org':
        return 'Intellectual Takeout'
    elif x == 'patriotpost.us':
        return 'Patriot Post'
    elif x == 'conservativedailynews':
        return 'Conservative Daily News'
    elif x == 'independentsentinel':
        return 'Independent Sentinel'
    elif x == 'www.patriotnewsalerts':
        return 'Patriot News Alerts'
    elif x == 'www.thepostmillennial':
        return 'The Post Millennial'
    elif x == 'www.mrc.org':
        return 'Media Research Center'
    elif x == 'www.jewishpolicycenter.org':
        return 'Jewish Policy Center'
    elif x == 'www.teapartypatriots.org':
        return 'Tea Party Patriots'
    elif x == 'www.ronpaullibertyreport':
        return 'Ron Paul Liberty Report'
    elif x == 'sonsoflibertymedia':
        return 'Sons of Liberty Media'
    elif x == 'libertyunyielding':
        return 'Liberty Unyielding'
    elif x == 'www.yaf.org':
        return "Young America's Foundation"
    elif x == 'progressivestoday':
        return 'Progressives Today'
    elif x == 'www.rd':
        return "Reader's Digest"
    elif x == 'newswithviews':
        return 'News with Views'
    elif x == 'conservativefiringline':
        return "Conservative Firing Line"
    elif x == 'commdiginews':
        return 'Communities Digital News'
    elif x == 'www.numbersusa':
        return 'Numbers USA'
    elif x == 'rare.us':
        return 'Rare'
    elif x == 'www.restoreamericanglory':
        return 'Restore American Glory'
    elif x == 'www.mediacircus':
        return 'Media Circus'
    elif x == 'newsbusters':
        return 'News Busters'
    elif x == 'www.populistwire':
        return 'Populist Wire'
    elif x == 'www.onenewsnow':
        return 'One News Now'
    elif x == 'www.christiannewsalerts':
        return 'Christian News Alerts'
    elif x == 'www.afpc.org':
        return 'Air Force Personnel Center'
    elif x == 'www.oann':
        return 'One American News Network'
    elif x == 'charismanews':
        return 'Charisma News'
    elif x == 'www.nationalcenter.org':
        return 'National Center'
    elif x == 'www.unwatch.org':
        return 'UN Watch'
    elif x == 'www.americanlibertyreport':
        return 'American Liberty Report'
    elif x == 'www.independentsentinel':
        return 'Independent Sentinel'
    elif x == 'www.acting-man':
        return 'Acting Man'
    elif x == 'www.forbes':
        return 'Forbes'
    elif x == 'www.washingtontimes':
        return 'Washington Times'
    elif x == 'www.chicagotribune':
        return 'Chicago Tribune'
    elif x == 'www.marketwatch':
        return 'Market Watch'
    elif x == 'www.sun-sentinel':
        return 'Sun Sentinel'
    elif x == 'www.dispatch':
        return 'Dispatch'
    elif x == 'www.dallasnews':
        return 'Dallas News'
    elif x == 'wwwmdiginews':
        return 'Communities Digital News'
    elif x == 'wwwmentarymagazine':
        return 'Commentary Magazine'
    elif x == 'www.omaha':
        return 'Omaha'
    elif x == 'www.conservativedailynews':
        return 'Conservative Daily News'
    elif x == 'www.crisismagazine':
        return 'Crisis Magazine'
    elif x == 'www.economicpolicyjournal':
        return 'Economic Policy Journal'
    elif x == 'www.charismanews':
        return 'Charisma News'
    elif x == 'www.city-journal.org':
        return 'City Journal'
    elif x == 'Www.post-gazette':
        return 'Post Gazette'
    elif x == 'houstonchronicle':
        return 'Houston Chronicle'
    elif x == 'www.conservativereview':
        return 'Conservative Review'
    elif x == 'www.conservativehq':
        return 'Conservative HQ'
    elif x == 'honululuadvertiser':
        return 'Honolulu Advertiser'
    elif x == 'realclearpolitics':
        return 'Real Clear Politics'
    elif x == 'www.sgtreport':
        return 'Sgt Report'
    elif x == 'www.firstthings':
        return 'First Things'
    elif x == 'www.nccivitas.org':
        return 'Civitas Institute'
    elif x == 'www.powerlineblog':
        return 'Powerline Blog'
    elif x == 'www.theamericanconservative':
        return 'The American Conservative'
    elif x == 'www.calgarysun':
        return 'Calgary Sun'
    elif x == 'www.opslens':
        return 'Ops Lens'
    elif x == 'www.chicksontheright':
        return 'Chicks on the Right'
    elif x == 'theroot':
        return 'The Root'
    elif x == 'www.torontosun':
        return 'Toronto Sun'
    elif x == 'www.campaignlifecoalition':
        return 'Campaign Life Coalition'
    elif x == 'pilotonline':
        return 'Pilot Online'
    elif x == 'www.cbn' or x == 'cbn':
        return 'Christian Broadcast Network'
    elif x == 'azcentral':
        return 'AZ Central'
    elif x == 'www.westernfreepress':
        return 'Western Free Press'
    elif x == 'www.influencewatch.org':
        return 'Influence Watch'
    elif x == 'www.conservativefiringline':
        return 'Conservative Firing Line'
    elif x == 'www.drudgereport':
        return 'Drudge Report'
    elif x == 'www.colddeadhands.us':
        return 'Cold Dead Hands'
    elif x == 'www.westmonster':
        return 'West Monster'
    elif x == 'bgr':
        return 'Boy Genius Report'
    elif x == 'www.post-gazette':
        return 'Post Gazette'
    elif x == 'www.faithwire':
        return 'Faith Wire'
    elif x == 'superglue:_msnbc':
        return 'MSNBC'
    elif x == 'msnbc':
        return 'MSNBC'
    elif x == 'www.defenddemocracy.org':
        return 'Defend Democracy'
    elif x == 'sgtreport':
        return 'Sgt Report'
    elif x == 'pj_media':
        return 'Pajamas Media'
    elif x == 'www.citizenfreepress':
        return 'Citizen Free Press'
    elif x == 'www.thedailyliberator':
        return 'The Daily Liberator'
    elif x == 'www.freerepublic':
        return 'Free Republic'
    elif x == 'www.newswithviews':
        return 'News with Views'
    elif x == 'www.blacknews':
        return 'Black News'
    elif x == 'www.conservativeinstitute.org':
        return 'Conservative Institute'
    elif x == 'www.aina.org':
        return 'AINA'
    elif x == 'www.cbn' or x == 'cbn':
        return 'Christian Broadcast Network'
    else:
        return ' '.join([w.capitalize() for w in x.split('_')]).strip()


def reorderLegend(ax=None,order=None,unique=False,loc='upper left',title=None,
          fontsize=18,title_fontsize=20):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels,loc=loc,title=title,
          fontsize=fontsize,title_fontsize=title_fontsize)
    return(handles, labels)


def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]


def create_data_report(dated_df):
    print('Creating report for df with shape {}'.format(dated_df.shape))

    print('\nDistribution of article leanings:')
    stance_dict = {'anti':'R-leaning','pro':'L-leaning','between':'Center'}
    print(tabulate(dated_df.stance.apply(lambda x: stance_dict[x]).value_counts().rename_axis('stance').reset_index(name='count'), headers='keys', tablefmt='psql'))

    print('\nDistribution of AP (True) vs. non-AP (False) articles:')
    print(tabulate(dated_df.is_AP.value_counts().rename_axis('is AP').reset_index(name='count'), headers='keys', tablefmt='psql'))

    print('\nDistribution of article outlets:')
    dated_df['pretty_domain'] = dated_df['domain'].apply(prettify_domain)
    fig = dated_df['pretty_domain'].value_counts().plot.pie()
    plt.show()
    fig[0].get_figure().savefig('output/outlet_distribution.png')

    print('\nDistribution of articles over time:')
    dated_df['year'] = [d.to_pydatetime().year
                             for d in dated_df.date]
    dated_df['month'] = [d.to_pydatetime().month
                             for d in dated_df.date]
    fig = dated_df.loc[dated_df.year.isin(range(2000,2021))].year.plot.hist()
    fig[0].get_figure().savefig('output/temporal_distribution.png')

    print('\nDistribution of article outlets over time:')
    top_domains = {'pro':set(dated_df.loc[dated_df.stance == 'pro'].\
    pretty_domain.value_counts().index[:9]),
                   'anti':set(dated_df.loc[dated_df.stance == 'anti'].\
    pretty_domain.value_counts().index[:9])}

    dfs = {}
    for side in ['pro','anti']:
        df = pd.DataFrame(columns=['outlet']+list(range(2007,2021)))
        for outlet in top_domains[side]:
            row = [outlet]
            counts = dated_df.loc[dated_df.pretty_domain == outlet].year.value_counts()
            row.extend([counts[int(c)] if c in counts else 0 for c in range(2007,2021)])
            row_df = pd.DataFrame(row).T
            row_df.columns = df.columns
            df = df.append(row_df,
                           ignore_index=True)
        row = ['other']
        counts = dated_df.loc[(dated_df.stance == side) &
                                 (~dated_df.pretty_domain.isin(top_domains[side]))].year.value_counts()
        row.extend([counts[int(c)] if c in counts else 0 for c in range(2007,2021)])
        row_df = pd.DataFrame(row).T
        row_df.columns = df.columns
        dfs[side] = df.append(row_df,
                       ignore_index=True)

    sns.set_palette('colorblind')
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(22,8),sharey=True)
    dfs['pro'].set_index('outlet').T.plot(kind='bar', stacked=True, fontsize=24,ax=ax1)
    dfs['anti'].set_index('outlet').T.plot(kind='bar', stacked=True, fontsize=24,ax=ax2)
    ax1.set_ylabel('Count',fontsize=32)
    ax2.set_ylabel('')
    ax1.set_xlabel('Left-wing media',fontsize=32)
    ax2.set_xlabel('Right-wing media',fontsize=32)
    ax1.set_xticklabels(rotation=0,labels=range(2007,2021))
    ax2.set_xticklabels(rotation=0,labels=range(2007,2021))
    n = 4
    for ax in [ax1,ax2]:
        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    ax1.legend()
    handles, labels = ax1.get_legend_handles_labels()
    reorderLegend(ax1,labels[:-1][::-1]+['other'],
                 title='Media outlet',fontsize=24,title_fontsize=26)
    ax2.legend()
    handles, labels = ax2.get_legend_handles_labels()
    reorderLegend(ax2,labels[:-1][::-1]+['other'],
                 title='Media outlet',fontsize=24,title_fontsize=26)

    for n, ax in enumerate([ax1,ax2]):
        ax.text(-0.03, 1.05, '{})'.format(string.ascii_lowercase[n]), transform=ax.transAxes,
                size=28, weight='bold')
    plt.tight_layout()
    plt.show()
    fig.savefig('output/top_RL_LL_outlets_over_time.png')

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_filename', type=str, default='output/filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl', help='/path/to/dataset/to/explore')
    args = arg_parser.parse_args()

    dedup_df = pd.read_pickle(args.input_data_filename)
    create_data_report(dedup_df)
