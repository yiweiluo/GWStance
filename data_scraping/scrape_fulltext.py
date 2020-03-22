import urllib
import requests
import ssl
from bs4 import BeautifulSoup
from urllib.request import urlopen
from dateutil.parser import parse
from dateutil import parser
from collections import defaultdict
import re
import pickle
import os
import pandas as pd
import numpy as np

from newspaper import Article
from newspaper import ArticleException

from urllib.error import HTTPError
from nltk.tokenize import sent_tokenize
from urllib.error import URLError

def soupify(url):
    if url[:8] != 'https://':
        url = 'https://'+url

    try:
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
        try:
            con = urllib.request.urlopen( req )
        except ssl.CertificateError:
            return None
        html = con.read()
        soup = BeautifulSoup(html,'html.parser')
        return soup
    except URLError:
        return None

def newspaper_parse(url):
    if url[:8] != 'https://':
        url = 'https://'+url

    try:
        article = Article(url)
        article.download()
        article.parse()
        return (article.title,
                article.text.replace('\n',' '))
    except ArticleException:
        return (None,None)


def get_fulltext(url,domain):
    stop_ix,title,text = None,None,None
    if domain == 'alternet':
        title,text = newspaper_parse(url)
    elif domain == 'american_conservative':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'activistpost':
        title,text = newspaper_parse(url)
        stop_ix = -4
    elif domain == 'american_thinker':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'bipartisan_report':
        title,text = newspaper_parse(url)
    elif domain == 'blaze':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'boston_globe':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'breitbart':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'buzzfeed':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'cbn':
        title,text = newspaper_parse(url)
    elif domain == 'charismanews':
        title,text = newspaper_parse(url)
        stop_ix = -14
    elif domain == 'chd':
        title,text = newspaper_parse(url)
    elif domain == 'https://www.citizens.org/':
        title,text = newspaper_parse(url)
    elif domain == 'cns':
        title,text = newspaper_parse(url)
    elif domain == 'commdiginews':
        title,text = newspaper_parse(url)
    elif domain == 'conservative_review':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'conservative_treehouse':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'conservativedailynews':
        title,text = newspaper_parse(url)
        stop_ix = -4
    elif domain == 'conservativefiringline':
        title,text = newspaper_parse(url)
        stop_ix = -10
    elif domain == 'cs_monitor':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'daily_caller':
        title,text = newspaper_parse(url)
        stop_ix = -3
    elif domain == 'daily_dot':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'dem_now':
        soup = soupify(url)
        try:
            ps = soup.find('div',attrs={'itemprop':'articleBody'}).find_all('p')
        except AttributeError:
            ps = soup.find('div',attrs={'class':'story_summary'}).find_all('p')
        text = ' '.join([p.text.replace('\n', ' ') for p in ps])
    elif domain == 'drudgereport':
        title,text = newspaper_parse(url)
    elif domain == 'fox':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'gateway_pundit':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'gawker':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'grabien':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'grist':
        title,text = newspaper_parse(url)
    elif domain == 'guardian_us':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'hot_air':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'https://adultvaccinesnow.org/blog/':
        title,text = newspaper_parse(url)
    elif domain == 'https://immunizationevidence.org/featured_issues/':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'https://shotofprevention/':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'https://www.voicesforvaccines.org/blog/':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'https://icandecide.org/':
        pass
    elif domain == 'independentsentinel':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'infowars':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'inthesetimes':
        title,text = newspaper_parse(url)
    elif domain == 'libertyunyielding':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'mj':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'nat_review':
        title,text = newspaper_parse(url)
    elif domain == 'nation':
        title,text = newspaper_parse(url)
    elif domain == 'nbc':
        soup = soupify(url)
        ps = soup.find('div',attrs={'class':'article-body__content'}).\
        find_all('p',attrs={'class':'endmarkEnabled'})
        text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'new_york_magazine':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'newsweek':
        soup = soupify(url)
        ps = soup.find('div',attrs={'class':'article-content'}).find_all('p')
        text = ' '.join([p.text.replace('\n', ' ') for p in ps])
    elif domain == 'newswithviews':
        title,text = newspaper_parse(url)
        stop_ix = -10
    elif domain == 'nyt':
        try:
            soup = soupify(url)
            if soup is not None:
                ps = soup.find('section',attrs={'itemprop':'articleBody'}).find_all('p')#,recursive=False)
                text = ' '.join([p.text.replace('\n', ' ') for p in ps])
                stop_ix = -5
        except HTTPError:
            pass
    elif domain == 'pajamas_media':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'pj':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'https://physiciansforinformedconsent.org/':
        soup = soupify(url)
        ps = soup.find('div',attrs={'class':'entry-content'}).find_all('p',attrs={'class':'responsiveNews'})
        text = ' '.join([p.text.replace('\n', ' ') for p in ps])
    elif domain == 'progressivestoday':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'quartz':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'rare.us':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'reason':
        title,text = newspaper_parse(url)
        stop_ix = -3
    elif domain == 'redstate':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'sgtreport':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'shoebat':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'sonsoflibertymedia':
        title,text = newspaper_parse(url)
        stop_ix = -1
    elif domain == 'the_american_conservative':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'the_american_spectator':
        title,text = newspaper_parse(url)
    elif domain == 'the_nation':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'the_progressive':
        title,text = newspaper_parse(url)
    elif domain == 'the_verge':
        title,text = newspaper_parse(url)
    elif domain == 'the_week':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'usa_today':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'vax_safety_commission':
        root_url = 'https://vaccinesafetycommission.org/studies.html'
        soup = soupify(root_url)
        panel_bodies = soup.find_all('div',attrs={'class':'panel-body'})
        #print(len(panel_bodies))
        text = ""
        for pb in panel_bodies:
            text += pb.text.strip()
    elif domain == 'vice':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'https://www.voicesforvaccines.org/blog/':
        title,text = newspaper_parse(url)
    elif domain == 'vox':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'wapo':
        title,text = newspaper_parse(url)
        stop_ix = -2
    else:
        print('Unknown domain:',domain)

    if text is not None and len(text) > 0:
        text = text.strip()
        sent_tokens = sent_tokenize(text)

        # Remove final 2 sentences (usually about social media)
        sent_tokens = sent_tokens[:stop_ix]
        text = ' '.join(sent_tokens)

    return (title,text)

SEP_TOK = '[SEP]'


if __name__ == "__main__":
    combined_df = pd.read_pickle('temp_combined_df.pkl')
    print(combined_df.shape)

    urls_needed = []
    url_unique_keys = {}

    for n,ix in enumerate(combined_df.index[10300+3100+14900:]):
        row = combined_df.loc[ix]
        assert row.shape == (7,)
        url = row['url']
        domain = row['domain']
        title = row['title']
        try:
            newspaper_title,ft = get_fulltext(url,domain)

            # Replace title w/ newspaper title if it's longer
            if newspaper_title is not None and \
            title is not None and \
            len(newspaper_title) > len(title):
                title = newspaper_title
            # Replace title w/ newspaper title if the former is null but
            # not the latter
            elif newspaper_title is not None and title is None:
                title = newspaper_title
            else:
                pass

            if ft is not None:
                save_url = SEP_TOK.join(url.split('/'))
                try:
                    with open('./fulltexts/{}.txt'.format(save_url),'w') as f:
                        f.write(ft)
                    url_unique_keys[url] = save_url
                except OSError:
                    with open('./fulltexts/{}.txt'.format(save_url[:90]),'w') as f:
                        f.write(ft)
                    url_unique_keys[url] = save_url[:90]
            else:
                urls_needed.append(ix)
        except AttributeError:
            urls_needed.append(ix)

        if n % 100 == 0:
            print(n)

    pickle.dump(url_unique_keys,open('url_2_unique_key.pkl','wb'))
    pickle.dump(urls_needed,open('fulltext_needed_urls.pkl','wb'))
