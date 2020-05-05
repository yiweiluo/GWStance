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

ssl.match_hostname = lambda cert, hostname: True
ssl._create_default_https_context = ssl._create_unverified_context

def soupify(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
        con = urllib.request.urlopen( req )
        html = con.read()
        soup = BeautifulSoup(html,'html.parser')
        return soup
    except (URLError,TimeoutError) as e:
        return None

def newspaper_parse(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return (article.title,
                article.text.replace('\n',' '))
    except (UnicodeError, ArticleException) as e:
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
    elif domain == 'the_atlanta_journal_constitution':
        soup = soupify(url)
        if soup is not None:
            ps = soup.find('div',attrs={'class':'story-text clearfix'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
            title = soup.find('div',attrs={'class':'tease__text'}).find('h1').text.strip()
            stop_ix = -5
    elif domain == 'bipartisanreport':
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
    elif domain == 'cbs_news':
        soup = soupify(url)
        ps = soup.find('section',attrs={'class':'content__body'}).find_all('p')
        text = ' '.join([p.text.replace('\n',' ') for p in ps])
        title = soup.find('header',attrs={'class':'content__header'}).find('h1').text.strip()
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
    elif domain == 'www.dailysignal' or domain == 'daily_signal':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            try:
                ps = soup.find('div',attrs={'class':'tds-content'}).find_all('p')
            except AttributeError:
                ps = soup.find('div',attrs={'class':'amp-wp-article-content'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'dallas_morning_news':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            ps = soup.find('div',attrs={'itemprop':'articleBody'})
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'dem_now':
        soup = soupify(url)
        try:
            ps = soup.find('div',attrs={'itemprop':'articleBody'}).find_all('p')
        except AttributeError:
            ps = soup.find('div',attrs={'class':'story_summary'}).find_all('p')
        text = ' '.join([p.text.replace('\n', ' ') for p in ps])
    elif domain == 'drudgereport':
        title,text = newspaper_parse(url)
    elif domain == 'forbes':
        soup = soupify(url)
        if soup is not None:
            try:
                ps = soup.find('div',attrs={'class':'article-body fs-article fs-responsive-text current-article'}).find_all('p')
                text = ' '.join([p.text.replace('\n', ' ') for p in ps])
                title = soup.find('h1').text.strip()
            except AttributeError:
                pass
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
        title = soup.find('h1').text.strip()
    elif domain == 'msnbc':
        soup = soupify(url)
        ps = soup.find('div',attrs={'itemprop':'articleBody'}).find_all('p')
        text = ' '.join([p.text.replace('\n',' ') for p in ps])
        title = soup.find('h1').text.strip()
    elif domain == 'new_york_magazine':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'new_york_sun':
        soup = soupify(url)
        if soup is not None:
            ps = soup.find('div',attrs={'itemprop':'articleBody'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
            title = soup.find('h1').text.strip()
    elif domain == 'newsbusters':
        soup = soupify(url)
        if soup is not None:
            ps = soup.find('div',attrs={'class':'field-item'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
            title = soup.find('h2').text.strip()
        else:
            title,text = newspaper_parse(url)
    elif domain == 'newsday':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('div',attrs={'class':'sticky'}).find('header').find('h1').text.strip()
            ps = soup.find('div',attrs={'id':'contentAccess'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
        else:
            title,text = newspaper_parse(url)
    elif domain == 'newsmax':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1',attrs={'itemprop':'headline'}).text.strip()
            text = soup.find('div',attrs={'id':'mainArticleDiv'}).text.replace('\n',' ')
        else:
            title,text = newspaper_parse(url)
    elif domain == 'newsweek':
        soup = soupify(url)
        ps = soup.find('div',attrs={'class':'article-content'}).find_all('p')
        text = ' '.join([p.text.replace('\n', ' ') for p in ps])
    elif domain == 'newswithviews':
        title,text = newspaper_parse(url)
        stop_ix = -10
    elif domain == 'ny_post':
        title,text = newspaper_parse(url)
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
    elif domain == 'patriotpost.us':
        title,text = newspaper_parse(url)
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
    elif domain == 'real_clear_politics':
        soup = soupify(url)
        if soup is not None:
            try:
                ps = soup.find('div',attrs={'class':'article-body-text'}).find_all('p')
                text = ' '.join([p.text.replace('\n',' ') for p in ps])
                title = soup.find('h1').text.strip()
            except AttributeError:
                try:
                    ps = soup.find('div',attrs={'class':'article_body'}).find_all('p',recursive=False)
                    text = ' '.join([p.text.replace('\n',' ') for p in ps])
                except AttributeError:
                    try:
                        ps = soup.find('div',attrs={'id':'alpha'}).find_all('p')
                        text = ' '.join([p.text.replace('\n',' ') for p in ps])
                    except AttributeError:
                        pass
        if title is None:
            try:
                title = soup.find('h2').text.strip()
            except AttributeError:
                pass

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
    elif domain == 'the_columbus_dispatch':
        title,text = newspaper_parse(url)
    elif domain == 'the_nation':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'the_progressive':
        title,text = newspaper_parse(url)
    elif domain == 'therealnews':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            try:
                ps = soup.find('div',attrs={'id':'column-content'}).find_all('p')
            except AttributeError:
                ps = soup.find('div',attrs={'class':'fl-rich-text'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'the_verge':
        title,text = newspaper_parse(url)
    elif domain == 'the_week':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'think_progress':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            ps = soup.find('div',attrs={'class':'post__content'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'townhall':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            ps = soup.find('section',attrs={'id':'article-body'}).find_all('p',recursive=False)
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    elif domain == 'unionleader':
        pass
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
    elif domain == 'washington_times':
        title,text = newspaper_parse(url)
    elif domain == 'wapo':
        title,text = newspaper_parse(url)
        stop_ix = -2
    elif domain == 'wnd':
        soup = soupify(url)
        if soup is not None:
            title = soup.find('h1').text.strip()
            ps = soup.find('div',attrs={'class':'entry-content'}).find_all('p')
            text = ' '.join([p.text.replace('\n',' ') for p in ps])
    else:
        print('Unknown domain:',domain)
        with open('unknown_domains.txt','w') as f:
            f.write(domain+'\n')

    if text is not None and len(text) > 0:
        text = text.strip()
        sent_tokens = sent_tokenize(text)

        # Remove final few sentences (usually about social media)
        sent_tokens = sent_tokens[:stop_ix]
        text = ' '.join(sent_tokens)

    return (title,text)


if __name__ == "__main__":
    combined_df = pd.read_pickle('/u/scr/yiweil/sci-debates/scraping/missing_temp_known_combined_df.pkl')
    #urls_without_ft = pickle.load(open('/u/scr/yiweil/sci-debates/scraping/new_urls_without_ft.pkl','rb'))
    #print(len(urls_without_ft))
    print(combined_df.shape)
    #combined_df.reset_index(drop=True,inplace=True)
    print(combined_df.index)
    #combined_df['guid'] = ['url_no_{}'.format(i) for i in range(len(combined_df))]
    #combined_df_missing_indices = combined_df.loc[combined_df.url.isin(urls_without_ft)]
    #covid_df = combined_df.loc[combined_df.topic=='covid']
    #print(covid_df.shape)

    new_n = 0
    if not os.path.exists('/u/scr/yiweil/sci-debates/scraping/fulltexts_{}'.format(new_n)):
        os.mkdir('/u/scr/yiweil/sci-debates/scraping/fulltexts_{}'.format(new_n))

    for n in range(len(combined_df)):
        row = combined_df.iloc[n]
        assert row.shape == (8,)
        url = row['url']
        scrape_url = 'https://'+url
        domain = row['domain']
        title = row['title']
        guid = row['guid']
        try:
            newspaper_title,ft = get_fulltext(scrape_url,domain)

            if ft is not None:
                with open('/u/scr/yiweil/sci-debates/scraping/fulltexts_{}/{}.txt'.format(new_n,guid),'w') as f:
                    f.write(ft)
        except AttributeError:
            pass

        if n % 100 == 0:
            print(n)

        if n % 5000 == 0:
            new_n = n
            if not os.path.exists('/u/scr/yiweil/sci-debates/scraping/fulltexts_{}'.format(new_n)):
                os.mkdir('/u/scr/yiweil/sci-debates/scraping/fulltexts_{}'.format(new_n))

    #pickle.dump(url_unique_keys,open('/u/scr/yiweil/sci-debates/scraping/url_2_unique_key.pkl','wb'))
    #pickle.dump(urls_needed,open('/u/scr/yiweil/sci-debates/scraping/fulltext_needed_urls.pkl','wb'))
