import os
import re
import glob
import json

from urllib.error import URLError
import urllib
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen

def soupify(url):
    if url[:8] != 'https://' and url[:7] != 'http://':
        url = 'http://'+url
        
    try:
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"}) 
        con = urllib.request.urlopen( req )
        html = con.read()
        soup = BeautifulSoup(html,'html.parser')
        return soup
    except URLError:
        return None

# fulltext_dir='/Users/yiweiluo/scientific-debates/1_data_scraping/fulltexts/'
# fulltext_dir_2='/Users/yiweiluo/scientific-debates/1_data_scraping/cc_fulltexts/'
# fnames = set(os.listdir(fulltext_dir)) | set(os.listdir(fulltext_dir_2))

fulltext_dir = './1_data_scraping/url_texts'
meta_dir = './1_data_scraping/url_meta'
fnames = set(os.listdir(fulltext_dir))
metanames = set(os.listdir(meta_dir))

def fulltext_exists(url_guid):
    return '{}.txt'.format(url_guid) in fnames

def meta_exists(url_guid):
    return '{}.json'.format(url_guid) in metanames

def get_fulltext(url_guid):
    if fulltext_exists(url_guid):
        with open(os.path.join(fulltext_dir,url_guid+'.txt'),'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            return lines[0]
        return ""
    return ""

def get_meta(url_guid):
    if meta_exists(url_guid):
        with open(os.path.join(meta_dir,url_guid+'.json'),'r') as f:
            json_obj = json.loads(f.read())
        return json_obj