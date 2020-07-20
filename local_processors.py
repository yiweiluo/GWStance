import os
import re
import glob
import json

from urllib.error import URLError
import urllib
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import shutil

def soupify(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
        con = urllib.request.urlopen( req )
        html = con.read()
        soup = BeautifulSoup(html,'html.parser')
        return soup
    except (URLError,TimeoutError) as e:
        return None

# fulltext_dir='/Users/yiweiluo/scientific-debates/1_data_scraping/fulltexts/'
# fulltext_dir_2='/Users/yiweiluo/scientific-debates/1_data_scraping/cc_fulltexts/'
# fnames = set(os.listdir(fulltext_dir)) | set(os.listdir(fulltext_dir_2))

BASE_DIR = '/Users/yiweiluo/scientific-debates'
QUOTES_DIR = os.path.join(BASE_DIR,'2_data_processing','url_quotes')
FULLTEXT_DIR = os.path.join(BASE_DIR,'1_data_scraping','url_texts')
#fnames = set(os.listdir(fulltext_dir))

def fulltext_exists(url_guid,fnames_set):
    return '{}.txt'.format(url_guid) in fnames_set

def get_fulltext(url_guid,fulltext_dir):
    if fulltext_exists(url_guid):
        with open(os.path.join(fulltext_dir,url_guid+'.txt'),'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            return lines[0]
        return ""
    return ""

def mv_files(subdir_name,outerdir_name):
    """Moves contents of subdir_name (usually smaller batches) to outerdir_name."""
    print('Moving contents of {} to {}...'.format(subdir_name,outerdir_name))
    print('Size of outerdir:',len(os.listdir(outerdir_name)))
    inner_fs = os.listdir(os.path.join(outerdir_name,subdir_name))
    print('Size of subdir:',len(inner_fs))
    for f in inner_fs:
        os.rename(os.path.join(outerdir_name,subdir_name,f),os.path.join(outerdir_name,f))
    print('New size of outerdir:',len(os.listdir(outerdir_name)))
    shutil.rmtree(os.path.join(outerdir_name,subdir_name))

def read_quote_json(url_guid,quotes_dir):
    with open(os.path.join(quotes_dir,'{}.json'.format(url_guid)),'r') as f:
        contents = f.read()
        if len(contents) > 0:
            return json.loads(contents)
        return None
