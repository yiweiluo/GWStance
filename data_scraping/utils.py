import os
import re
import glob

fulltext_dir='./fulltexts/'
fnames = set(os.listdir(fulltext_dir))

def fulltext_exists(url,fulltext_dir=fulltext_dir):
    fname = url.replace('/','[SEP]')
    return fname+'.txt' in fnames or fname[:90]+'.txt' in fnames

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