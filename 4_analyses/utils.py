import os
import glob
import pandas as pd
import numpy as np
from math import log, sqrt
from collections import Counter,defaultdict
import re
import pickle
import seaborn as sns
sns.set(font_scale=2)
import matplotlib.pyplot as plt
import csv
import json
from dateutil.parser import parse
import shutil
import nltk
import scipy
from scipy import stats
from scipy.stats import chisquare

# TO DO: add config for input paths

def int2str_label(int_label):
    """
    Converts the integer class label to str.
    
    :param int_label: the int class
    """
    return {0: 'disagrees', 1: 'neutral', 2: 'agrees'}[int_label]


def get_rel_stance_label(int_label,media_bias):
    """Returns str stance label relative to the source media's bias."""
    
    if media_bias == 'anti' or media_bias == 'RL':
        return {'agrees': 'opposing', 'neutral': 'neutral', 'disagrees': 'own'}[int2str_label(int_label)]
    else:
        return {'agrees': 'own', 'neutral': 'neutral', 'disagrees': 'opposing'}[int2str_label(int_label)]
    
    
def get_abs_stance_label(rel_label,media_bias):
    """Returns the absolute stance label given a relative label and the media stance."""
    if media_bias == 'anti' or media_bias == 'RL':
        return {'own': 'disagrees', 'neutral': 'neutral', 'opposing': 'agrees'}[rel_label]
    else:
        return {'opposing': 'disagrees', 'neutral': 'neutral', 'own': 'agrees'}[rel_label]
    
    
def get_pronouns():
    """Returns the set of pronouns."""
    with open('/Users/yiweiluo/sci-debates-tester/scientific-debates/2_data_processing/pronouns.txt','r') as f:
        PRONOUNS = set(f.read().splitlines())
    return PRONOUNS


def log_odds(c, n):
    """
    Computes log odds.
    
    :param c: int count of event of interest in a sample
    :paran n: int total count of events in sample
    """
    return log(c) - log(n - c) # equivalent to log(c/(n-c)) = log((c/n)/(1-c/n)) = log(c/(n-c))


def log_odds_ratio(c1, n1, c2, n2):
    """
    Computes log odds ratio between log_odds(c1, n1) and log_odds(c2, n2).
    
    :param c1: int count of event in corpus 1
    :param n1: int total count of events in corpus 1
    :param c2: int count of event in corpus 2
    :param n2: int total count of events in corpus 2
    """
    return log_odds(c1, n1) - log_odds(c2, n2)