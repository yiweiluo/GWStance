#!/usr/bin/env python
import argparse
import re
import pandas as pd
import os
import json

config = json.load(open('../config.json', 'r'))
REMOTE_SCRAPE_DIR = config['REMOTE_SCRAPE_DIR']

def regularize_title(t):
    return re.sub("[^A-Za-z0-9.,']+", ' ', t.lower().strip()).strip()

def d_l_dist(s1, s2):
    """Compute Damerau-Levenshtein distance between s1 and s2"""
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

def is_same(u1,u2):
    """Determine whether Djk ≤ 0.2 × Min.[|Tj|,|Tk|]"""
    D_jk = d_l_dist(u1,u2)
    t_j = len(u1)
    t_k = len(u2)
    min_ = min(t_j,t_k)
    return D_jk < 0.2*min_



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--use_remote', action="store_true", help='whether script is being run on remote cluster')
    arg_parser.add_argument('--input_df_filename', type=str, default=None,
                      help='/path/to/df/with/urls/and/titles')
    arg_parser.add_argument('--output_df_filename', type=str, default=None,
                                        help='/path/to/saved/df/with/urls/and/titles')
    args = arg_parser.parse_args()
    from_name = os.path.join(REMOTE_SCRAPE_DIR,args.input_df_filename) if args.use_remote else args.input_df_filename
    save_name = os.path.join(REMOTE_SCRAPE_DIR,args.output_df_filename) if args.use_remote else args.output_df_filename

    combined_df_ft = pd.read_pickle(from_name)
    print('Read in data with shape {}.'.format(combined_df_ft.shape))
    print('Regularizing titles...')
    combined_df_ft['reg_title'] = combined_df_ft['title'].apply(regularize_title)
    print('Grouping by outlet sources...')
    outlet_groups = combined_df_ft.groupby('domain')

    for outlet in outlet_groups.first().index:
        outlet_df = outlet_groups.get_group(outlet)
        print('Processing {} with {} URLS'.format(outlet,len(outlet_df)))
        for ix1 in range(len(outlet_df.index)-1):
            for ix2 in range(ix1+1,len(outlet_df.index)):
                index1 = outlet_df.index[ix1]
                index2 = outlet_df.index[ix2]
                #print('Comparing titles of {} and {}...'.format(index1,index2))
                t1 = outlet_df.loc[index1].reg_title
                t2 = outlet_df.loc[index2].reg_title
                #print('Titles: {}, {}'.format(t1,t2))
                if is_same(t1,t2):
                    print('Match found!')
                    # Set reg_title of t1 to be t2
                    print("Changing df title value from '{}' to '{}'".format(t1,t2))
                    print('Previous row:', combined_df_ft.loc[index1])
                    print('Match row:', combined_df_ft.loc[index2])
                    combined_df_ft.at[index1,'reg_title'] = t2

    combined_df_ft = combined_df_ft.drop_duplicates('reg_title',keep='first')
    print('Finished! New shape: {}. Saving deduplicated df to {}...'.format(combined_df_ft.shape,save_name))
    combined_df_ft.to_pickle(save_name)
