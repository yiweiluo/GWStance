#!/usr/bin/env python

import pandas as pd
import os
import argparse
# os.chdir('..')
# from local_processors import fulltext_exists,get_fulltext
# os.chdir('./1_data_scraping')

def fulltext_exists(url_guid,fnames_set):
    return '{}.txt'.format(url_guid) in fnames_set


def get_fulltext(url_guid,fulltext_dir,fnames_set):
    if fulltext_exists(url_guid,fnames_set):
        with open(os.path.join(fulltext_dir,url_guid+'.txt'),'r') as f:
            lines = f.readlines()
        if len(lines) > 0:
            return lines[0]
        return ""
    return ""


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_df_filename', type=str, default='output/dedup_combined_df_2000_1_1_to_2020_4_12.pkl', help='/path/to/dataset/to/explore')
    arg_parser.add_argument('--url_text_dir', type=str, default='new_url_texts', help='/path/to/url/fulltext/directory')
    arg_parser.add_argument('--output_df_filename', type=str, default='filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl', help='/path/to/save/filtered/dataset')
    args = arg_parser.parse_args()

    dedup_df = pd.read_pickle(args.input_df_filename)
    print('Input df shape:',dedup_df.shape)
    ft_set = set(os.listdir(args.url_text_dir))

    # Filter to subset of articles with a non-null publish date and
    # non-empty fulltext
    print('Filtering to articles with successfully scraped fulltext...')
    dedup_df_ft = dedup_df.loc[dedup_df.guid.apply(lambda x: fulltext_exists(x,ft_set))]
    print('\tNew df shape:',dedup_df_ft.shape)
    print('Filtering to articles with non-empty fulltext...')
    dedup_df_nonempty_ft = dedup_df_ft.loc[dedup_df_ft.guid.apply(
    lambda x: len(get_fulltext(x,args.url_text_dir,ft_set)) > 0)]
    print('\tNew df shape:',dedup_df_nonempty_ft.shape)
    print('Filtering to articles with non-null publish date...')
    dedup_df_ft_date = dedup_df_nonempty_ft.loc[~pd.isnull(dedup_df_nonempty_ft.date)]
    print('\tFinal df shape:',dedup_df_ft_date.shape)
    print('Saving filtered df to {}'.format(os.path.join('output',args.output_df_filename)))
    dedup_df_ft_date.to_pickle(os.path.join('output',args.output_df_filename))
