#!/usr/bin/env python

import pandas as pd
import os
from local processors import fulltext_exists,get_fulltext

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_filename', type=str, default='output/dedup_combined_df_2000_1_1_to_2020_4_12.pkl', help='/path/to/dataset/to/explore')
    arg_parser.add_argument('--url_text_dir', type=str, default='new_url_texts', help='/path/to/url/fulltext/directory')
    arg_parser.add_argument('--output_data_filename', type=str, default='filtered_dedup_combined_df_2000_1_1_to_2020_4_12.pkl', help='/path/to/save/filtered/dataset')
    args = arg_parser.parse_args()

    dedup_df = pd.read_pickle(args.input_data_filename)
    print('Input df shape:',dedup_df.shape)
    ft_set = set(os.listdir(args.url_text_dir))

    # Filter to subset of articles with a non-null publish date and
    # non-empty fulltext
    print('Filtering to articles with successfully scraped fulltext...')
    dedup_df_ft = dedup_df.loc[dedup_df.guid.apply(lambda x: fulltext_exists(x,ft_set))]
    print('\tNew df shape:',dedup_df_ft.shape)
    print('Filtering to articles with non-empty fulltext...')
    dedup_df_nonempty_ft = dedup_df_ft.loc[dedup_df_ft.guid.apply(
    lambda x: len(get_fulltext(x,args.url_text_dir)) > 0)]
    print('\tNew df shape:',dedup_df_nonempty_ft.shape)
    print('Filtering to articles with non-null publish date...')
    dedup_df_ft_date = dedup_df_nonempty_ft.loc[~pd.isnull(dedup_df_nonempty_ft.date)]
    print('\tFinal df shape:',dedup_df_ft_date.shape)
    print('Saving filtered df to {}'.format(os.path.join('output',args.output_data_filename)))
    dedup_df_ft_date.to_pickle(os.path.join('output',args.output_data_filename))
