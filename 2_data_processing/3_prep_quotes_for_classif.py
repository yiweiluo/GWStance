import os
import pandas as pd
import numpy as np
import re

def read_stem_str(s):
    """Reads string formatted set of stems into an iterable list."""
    return s[2:-2].split("', '")

def prettify(clause):
    """Clean comp. clause for classification."""
    clause = re.sub('[^a-zA-Z0-9’“”"\'% \n\.]', '', clause)

    # remove initial that, add periods, capitalize first word
    split_clause = clause.split()
    if split_clause[0] == 'that':
        split_clause = split_clause[1:]
    split_clause[0] = split_clause[0].capitalize()
    clause_str = " ".join(split_clause).strip()

    if clause_str[-1] != '.':
        clause_str += "."

    return clause_str

if __name__ == "__main__":
    quotes_df = pd.read_csv('./output/keyword_filtered_comp_clauses.tsv',sep='\t',header=0,index_col=0)
    # Parse string list into actual list of stems
    #quotes_df['quote_stem_list'] = quotes_df['quote_stems'].apply(read_stem_str)
    #quotes_df['quote_stem_list_coref'] = quotes_df['quote_stems_coref'].apply(read_stem_str)

    print('Cleaning comp. clauses for classification...')
    quotes_df['clean_quote'] = quotes_df['quote_text'].apply(prettify)
    quotes_df['clean_quote_coref'] = quotes_df['coref'].apply(prettify)
    print('Saving...')

    save_df = quotes_df[['guid','sent_no','quote_no','clean_quote','clean_quote_coref']]
    save_df.to_csv('./output/keyword_filtered_comp_clauses_for_classif.tsv'
                                                                            ,sep='\t',header=True)
    batch_size = 50000
    os.mkdir('./output/batched')
    for batch_no in range(round(len(save_df)/batch_size+0.5)):
        os.mkdir('./output/batched/{}'.format(batch_no))
        batch_df = save_df[batch_no*batch_size:batch_no*batch_size+batch_size]
        batch_df.to_csv('./output/batched/{}/test.tsv'.format(batch_no),sep='\t',header=True)

    print('Done!\n')

    # Want to divide into batches for prediction?
