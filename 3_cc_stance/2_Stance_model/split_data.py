import os
import random
from optparse import OptionParser

import numpy as np
import pandas as pd


# Script to take the output of the item response model and create multiple folds of data for training and evaluation

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--folds', type=int, default=5,
                      help='Number of test folds: default=%default')
    parser.add_option('--seed', type=int, default=42,
                      help='Random seed: default=%default')
    parser.add_option('--basedir', type=str, default='splits',
                      help='Base output directory: default=%default')

    (options, args) = parser.parse_args()

    infile = 'data/sent_scores_df_final.tsv'
    infile_test = 'data/held_out_balanced_test.tsv'

    df = pd.read_csv(infile, header=0, index_col=0, sep='\t')
    print("All data:", df.shape)
    test_df = pd.read_csv(infile_test, header=0, index_col=None, sep='\t')
    test_indices = np.arange(len(test_df))
    print("Test data:", test_df.shape)

    all_rounds = df['round'].values
    all_batches = df['batch'].values
    all_ids = df['sent_id'].values
    all_guids = []
    for i in range(len(all_rounds)):
        all_guids.append(str(all_rounds[i]) + '_' + str(all_batches[i]) + '_' + str(all_ids[i]))
    df['guid'] = all_guids

    test_guids = set(test_df['guid'].values)
    nontest_indices = [i for i, g in enumerate(all_guids) if g not in test_guids]
    n_nontest = len(nontest_indices)
    print("{:d} non-test".format(len(nontest_indices)))

    for weights in [False, True]:

        np.random.seed(options.seed)
        random.seed(options.seed)

        n_folds = options.folds
        basedir = options.basedir

        order = nontest_indices
        np.random.shuffle(order)

        for f in range(n_folds):
            dev_indices = [order[i] for i in np.arange(n_nontest) if i % n_folds == f]
            train_indices = list(set(nontest_indices) - set(dev_indices))
            print(len(train_indices), len(dev_indices), len(test_indices))

            if weights:
                outdir = os.path.join(basedir, 'climate-weight', 'folds',  str(f))
            else:
                outdir = os.path.join(basedir, 'climate', 'folds',  str(f))
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            outfile = os.path.join(outdir, 'train.tsv')
            write_to_file(outfile, df, train_indices, weights, train=True)

            outfile = os.path.join(outdir, 'dev.tsv')
            write_to_file(outfile, df, dev_indices, weights)

            outfile = os.path.join(outdir, 'test.tsv')
            write_to_file(outfile, test_df, test_indices, weights)

        if weights:
            outdir = os.path.join(basedir, 'climate-weight', 'folds',  'no-dev')
        else:
            outdir = os.path.join(basedir, 'climate', 'folds',  'no-dev')
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        print(len(nontest_indices), len(test_indices))
        train_indices = nontest_indices
        outfile = os.path.join(outdir, 'train.tsv')
        write_to_file(outfile, df, train_indices, weights, train=True)

        outfile = os.path.join(outdir, 'test.tsv')
        write_to_file(outfile, test_df, test_indices, weights)


def write_to_file(outfile, df, indices, weights=False, train=False):
    labels = ['disagree', 'neutral', 'agree']
    outlines = []
    label_counts = np.zeros(3)
    for index in indices:
        text = df.loc[index, 'sentence']
        text = text.strip()
        max_label = int(np.argmax(df.loc[index, labels].values))
        max_prob = np.max(df.loc[index, labels].values)
        if weights and train:
            for label_i, label in enumerate(labels):
                prob = float(df.loc[index, label])
                outlines.append(text + '\t' + labels[label_i] + '\t' + str(prob) + '\n')
                label_counts[label_i] += 1
        elif weights:
            outlines.append(text + '\t' + labels[max_label] + '\t' + str(max_prob) + '\n')
            label_counts[max_label] += 1
        else:
            outlines.append(text + '\t' + labels[max_label] + '\n')
            label_counts[max_label] += 1

    if train:
        random.shuffle(outlines)

    print(label_counts / np.sum(label_counts))
    with open(outfile, 'w') as fo:
        fo.writelines(outlines)


if __name__ == '__main__':
    main()
