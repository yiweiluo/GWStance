import csv
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import os
import re

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from statistics import *

def clean_ascii(text):
    # function to remove non-ASCII chars from data
    return ''.join(i for i in str(text) if ord(i) < 128)

def clean_space(text):
    return re.sub('\s+',' ',text)

def _stance(path, train_data_type=None, test_data_type=None):
    orig = pd.read_pickle(path)#, encoding = "latin-1", index_col = 0)
    orig['text'] = orig['text'].apply(clean_ascii).apply(clean_space)
    df = orig
    print(df.shape)

    # Get only those items from a single data type in the training data
    if test_data_type is not None:
        test_df = df.loc[df.type.isin(test_data_type)]
    print('Test df shape:',test_df.shape)

    if train_data_type is not None:
        train_df = df.loc[(df.type.isin(train_data_type)) & (~df.type.isin(test_data_type))]
    else:
        train_df = df.loc[~df.type.isin(test_data_type)]
    print('Train df shape:',train_df.shape)

    stances = ["agree", "neutral", "disagree"]
    class_nums = {s: i for i, s in enumerate(stances)}
    print('class_nums:',class_nums)
    # print(train_df.head())
    # print(train_df.stance.value_counts())
    # print(test_df.head())
    train_df_by_stance = {s: train_df.loc[train_df.stance == s] for i,s in enumerate(stances)}
    test_df_by_stance = {s: test_df.loc[test_df.stance == s] for i,s in enumerate(stances)}

    train_X_by_stance = {s: train_df_by_stance[s].text.values for i,s in enumerate(stances)}
    test_X_by_stance = {s: test_df_by_stance[s].text.values for i,s in enumerate(stances)}

    train_Y_by_stance = {s: np.array([class_nums[x] for x in train_df_by_stance[s].stance]) for i,s in enumerate(stances)}
    test_Y_by_stance = {s: np.array([class_nums[x] for x in test_df_by_stance[s].stance]) for i,s in enumerate(stances)}

    return train_X_by_stance, train_Y_by_stance, test_X_by_stance, test_Y_by_stance

def flat_stance(data_dir):
    path = Path(data_dir)
    datafile = 'flat_mturk_df.pkl'
    flat_mturk_df = pd.read_pickle(path/datafile)
    flat_mturk_df['text'] = flat_mturk_df['text'].apply(clean_ascii).apply(clean_space)
    flat_mturk_source_groups = flat_mturk_df.groupby('source')
    train_sources, dev_test_sources = train_test_split(flat_mturk_source_groups.first().index,
             test_size=0.3, random_state=seed1)
    dev_sources,test_sources = train_test_split(dev_test_sources,test_size=0.6,random_state=seed2)
    train_df = flat_mturk_df[flat_mturk_df.source.isin(train_sources)]
    dev_df = flat_mturk_df[flat_mturk_df.source.isin(dev_sources)]
    test_df = flat_mturk_df[flat_mturk_df.source.isin(test_sources)]

    train_X = train_df.text.values
    train_Y = train_df.stance.values

    dev_source_groups = dev_df.groupby('source')
    dev_X,dev_Y = [],[]
    for source in dev_source_groups.first().index:
        group = dev_source_groups.get_group(source)
        dev_X.append(group.text.values[0])
        try:
            dev_Y.append(mode(group.stance.values))
        except StatisticsError:
            dev_Y.append(group.MACE_pred.values[0])

    test_source_groups = test_df.groupby('source')
    test_X,test_Y = [],[]
    for source in test_source_groups.first().index:
        group = test_source_groups.get_group(source)
        test_X.append(group.text.values[0])
        try:
            test_Y.append(mode(group.stance.values))
        except StatisticsError:
            test_Y.append(group.MACE_pred.values[0])

    train_df = pd.DataFrame({'stance':train_Y,'text':train_X})
    dev_df = pd.DataFrame({'stance':dev_Y,'text':dev_X})
    test_df = pd.DataFrame({'stance':test_Y,'text':test_X})

    stances = ["agrees", "neutral", "disagrees"]
    class_nums = {s: i for i, s in enumerate(stances)}
    train_df_by_stance = {s: train_df.loc[train_df.stance == s] for s in stances}
    dev_df_by_stance = {s: dev_df.loc[dev_df.stance == s] for s in stances}
    test_df_by_stance = {s: test_df.loc[test_df.stance == s] for s in stances}
    #print(train_df_by_stance)

    train_X_by_stance = {s: train_df_by_stance[s].text.values for i,s in enumerate(stances)}
    dev_X_by_stance = {s: dev_df_by_stance[s].text.values for i,s in enumerate(stances)}
    test_X_by_stance = {s: test_df_by_stance[s].text.values for i,s in enumerate(stances)}

    train_Y_by_stance = {s: np.array([class_nums[x] for x in train_df_by_stance[s].stance]) for i,s in enumerate(stances)}
    dev_Y_by_stance = {s: np.array([class_nums[x] for x in dev_df_by_stance[s].stance]) for i,s in enumerate(stances)}
    test_Y_by_stance = {s: np.array([class_nums[x] for x in test_df_by_stance[s].stance]) for i,s in enumerate(stances)}
    #print(train_Y_by_stance)

    if args.downsample:
        min_N = min([len(train_X_by_stance[s]) for s in stances])
        print('Downsampling to ~{} examples per stance.'.format(min_N))
        for s in stances:
            train_X_by_stance[s] = train_X_by_stance[s][:min_N+200]

    trX = []
    trY = []
    for i,s in enumerate(stances):
        for t, y in zip(train_X_by_stance[s], train_Y_by_stance[s]):
            trX.append(t)
            trY.append(y)

    teX = []
    teY = []
    for i,s in enumerate(stances):
        for t, y in zip(test_X_by_stance[s], test_Y_by_stance[s]):
            teX.append(t)
            teY.append(y)

    vaX = []
    vaY = []
    for i,s in enumerate(stances):
        for t, y in zip(dev_X_by_stance[s], dev_Y_by_stance[s]):
            vaX.append(t)
            vaY.append(y)

    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    teY = np.asarray(teY, dtype=np.int32)
    #pd.DataFrame({'sentence':teX,'stance':teY,'type':[test_data_type]*len(teX)}).to_csv(data_dir+'/test_data_{}.csv'.format(test_data_type))

    return (trX, trY), (vaX, vaY), (teX, teY)

def stance(data_dir, train_data_type=None, test_data_type=None, splits=None):
    path = Path(data_dir)
    datafile = 'labeled_data_df.pkl'

    train_X_by_stance, train_Y_by_stance, test_X_by_stance, test_Y_by_stance = _stance(path/datafile, train_data_type=train_data_type, test_data_type=test_data_type)
    stances = ["agree", "neutral", "disagree"]
    test_props_by_stance = splits

    # split test_X, test_Y into true_test and test to merge back into train
    retrain_X_by_stance, teX_by_stance, retrain_Y_by_stance, teY_by_stance = {},{},{},{}
    for i,s in enumerate(stances):
        test_X = test_X_by_stance[s]
        test_Y = test_Y_by_stance[s]
        assert len(test_X) == len(test_Y)
        proportion = test_props_by_stance[s]
        if len(test_Y) > 0:
            retrain_X_by_stance[s],teX_by_stance[s],retrain_Y_by_stance[s],teY_by_stance[s] = train_test_split(test_X_by_stance[s], test_Y_by_stance[s], test_size=proportion, random_state=seed1)
            print('test')
            print(len(retrain_X_by_stance[s]))
        else: # if there are 0 examples of a given stance in the test set
            teX_by_stance[s],teY_by_stance[s] = [],[]
            retrain_X_by_stance[s],retrain_Y_by_stance[s] = [],[]
    #print(len(retrain_X),len(retrain_Y),len(teX),len(teY))
    for i,s in enumerate(stances):
        train_X_by_stance[s] = np.concatenate([train_X_by_stance[s],retrain_X_by_stance[s]],axis=0)
        train_Y_by_stance[s] = np.concatenate([train_Y_by_stance[s],retrain_Y_by_stance[s]],axis=0)
    #print(len(train_X),len(train_Y))

    tr_text_by_stance, va_text_by_stance, tr_stance_by_stance, va_stance_by_stance = {},{},{},{}
    for i,s in enumerate(stances):
        tr_text_by_stance[s],va_text_by_stance[s],tr_stance_by_stance[s],va_stance_by_stance[s] = train_test_split(train_X_by_stance[s], train_Y_by_stance[s], test_size=0.2, random_state=seed2)
    #tr_text, teX, tr_stance, teY = train_test_split(tr_te_text, tr_te_stance, test_size=0.1, random_state=seed2)
    if args.downsample:
        min_N = min([len(tr_text_by_stance[s]) for s in stances])
        print('Downsampling to ~{} examples per stance.'.format(min_N))
        for s in stances:
            tr_text_by_stance[s] = tr_text_by_stance[s][:min_N+200]

    trX = []
    trY = []
    for i,s in enumerate(stances):
        for t, y in zip(tr_text_by_stance[s], tr_stance_by_stance[s]):
            trX.append(t)
            trY.append(y)

    teX = []
    teY = []
    for i,s in enumerate(stances):
        for t, y in zip(teX_by_stance[s], teY_by_stance[s]):
            teX.append(t)
            teY.append(y)

    vaX = []
    vaY = []
    for i,s in enumerate(stances):
        for t, y in zip(va_text_by_stance[s], va_stance_by_stance[s]):
            vaX.append(t)
            vaY.append(y)

    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    teY = np.asarray(teY, dtype=np.int32)
    #pd.DataFrame({'sentence':teX,'stance':teY,'type':[test_data_type]*len(teX)}).to_csv(data_dir+'/test_data_{}.csv'.format(test_data_type))

    return (trX, trY), (vaX, vaY), (teX, teY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, default="downsampled") # description of data
    parser.add_argument('--save_dir', type=str, default='./save')
    parser.add_argument('--data_dir', type=str, default='../data')
    #parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data_type', type=str, default=None)
    parser.add_argument('--test_data_type', type=str, default='mturk')
    # these are the defaults that result in roughly equal amounts of disagree/agree/neutral training data
    parser.add_argument('--disagree_split_size', type=float, default=0.1)
    parser.add_argument('--neutral_split_size', type=float, default=0.5)
    parser.add_argument('--agree_split_size', type=float, default=0.5)
    parser.add_argument('--downsample',action='store_true')
    parser.add_argument('--flatten',action='store_true') # includes each MTurk label by an annotator as a datapoint

    args = parser.parse_args()
    print(args)

    # Constants
    desc = args.desc
    save_dir = args.save_dir
    data_dir = args.data_dir
    train_data_type = set(args.train_data_type.split(','))
    test_data_type = set(args.test_data_type.split(','))
    print('train data types:',train_data_type)
    print('test data types:',test_data_type)
    split_sizes = {"disagree": args.disagree_split_size,
    "neutral": args.neutral_split_size,
    "agree": args.agree_split_size}

    ## Test
    seed1 = 3535999445
    seed2 = 1236454357
    #data_dir = "../data"

    if args.flatten:
        (trX, trY), (vaX, vaY), (teX, teY) = flat_stance(data_dir)
    else:
        (trX, trY), (vaX, vaY), (teX, teY) = stance(data_dir,train_data_type=train_data_type,test_data_type=test_data_type,splits=split_sizes)

    print(trX[:5], trY[:5])
    print(len(trX))
    print(len(teX))
    print(len(teX[0]))
    print(teX[:5])

    test_df = pd.DataFrame({'sentence':teX,'stance':teY})
    train_df = pd.DataFrame({'sentence':trX,'stance':trY})
    val_df = pd.DataFrame({'sentence':vaX,'stance':vaY})
    print(test_df.stance.value_counts())
    print(train_df.stance.value_counts())
    print(val_df.stance.value_counts())

    save_path = Path(save_dir)/desc
    os.mkdir(save_path)
    test_df.to_csv(save_path/'test.tsv',sep='\t',header=None,index=False)
    train_df.to_csv(save_path/'train.tsv',sep='\t',header=None,index=False)
    val_df.to_csv(save_path/'dev.tsv',sep='\t',header=None,index=False)
