import argparse
%matplotlib inline
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os
from sklearn.cluster import KMeans
from sklearn import svm
from numpy.random import RandomState
from scipy.stats import ttest_ind
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import metrics
from datetime import timedelta
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from keras import Sequential, backend, regularizers
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend
from keras.utils import to_categorical

def label2float(x):
    if x == 0:
        return 1.0
    elif x == 2:
        return -1.0
    else:
        return 0.0


def load_data_splits(datadir):
    """
    Loads train,dev,test.tsv files from datadir.
    :param datadir: name of source directory
    :return: train_df, test_df, and dev_df if present
    """

    training_data = pd.read_csv(os.path.join(DATA_DIR,datadir,'train.tsv'),sep='\t',header=None)
    test_data = pd.read_csv(os.path.join(DATA_DIR,datadir,'test.tsv'),sep='\t',header=None)
    if os.path.exists(os.path.join(DATA_DIR,datadir,'dev.tsv')):
        val_data = pd.read_csv(os.path.join(DATA_DIR,datadir,'dev.tsv'),sep='\t',header=None)
    else:
        val_data = None

    training_data.columns = ['sentence','stance','nli_label']
    if val_data is not None:
        val_data.columns = ['sentence','stance','nli_label']
    test_data.columns = ['sentence','stance','nli_label']

    return {'train': training_data, 'dev': val_data, 'test': test_data}


def make_corpus(datadir,train_dat,val_dat=None):
    """
    Generate the corpus for training and dev data for different featurization methods.
    :param dat: data to transform
    :return: corpus
    """
    train_dat_ = train_dat.copy()

    if val_dat is None:
        train_dat_.reset_index(drop=True,inplace=True)
        print(len(train_dat_))
        tr_ix,dev_ix = train_test_split(train_dat_.index,random_state=42,test_size=0.2)
        print(len(tr_ix),len(dev_ix))
        train_df = train_dat_.loc[tr_ix]
        val_df = train_dat_.loc[dev_ix]
        print(len(train_df),len(val_df))

    corpus_all = train_df
    corpus_all.to_csv(datadir+'_corpus.csv', sep=',')
    corpus = corpus_all['sentence']
    corpus_val = val_df
    print("length of the corpus: ",len(corpus))

    return (corpus_all,corpus_val)


def do_svm(train_corpus,val_corpus,test_dat,unigram=False):

    if unigram:
        vectorizer = CountVectorizer(stop_words='english')
    else:
        ngram_char_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3))
        ngram_word_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2,5))
        vectorizers = [('word', ngram_word_vectorizer), ('char', ngram_char_vectorizer)]
        vectorizer = FeatureUnion(vectorizers)

    trainNgram = vectorizer.fit_transform(train_corpus['sentence']).toarray()
    testNgram = vectorizer.transform(test_dat['sentence']).toarray()
    valNgram = vectorizer.transform(val_corpus['sentence']).toarray()
    print(len(train_corpus),len(trainNgram))

    clf = svm.SVC(kernel='linear') #gamma='scale'
    clf.fit(trainNgram, train_corpus['stance'].apply(label2float))
    pred_val = clf.predict(valNgram)
    score_val_u4 = metrics.accuracy_score(val_corpus['stance'].apply(label2float),
                                          pred_val)
    pred = clf.predict(testNgram)
    score_u4 = metrics.accuracy_score(test_dat['stance'].apply(label2float), pred)
    print("validation accuracy:   %0.3f" % score_val_u4)
    print("test accuracy:   %0.3f" % score_u4)


def do_rnn(train_corpus,val_corpus,test_dat):

    Y_train = train_corpus['stance']
    Y_val = val_corpus['stance']
    Y_test = test_dat['stance']
    Y_train_float = train_corpus['stance'].apply(label2float)
    Y_val_float = val_corpus['stance'].apply(label2float)
    Y_test_float = test_dat['stance'].apply(label2float)
    Y_train_cat = to_categorical(Y_train)
    Y_val_cat = to_categorical(Y_val)
    Y_test_cat = to_categorical(Y_test)

    voc_size = 8000
    max_words = 128
    tokenizer = Tokenizer(num_words=voc_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,split=' ')
    tokenizer.fit_on_texts(train_corpus)
    word_index = tokenizer.word_index
    X_train = tokenizer.texts_to_sequences(train_corpus['sentence'])
    X_train = pad_sequences(X_train,maxlen=max_words)
    X_val = tokenizer.texts_to_sequences(val_corpus['sentence'])
    X_val = pad_sequences(X_val,maxlen=max_words)
    X_test = tokenizer.texts_to_sequences(test_dat['sentence'])
    X_test = pad_sequences(X_test,maxlen=max_words)

    EMBEDDING_DIM=300
    MAX_SEQUENCE_LENGTH=max_words
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH
                                #,
                                #trainable=False
                               )

    lstm_size = 20
    dense_layer = 250
    backend.clear_session()
    model=Sequential()
    model.add(embedding_layer)# input_length=max_words))#,dropout = 0.2
    model.add(Dense(dense_layer, activation='relu'))
    model.add(LSTM(lstm_size,dropout=0.4, recurrent_dropout=0.4, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    batch_size = 64
    num_epochs = 5
    history = model.fit(X_train, Y_train_cat, validation_data=(X_val, Y_val_cat),
                        batch_size=batch_size, epochs=num_epochs)

    score_val_t5 = history.history['val_accuracy'][-1]
    score_t5 = model.evaluate(X_test, Y_test_cat, verbose=0)[1]

    print("validation accuracy:   %0.3f" % score_val_t5)
    print("test accuracy:   %0.3f" % score_t5)

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--glove_dir",
        default=None,
        type=str,
        required=True,
        help="/path/to/glove/embeddings",
    )

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Path to directory containing datasets",
    )

    parser.add_argument(
        "--dataname",
        default=None,
        type=str,
        required=True,
        help="Name of folder containing train.tsv etc.",
    )

    args = parser.parse_args()
    glove_dir = args.glove_dir
    DATA_DIR = args.data_dir

    embeddings_index = {}
    f = open(glove_dir)
    for line in f:
        values = line.split()
        word = ' '.join(values[0:len(values)-300])
        try:
            coefs = np.asarray(values[len(values)-300:], dtype='float32')
        except ValueError:
            print(line)
        embeddings_index[word] = coefs
    f.close()

    res = load_data_splits(dataname)
    print(len(res['train']),len(res['test']))

    tr_corpus,val_corpus = make_corpus(dataname,res['train'])
    do_svm(tr_corpus,val_corpus,res['test'],unigram=True)
    do_svm(tr_corpus,val_corpus,res['test'],unigram=False)
    do_rnn(tr_corpus,val_corpus,res['test'])
