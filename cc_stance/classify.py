from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import  confusion_matrix
import numpy
import gensim
import sys
import pickle
import csv
import os

"""
Script for running classification experiments
The script takes a directory as an argument. In this directory classified data in the
format described here should be available
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_files.html
The script performs 30-fold cross validation on this data with the LinearSVC class
"""
class DataVectorizer():

    def __init__(self):
        self.model_path = "/Users/yiweiluo/Downloads/datasets/GoogleNews-vectors-negative300.bin.gz"
        self.word2vec_model = None
        self.semantic_vector_length = 300
        self.default_vector = numpy.array([0] * self.semantic_vector_length)
        self.use_word2vec = False
        self.use_feature_select = False

    def get_vector(self, word):
        if not self.use_word2vec :
            return self.default_vector
        """
        load the semantic space in the memory
        """

        if self.word2vec_model == None:
            print("Loading word2vec model, this might take a while ....")
            # If an older version of gensim is used, use this instead:
            #self.word2vec_model = gensim.models.Word2Vec.load_word2vec_format(self.model_path, binary=True)
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)
            print("Loaded word2vec model")
        try:
            return self.word2vec_model[word]
        except KeyError:
            return self.default_vector

    def pre_process_data(self, data):
        return [el.replace("\u0096", "-").replace(". . .", " ").replace("...", " ").replace("I ", "i_i ").replace("’", "") for el in data]

    def get_semantic_features(self, data):
        # Determine which tokens are used for each data point
        vectorizer_to_find_tokens = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=0.01, max_df=1.0, ngram_range=(1, 1), stop_words = self.stop_word_list_longer)
        #print('data:',data)
        transformed_to_find_tokens = vectorizer_to_find_tokens.fit_transform(data)
        semantic_features = []
        for data_point in data:
            transformed_point = vectorizer_to_find_tokens.transform([data_point])
            inversed = vectorizer_to_find_tokens.inverse_transform(transformed_point)[0]
            summed_tokens = numpy.copy(self.default_vector)
            for token in inversed:
                summed_tokens = summed_tokens + self.get_vector(token)
            semantic_features.append(summed_tokens)
        return semantic_features

    def combine_token_data(self, transformed_x, transformed_x_more_stop_words):
        train_data_combined = []
        for el1, el2 in zip(transformed_x.toarray(), transformed_x_more_stop_words.toarray()):
            concatenated = numpy.concatenate([el1,el2])
            train_data_combined.append(concatenated)
        return train_data_combined

    def combined_with_semantic(self, X_train_selected, semantic_features):
        train_data_combined_with_semantic = []
        for el1, el2, in zip(X_train_selected, semantic_features):
            concatenated_with_semantic = numpy.concatenate([el1,el2])
            train_data_combined_with_semantic.append(concatenated_with_semantic)
        return train_data_combined_with_semantic

    def fit_transform(self, data, targets):
        data = self.pre_process_data(data)

        # Load two types of stop word lists, one very short (since normal stop words such as "not", might be useful for the classifier), and one
        # longer with normal stop words.
        stop_word_file = open("nltk_english_filtered_stopwords.txt")
        stop_word_list = [el.strip() for el in stop_word_file.readlines()]
        stop_word_file_longer = open("nltk_english_filtered_stopwords_added.txt")
        self.stop_word_list_longer = [el.strip() for el in stop_word_file_longer.readlines()]
        stop_word_file.close()
        stop_word_file_longer.close()

        # Get the list of 4-grams that occur at least twice in the data
        MAX_NGRAM = 4
        vectorizer_to_get_list_of_ngrams_to_use = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=2, max_df=0.2, ngram_range=(2, MAX_NGRAM), stop_words = stop_word_list)
        transformed_x_ngram = vectorizer_to_get_list_of_ngrams_to_use.fit_transform(data)
        feature_names_ngram = vectorizer_to_get_list_of_ngrams_to_use.get_feature_names()

        # Get the list of tokens that occur at least once in the data
        vectorizer_to_get_list_of_tokens_to_use = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=1, max_df=0.2, ngram_range=(1, 1), stop_words = stop_word_list)
        transformed_x_token =  vectorizer_to_get_list_of_tokens_to_use.fit_transform(data)
        feature_names_token =  vectorizer_to_get_list_of_tokens_to_use.get_feature_names()

        # Use the two create lists to create a list of the tokens and n-gram to include
        all_feature_names = feature_names_ngram + feature_names_token

        # Create the transformed data, when using the short stop word list
        self.vectorizer = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=1, max_df=0.2, ngram_range=(1, 4), stop_words = stop_word_list, vocabulary =  all_feature_names)
        transformed_x =  self.vectorizer.fit_transform(data)

        # Create the transformed data, when using the long stop word list
        training_data_vectorizer_ngram_more_stop_words = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=2, max_df=0.2, ngram_range=(2, MAX_NGRAM), stop_words = self.stop_word_list_longer)
        transformed_x_ngram_more_stop_words = training_data_vectorizer_ngram_more_stop_words.fit_transform(data)
        feature_names_ngram_more_stop_words  = training_data_vectorizer_ngram_more_stop_words.get_feature_names()
        vocabulary_more_stop_words = [el for el in feature_names_ngram_more_stop_words if el not in feature_names_ngram ]
        self.vectorizer_more_stop_words = TfidfVectorizer(use_idf=True, norm='l2', binary=False, sublinear_tf=True, \
                                                        min_df=2, max_df=0.2, ngram_range=(2, MAX_NGRAM), stop_words = self.stop_word_list_longer, vocabulary = vocabulary_more_stop_words)
        transformed_x_more_stop_words = self.vectorizer_more_stop_words.fit_transform(data)


        # Combine the two transformed vectors to one


        # Get a list of all feature names
        available_feature_names =  self.vectorizer.get_feature_names()
        available_feature_names_more_stop_words =  self.vectorizer_more_stop_words.get_feature_names()
        all_available_feature_names = available_feature_names + available_feature_names_more_stop_words #+ [str(el) for el in range(0, self.semantic_vector_length)]


        # Determine how many features to use

        nr_of_features_to_use = 10 * len(data) # Ten times the number of features as there is training data samples
        if nr_of_features_to_use >= len(available_feature_names) or not self.use_feature_select:
            nr_of_features_to_use = 'all'
            print("Use all features")

        train_data_combined =  self.combine_token_data(transformed_x, transformed_x_more_stop_words)

        # Select the best features from the transformed data
        # Will select all if nr_of_features_to_use = 'all'
        self.ch2 = SelectKBest(chi2, k=nr_of_features_to_use)
        X_train_selected = self.ch2.fit_transform(train_data_combined, targets)
        self.feature_names = [all_available_feature_names[i] for i
                         in self.ch2.get_support(indices=True)]
        print(len(self.feature_names))

        # Combine with semantic features
        # (will be set to 0 if self.use_word2vec = False)
        semantic_features = self.get_semantic_features(data)
        train_data_combined_with_semantic =  self.combined_with_semantic(X_train_selected, semantic_features)

        print("Vectorized data")
        return train_data_combined_with_semantic

    def transform(self, data):
        data = self.pre_process_data(data)

        # Create the transformed data, when using the short stop word list
        transformed_x =  self.vectorizer.transform(data)

        # Create the transformed data, when using the long stop word list
        transformed_x_more_stop_words = self.vectorizer_more_stop_words.transform(data)

        train_data_combined =  self.combine_token_data(transformed_x, transformed_x_more_stop_words)
        X_train_selected = self.ch2.transform(train_data_combined)

        # Combine with semantic features
        semantic_features = self.get_semantic_features(data)
        train_data_combined_with_semantic =  self.combined_with_semantic(X_train_selected, semantic_features)
        return train_data_combined_with_semantic

# Use_model is not tested
def use_model(data, datavectorizer, trained_model, target_names):
    print("Using model")
    X_selected = datavectorizer.transform(data)

    predicted_labels = trained_model.predict(X_selected)
    #print('predicted labels:',predicted_labels)

    for predicted, text in zip(predicted_labels, data):
        transformed = datavectorizer.vectorizer.transform([text])
        inversed = [el for el in datavectorizer.vectorizer.inverse_transform(transformed)[0] if el in datavectorizer.feature_names]
        transformed_more_stop_words = datavectorizer.vectorizer_more_stop_words.transform([text])
        inversed_more_stopwords = [el for el in datavectorizer.vectorizer_more_stop_words.inverse_transform(transformed_more_stop_words)[0] if el in datavectorizer.feature_names]
        #print('predicted int label:',predicted)
        output = ["Predicted:", target_names[predicted], "Text:", text, "", "", "", "Features:"] + inversed + inversed_more_stopwords
        #print("\t".join(output))

    return predicted_labels

def train_model(train_data, directory_name, cross_validate = True):
    print("Training model")
    result_file = "result_" + directory_name + ".txt"
    print(result_file)
    dv = DataVectorizer()
    X_train_selected = dv.fit_transform(train_data.data, train_data.target)

    # Parameter settings
    svc = LinearSVC(class_weight='balanced', random_state=0, penalty = "l1",  dual = False)
    K_FOLD = 30
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=0) # giving random_state should result in same shuffle every time
    f1_scorer = make_scorer(f1_score, average = "macro")

    if cross_validate:
        parameters={'C': [0.1, 1, 5, 10]}
        grid_search_clf = GridSearchCV(svc, parameters, cv=K_FOLD, n_jobs = 1, scoring = f1_scorer)
        grid_search_clf.fit(X_train_selected, train_data.target)
        best_estimator = grid_search_clf.best_estimator_
        print("Macro f1 cross validation", grid_search_clf.best_score_)
        print(best_estimator.get_params())
    else:
        print("No parameter setting")
        best_estimator = svc

    # Predict using the best paramter
    predicted_labels = cross_val_predict(best_estimator, X_train_selected, train_data.target, cv=skf, n_jobs=10, verbose=10,\
                                             fit_params=None, method='predict')

    # Print the result
    output = open(result_file, "w")
    for predicted, gold, text in zip(predicted_labels, train_data.target, train_data.data):
        transformed = dv.vectorizer.transform([text])
        inversed = [el for el in dv.vectorizer.inverse_transform(transformed)[0] if el in dv.feature_names]
        transformed_more_stop_words = dv.vectorizer_more_stop_words.transform([text])
        inversed_more_stopwords = [el for el in dv.vectorizer_more_stop_words.inverse_transform(transformed_more_stop_words)[0] if el in dv.feature_names]

        output.write("\t".join(["Predicted:", train_data.target_names[predicted], "Gold:", train_data.target_names[gold], "Text:", text, "Features:", str(inversed) + str(inversed_more_stopwords)]) + "\n")

    f1_micro =  f1_score(train_data.target, predicted_labels, average = "micro")
    print("f1_micro", f1_micro)
    output.write("f1_micro" + "\t" +  str(f1_micro) + "\n")
    f1_macro =  f1_score(train_data.target, predicted_labels, average = "macro")
    output.write("f1_macro" + "\t" +  str(f1_macro) + "\n")
    print("f1_macro", f1_macro)
    accuracy = accuracy_score(train_data.target, predicted_labels)
    output.write("accuracy" + "\t" +  str(accuracy) + "\n")
    print("accuracy", accuracy)

    cm = confusion_matrix(train_data.target, predicted_labels)
    print("confusion matrix")
    print(cm)
    output.write("confusion matrix\n")
    output.write(str(cm))
    output.write("\n")

    print("target names", train_data.target_names)
    output.write("target names\n")
    output.write(str(train_data.target_names))
    output.write("\n")
    output.close()

    pickle.dump(best_estimator,open('model.sav','wb'))
    return best_estimator



def run_experiment():

    if len(sys.argv) < 2:
        sys.exit("You need to give the directory name of the training data")
    debate_directory = sys.argv[1]
    #do_train =

    print("Extracting training data from {}, binning by stance...".format(debate_directory))
    lines = []
    import codecs
    with codecs.open(debate_directory+'/train.csv','r',encoding='utf-8',errors='ignore') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            tweet = row[0]
            target = row[1]
            stance = row[2]
            if target == "Climate Change is a Real Concern":
                lines.append((tweet, stance))
    texts_per_label = {}
    texts_per_label['for'] = [line[0] for line in lines if line[1] == 'FAVOR']
    texts_per_label['against'] = [line[0] for line in lines if line[1] == 'AGAINST']
    texts_per_label['neither'] = [line[0] for line in lines if line[1] == 'NEITHER']
    print("Number of for, against, and neither texts: {}, {}, {}".format(len(texts_per_label['for']),len(texts_per_label['against']),
    len(texts_per_label['neither'])))

    print("Writing to new directory: train_data")
    os.mkdir('train_data')
    for stance_label in ['for','against','neither']:
        os.mkdir('train_data/'+stance_label)
        for ix_t,text in enumerate(texts_per_label[stance_label]):
            with open('train_data/{}/text_no_{}.txt'.format(stance_label,ix_t),'w') as FileObj:
                FileObj.write(text)

    train_directory = 'train_data'

    print("Will evaluate files in the directory " + train_directory)

    train_data = \
       load_files(train_directory, encoding='utf-8', shuffle=True, random_state=0)
    model = train_model(train_data, str(debate_directory).replace("/","_").replace("\\","_"))
    #model = pickle.load(open('model.sav','rb'))

    dv = DataVectorizer()
    X_train_selected = dv.fit_transform(train_data.data, train_data.target)
    # test_data_file = sys.argv[2] # 'vaccines can be quite risky'
    # test_data_stance = test_data_file.split('/')[1].split('-')[0]
    # test_data_obj = open(test_data_file,'r')
    # test_data_lines = test_data_obj.readlines()
    # test_data_lines = [l.strip() for l in test_data_lines]
    # print(test_data_lines)
    # data = numpy.array(test_data_lines)
    # print('length of data:',len(data))
    # print('data:',data)
    # target_names = ['for','against','uncertain']
    # predicted_results = use_model(data, dv, model, target_names)
    # with open('test_data/{}-vax/test_predictions.txt'.format(test_data_stance),'w') as FileObj:
    #     for ix,p in enumerate(predicted_results):
    #         FileObj.write('{}\t{}\n'.format(p,data[ix]))
    # return predicted_results

run_experiment()
