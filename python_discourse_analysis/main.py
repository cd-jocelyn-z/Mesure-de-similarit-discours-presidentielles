# -------------------------- IMPORTS AND INITIAL SETUP -------------------------- #
import os
import os
import numpy as np
import spacy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
nlp = spacy.load("en_core_web_sm")

# ------------------------- LOAD CORPUS AND CREATE DICTIONARY ------------------------- #
def get_corpus_dict(corpus_path):
    corpus_dict = {}

    for file_name in os.listdir(corpus_path):
        file_path = os.path.join(corpus_path, file_name)
        with open(file_path, 'r') as file:
            file_content = file.read()
            corpus_dict[file_name] = file_content

    return corpus_dict

folder_path =  os.path.join("../", "US_Inaugural_Addresses")
corpus_dict = get_corpus_dict(folder_path)

# ------------------------- PREPROCESS CORPUS: TOKENIZATION AND CLEANING ------------------------- #
def preprocess_corpus(corpus_dict):

    for file_name, content in corpus_dict.items():
        doc = nlp(content)
        unwanted_tokens = {"<", ">", "br", "--", "'re", "'m", "'ve","n't", "br>--that"}
        
        tokenized = [
            token.text.lower() for token in doc 
            if not token.is_punct and not token.is_space and token.text.lower() not in unwanted_tokens
        ]
        corpus_dict[file_name] = tokenized

    return corpus_dict

preprocessed_dict = preprocess_corpus(corpus_dict)

# ------------------------- CREATE VOCABULARY SET ------------------------- #
def get_vocab_set(corpus):
    vocab = set()
    for word in corpus.values():
        vocab.update(word)
    return vocab 

corpus = preprocessed_dict
vocab_set = get_vocab_set(corpus)
vocabulary =list(vocab_set)

# ------------------------- FEATURE REPRESENTATION: BINARY VECTORS ------------------------- #
def get_feature_dict(corpus_dict):
    vocab_set = get_vocab_set(corpus_dict)
    w2i = {word: i for i, word in enumerate(vocab_set)}
    n = len(vocab_set)

    feature_dict = dict()
    for file_name, tokens in corpus_dict.items():
        vector = np.zeros(n, dtype=int)
        feature_dict[file_name] = vector

        for token in tokens:
            if token in w2i:
                idx = w2i[token]
                vector[idx] = 1
    return feature_dict

feature_dict = get_feature_dict(corpus)


# ------------------------- CALCULATE SPARSITY OF VECTORS ------------------------- #
def get_null_calcs(feature_dict):
    null_counts = []
    for vector in feature_dict.values():
        null_count = np.sum(vector == 0)
        null_counts.append(null_count)
    mean_null = np.mean(null_counts)
    min_null = np.min(null_counts)
    max_null = np.max(null_counts)
    return mean_null, min_null, max_null

mean_null, min_null, max_null = get_null_calcs(feature_dict)


# ------------------------- FIND TOP 3 CLOSEST SPEECHES ------------------------- #
class Document:
    def __init__(self, sparse_vector):
        self.n = sparse_vector.shape[0]
        self.non_null_components_dict = {idx: val for idx, val in enumerate(sparse_vector) if val != 0}

    def get_dot(self, other):
        doc_a = self
        doc_b = other
        dot_product = 0

        if doc_a.n != doc_b.n:
            raise ValueError("Les documents n'ont pas la même taille de vecteur et ne peuvent pas être comparés.")

        for idx, value in doc_a.non_null_components_dict.items():
            if idx in doc_b.non_null_components_dict:
                doc_a_component = value
                doc_b_component = doc_b.non_null_components_dict[idx]
                dot_product += doc_a_component * doc_b_component 

        return dot_product

document_objects = {file_name: Document(vector) for file_name, vector in feature_dict.items()}
obama_speech = document_objects["56_obama_2009.txt"]

similarities = {}
for file_name, doc in document_objects.items():
    if file_name != "56_obama_2009.txt":  # Skip Obama's own speech
        similarity = obama_speech.get_dot(doc)
        similarities[file_name] = similarity

top_3_closest = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]

# ------------------------- FIND COMMON WORDS BETWEEN DOCUMENTS ------------------------- #
def get_common_words(doc_a, doc_b):
    common_words = set(doc_a).intersection(set(doc_b))

    word_counts = dict()
    for comm_word in common_words:
        counter = 0
        for word in doc_b:
            if comm_word == word:
                counter+=1
        word_counts[comm_word] = counter

    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_word_counts

doc_a = preprocessed_dict.get("56_obama_2009.txt")
doc_b = preprocessed_dict.get("58_trump_2017.txt")

# ------------------------- TF-IDF CALCULATION SETUP ------------------------- #
class InverseDocumentFrequency:

    def __init__(self, directory_name):
        self.directory_name = directory_name


    def get_idf(self, word):
        corpus_path =  os.path.join("../", self.directory_name)
        total_docs = len(os.listdir(corpus_path))
        df= np.sum([1 for doc in preprocessed_dict.values() if word in doc])
        
        if df > 0:
            idf = np.log(total_docs / df)
            return idf

idf_calculator = InverseDocumentFrequency("US_Inaugural_Addresses")


# ------------------------- TF-IDF REPRESENTATION OF DOCUMENTS ------------------------- #
class DocumentV2:
    def __init__(self, feature_dict): 
        self.feature_dict = feature_dict

    def get_tf_idf_dict(self):
        idf_calculator = InverseDocumentFrequency("US_Inaugural_Addresses")
        tf_idf_dict = {}
        
        for file_name, vector in self.feature_dict.items():
            tf_idf_vector = np.zeros(vector.shape[0],dtype=float)

            for component_idx, component in enumerate(vector):
                word = list(vocab_set)[component_idx]

                if component == 1:         
                    tf = sum(1 for token in preprocessed_dict[file_name] if token == word)
                    idf = idf_calculator.get_idf(word)
                    tf_idf_vector[component_idx] = tf*idf
                else:
                    tf_idf_vector[component_idx] = 0.00
            
            tf_idf_dict[file_name] = tf_idf_vector

        return tf_idf_dict


    def get_topN(self, tf_idf_dict, topN):
        vocabulary = list(vocab_set)
        importance_dict = {}

        for file_name in sorted(tf_idf_dict.keys()):
            tfidf_vector = tf_idf_dict[file_name]
            importance_list = set()

            for score_idx, score in enumerate(tfidf_vector):
                word = vocabulary[score_idx]
                importance_list.add((word, score))
            
            topN_words = sorted(importance_list, key=lambda x: x[1], reverse=True)[:topN]
            importance_dict[file_name] = topN_words

            print(f"\nSpeech: {file_name}, Top Words: ")
            for idx, info_tuple in enumerate(topN_words):
                print(idx, info_tuple)
            

documents = DocumentV2(feature_dict)
tf_idf_dict = documents.get_tf_idf_dict()
documents.get_topN(tf_idf_dict, 10)


## ------------------------- IMPORTANCE OF SELECT WORDS ACROSS CORPUS ------------------------- #
target_words = ['government', 'borders', 'people', 'obama', 'war', 'honor','foreign', 'men', 'women', 'children']
importance_dict = {}

for file_name in sorted(tf_idf_dict.keys()):
    tfidf_vector = tf_idf_dict[file_name]
    importance_list = []

    for target_word in target_words:
        if target_word in vocabulary:
            score_idx = vocabulary.index(target_word)
            score = tfidf_vector[score_idx]
            info_tuple = (target_word, score)
            importance_list.append(info_tuple)
  
    importance_dict[file_name] = importance_list

    print(f"\nSpeech: {file_name}, Top Words: ")
    for idx, info_tuple in enumerate(importance_list):
        print(idx, info_tuple)
    