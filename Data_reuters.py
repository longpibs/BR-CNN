# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 17:02:42 2023

@author: longp
"""

import datetime
import re,string
import nltk
from nltk.corpus import reuters
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import LazyCorpusLoader, CategorizedPlaintextCorpusReader
from collections import Counter
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer,\
    CountVectorizer, HashingVectorizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import multiprocessing


reuters = LazyCorpusLoader('reuters', CategorizedPlaintextCorpusReader, 
                           '(training|test).*', cat_file='cats.txt', encoding='ISO-8859-2',
                          nltk_data_subdir='corpora/')
# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/
reuters.words()

train_docs = list(filter(lambda doc: doc.startswith("train"),
                        reuters.fileids()))
test_docs = list(filter(lambda doc: doc.startswith("test"),
                        reuters.fileids()))

train_documents, train_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('training/')])
test_documents, test_categories = zip(*[(reuters.raw(i), reuters.categories(i)) for i in reuters.fileids() if i.startswith('test/')])





from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_categories)
y_test = mlb.transform(test_categories)
dict_class_rev = {v: k for k, v in mlb._cached_dict.items()}
test_dis_y= np.sum(y_train,axis =0)





stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
    



def preprocessingText(text, stop=stop):
  text = text.lower() #text to lowercase
  text = re.sub(r'<', '', text) #remove '<' tag
  text = re.sub(r'<.*?>', '', text) #remove html
  text = re.sub(r'[0-9]+', '', text) #remove number
  text = " ".join([word for word in text.split() if word not in stop]) #remove stopwords
  text = re.sub(r'[^\w\s]', '', text) #remove punctiation
  text = re.sub(r'[^\x00-\x7f]', '', text) #remove non ASCII strings
  for c in ['\r', '\n', '\t'] :
    text = re.sub(c, ' ', text) #replace newline and tab with tabs
  text = re.sub('\s+', ' ', text) #replace multiple spaces with one space
  text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
  return text

preprocessed_text_train = [preprocessingText(text) for text in train_documents]
preprocessed_text_test = [preprocessingText(text) for text in test_documents]





from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer_train = Tokenizer(oov_token = "<unk>")
tokenizer_train.fit_on_texts(preprocessed_text_train)
sequences_text_train = tokenizer_train.texts_to_sequences(preprocessed_text_train)
sequences_text_test = tokenizer_train.texts_to_sequences(preprocessed_text_test)
lens_train = [len(x) for x in sequences_text_train]
lens_test = [len(x) for x in sequences_text_test]
lens_all = lens_train + lens_test
max_lens_all = max(lens_all)
x_train = pad_sequences(sequences_text_train, maxlen=max_lens_all,padding = 'post',truncating = 'post')
x_test = pad_sequences(sequences_text_test, maxlen=max_lens_all,padding = 'post',truncating = 'post')







def loadWRVModel(File):
    print("Loading Word Representation Vector Model")
    f = open(File,'r',encoding='utf-8')
    WRVModel = {}
    for line in f:
        splitLines = line.split()
        if len(splitLines) > 301:
            print("word of multiple character->" + ' '.join(splitLines[0:-300]))
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[-300:]])
        WRVModel[word] = wordEmbedding
    print(len(WRVModel)," words loaded!")
    return WRVModel

WRVModel = loadWRVModel('./glove.6B.300d.txt')
VOCAB_SIZE = len(tokenizer_train.word_index) + 1

embedding_matrix = torch.zeros((VOCAB_SIZE,300))


unk = 0
for i in range(1, VOCAB_SIZE):
  word = tokenizer_train.index_word[i]
  if word in WRVModel.keys():
    embedding_matrix[i] = torch.from_numpy(WRVModel[word]).float()
  else:
    unk +=1
print('VOCAB_SIZE : {}'.format(VOCAB_SIZE))
print('TOTAL OF UNKNOWN WORD : {}'.format(unk))
embedding_matrix = embedding_matrix.numpy()
embedding_matrix[0,:] = 0
embedding_matrix[1,:] = 0



label_dict_save = pd.DataFrame.from_dict(dict_class_rev,orient='index')
label_dict_save.to_csv('data_reuter_dict.csv')

with h5py.File('data_reuters.h5',"w") as f:

    
    f["w2v"] = embedding_matrix
    f['train'] = x_train
    f['train_label'] = y_train
    f['test'] = x_test
    f['test_label'] = y_test