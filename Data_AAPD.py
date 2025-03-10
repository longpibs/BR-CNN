# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:01:28 2023

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
from sklearn.datasets import fetch_rcv1
import csv
        



train_data = list()
with open("text_train") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        train_data.append(row[0])

test_data = list()
with open("text_test") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        test_data.append(row[0])
        

train_label = list()
with open("label_train") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        temp = row[0].split(' ')
        train_label.append(temp)

test_label = list()
with open("label_test") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        temp = row[0].split(' ')
        test_label.append(temp)

        
        
        
from sklearn.preprocessing import MultiLabelBinarizer
        
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_label)
y_test = mlb.transform(test_label)
dict_class_rev = {v: k for k, v in mlb._cached_dict.items()}
        
        
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer_train = Tokenizer(oov_token = "<unk>")
tokenizer_train.fit_on_texts(train_data)
sequences_text_train = tokenizer_train.texts_to_sequences(train_data)
sequences_text_test = tokenizer_train.texts_to_sequences(test_data)
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
        word = splitLines[0]
        try:
          wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        except:
          print(splitLines[1:])
          print(len(splitLines[1:]))
          break
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
label_dict_save.to_csv('data_AAPD_dict.csv')

with h5py.File('data_AAPD.h5',"w") as f:
    
    f["w2v"] = embedding_matrix
    f['train'] = x_train
    f['train_label'] = y_train
    f['test'] = x_test
    f['test_label'] = y_test

