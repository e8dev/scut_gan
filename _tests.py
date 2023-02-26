### GAN with feedback score
### Author: Dmitrii Kuznetsov dk.scut@gmail.com
### Original LaTextGAN code forked from https://github.com/GerritBartels/LaTextGAN

import sys
import ae 
import ae_training as ae_training
import tensorflow as tf
import nltk
from tqdm import tqdm #show progress bar
import json
from nltk.tokenize import word_tokenize
import re 
import random
import numpy as np
import pandas as pd

import gensim.downloader as api
from gensim.models import Word2Vec

import standard_latextgan as latextgan
import standard_latextgan_training as latextgan_training
import standard_latextgan_evaluation as eval 


# DATA
print("GAN with feedback score")
dataset_prefix = "aws_music_tools"
save_path = (f"./_save/{dataset_prefix}")


df_s = pd.read_csv('./data/prepared_sentences.csv')
sentences=df_s['reviewText']

df_reviews=pd.read_csv('./data/prepared_reviews.csv')
review_scores=df_reviews['overall']

# Removing/Replacing special characters etc.
sentences_clean = [re.sub(r'\(cont\)|[\'’"]|http\S+|\n', '',text.replace("\'", "'").replace("&amp", "and")) for text in sentences]

sentences_clean = [re.sub(r'\.\.+|—+|-+|\*\*+', ' ',text) for text in sentences_clean]

sentences_clean = [re.sub(r'\d+\b', ' <NUM> ', text) for text in sentences_clean]


amz_reviews_tokenized = [word_tokenize(text.lower()) for text in sentences_clean]

print('`logits111` has type {0}'.format(type(amz_reviews_tokenized)))

print(amz_reviews_tokenized[22])
print()
print(f"Initial after preprocesing: {len(amz_reviews_tokenized)}")
    

# REMOVE RARE WORDS

# Create a frequency dict of all tokens
freqs = {}
for text in amz_reviews_tokenized:
  for word in text:
    freqs[word] = freqs.get(word, 0) + 1 

# Removing all words that occurr less than 7 times
remove=False
cache = []
for text in amz_reviews_tokenized:
  for word in text:
    if freqs[word]<7:
      remove=True
  if remove == False:
    cache.append(text)
  remove=False 
amz_reviews_tokenized = cache

print()
print(f"Remaining texts after preprocesing: {len(amz_reviews_tokenized)}")
     





# Set a seed to make results comparable
random.seed(69)
# Shuffle the dataset once, to obtain random train and test partitions later
random.shuffle(amz_reviews_tokenized)

# Add start and end of sequence token to every text
# and create the two datasets
train_data = []
word2vec_data = []

for text in amz_reviews_tokenized:

    text.insert(len(text), "<End>")
    text.insert(0, "<Start>")

    train_data.append((text, text[1:], text[:-1]))
    word2vec_data.append(text)


max_length = 0
idx = 0
for text in amz_reviews_tokenized:
  if len(text) > max_length:
    max_length = len(text)

print(f"Longest text has {max_length} tokens.")  


# Load previously saved embeddings
word2vec_model = Word2Vec.load(save_path + "/skip_gram_embedding_2.model")
words = list(word2vec_model.wv.index_to_key)
#print(words)
vocab_size = len(words)
print(f"Vocab size of our word2vec model: {vocab_size}")

embedding_matrix = np.zeros((len(words), 200))
for i in range(len(words)):
    embedding_vector = word2vec_model.wv[word2vec_model.wv.index_to_key[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Add a single row to shift the matrix to the right (since later we use 0 padding for our batches)
embedding_matrix = np.insert(arr=embedding_matrix, obj=0, values=np.zeros(200), axis=0)
word2index_dict = {token: token_index for token_index, token in enumerate(word2vec_model.wv.index_to_key)}

# PRE-TRAINING

amzAE = ae.AutoEncoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, bidirectional=False)
amzAE.compile()
amzAE.load_weights(save_path + '/model_weights_ae/ae-epoch-23')

LaTextGAN_Generator = latextgan.Generator()
LaTextGAN_Generator.compile()

#eval.text_generator(generator=LaTextGAN_Generator, autoencoder=amzAE, word2vec_model=word2vec_model, num_texts=20)

LaTextGAN_Generator.load_weights(save_path + '/model_weights_latextgan/la-epoch-last')





##### TESTING

eval.bleu4_score(LaTextGAN_Generator, amzAE, word2vec_model, amz_reviews_tokenized[0:100], 100)





#eval.text_generator(generator=LaTextGAN_Generator, autoencoder=amzAE, word2vec_model=word2vec_model, num_texts=20)
############ STOP ############
quit()
############ STOP ############