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

print(f"Initial after preprocesing: {len(amz_reviews_tokenized)}")
    

# REMOVE RARE WORDS

# Create a frequency dict of all tokens
freqs = {}
for text in amz_reviews_tokenized:
  for word in text:
    freqs[word] = freqs.get(word, 0) + 1 

# Removing all words that occurr less than 7 times
remove=False
cache_text = []
cache_score = []
for idx, text in enumerate(amz_reviews_tokenized):
#for text in amz_reviews_tokenized:
  for word in text:
    if freqs[word]<7:
      remove=True
  if remove == False:
    cache_text.append(text)
    cache_score.append(review_scores[idx])
    #print("--------")
    #print(text)
    #print(review_scores[idx])
    #print("--------")
  remove=False 
amz_reviews_tokenized = cache_text
amz_reviews_scores = cache_score

print()
print(f"Remaining texts after preprocesing: {len(amz_reviews_tokenized)}")

# Set a seed to make results comparable
#random.seed(69)
# Shuffle the dataset once, to obtain random train and test partitions later
#random.shuffle(amz_reviews_tokenized)

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

'''
print(amz_reviews_tokenized[0])
print(amz_reviews_scores[0])
print("test")
print(amz_reviews_tokenized[75])
print(amz_reviews_scores[75])
print("test")
'''


'''
#Word2Vec
word2vec_model = Word2Vec(sentences=word2vec_data, vector_size=200, window=5, min_count=1, workers=4, sg=1, negative=50, epochs = 50)
#Save the trained embeddings
word2vec_model.save(save_path + "/skip_gram_embedding_2.model")
'''

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


#print(embedding_matrix)



# Add a single row to shift the matrix to the right (since later we use 0 padding for our batches)
embedding_matrix = np.insert(arr=embedding_matrix, obj=0, values=np.zeros(200), axis=0)

embedding_matrix.shape

word2index_dict = {token: token_index for token_index, token in enumerate(word2vec_model.wv.index_to_key)}

text2index_input = []
text2index_target = []
text2index_teacher_forcing = []


# +1 to each index as we use zero paddding and masking (therefore zeros need to be ignored) 
for input, target, teacher in train_data:
  input = [word2index_dict[key]+1 for key in input]
  target = [word2index_dict[key]+1 for key in target]
  teacher = [word2index_dict[key]+1 for key in teacher]
  text2index_input.append(input)
  text2index_target.append(target)
  text2index_teacher_forcing.append(teacher)



# We split the data into train data (90%) and test data (10%)
# Ragged Tensors allow us to create tf.Datasets containing different sequence lengths
train_ragged_dataset_input = tf.ragged.constant(text2index_input[0:int(len(text2index_input)*0.9)])
train_ragged_dataset_target = tf.ragged.constant(text2index_target[0:int(len(text2index_target)*0.9)])
train_ragged_dataset_teacher = tf.ragged.constant(text2index_teacher_forcing[0:int(len(text2index_teacher_forcing)*0.9)])
#add scores
train_ragged_dataset_scores = tf.ragged.constant(amz_reviews_scores[0:int(len(amz_reviews_scores)*0.9)])


train_dataset_input = tf.data.Dataset.from_tensor_slices(train_ragged_dataset_input)
train_dataset_target = tf.data.Dataset.from_tensor_slices(train_ragged_dataset_target)
train_dataset_teacher = tf.data.Dataset.from_tensor_slices(train_ragged_dataset_teacher)
#add scores
train_dataset_scores = tf.data.Dataset.from_tensor_slices(train_ragged_dataset_scores)




# Convert ragged tensors to dense tensor in order to allow us to create padded batches
# See: https://github.com/tensorflow/tensorflow/issues/39163
train_dataset_input = train_dataset_input.map(lambda x: x)
train_dataset_target = train_dataset_target.map(lambda x: x)
train_dataset_teacher = train_dataset_teacher.map(lambda x: x)
train_dataset_scores = train_dataset_scores.map(lambda x: x)


train_dataset = tf.data.Dataset.zip((train_dataset_input, train_dataset_target, train_dataset_teacher, train_dataset_scores)).cache().shuffle(buffer_size=50000, reshuffle_each_iteration=True).padded_batch(50).prefetch(tf.data.experimental.AUTOTUNE)



# Repeat for test data
test_ragged_dataset_input = tf.ragged.constant(text2index_input[int(len(text2index_input)*0.9):len(text2index_input)])
test_ragged_dataset_target = tf.ragged.constant(text2index_target[int(len(text2index_target)*0.9):len(text2index_target)])
test_ragged_dataset_teacher = tf.ragged.constant(text2index_teacher_forcing[int(len(text2index_teacher_forcing)*0.9):len(text2index_teacher_forcing)])
#add scores
test_ragged_dataset_scores = tf.ragged.constant(amz_reviews_scores[int(len(amz_reviews_scores)*0.9):len(amz_reviews_scores)])


test_dataset_input = tf.data.Dataset.from_tensor_slices(test_ragged_dataset_input)
test_dataset_target = tf.data.Dataset.from_tensor_slices(test_ragged_dataset_target)
test_dataset_teacher = tf.data.Dataset.from_tensor_slices(test_ragged_dataset_teacher)
test_dataset_scores = tf.data.Dataset.from_tensor_slices(test_ragged_dataset_scores)

test_dataset_input = test_dataset_input.map(lambda x: x)
test_dataset_target = test_dataset_target.map(lambda x: x)
test_dataset_teacher = test_dataset_teacher.map(lambda x: x)
test_dataset_scores = test_dataset_scores.map(lambda x: x)

test_dataset = tf.data.Dataset.zip((test_dataset_input, test_dataset_target, test_dataset_teacher, test_dataset_scores)).cache().shuffle(buffer_size=10000, reshuffle_each_iteration=True).padded_batch(50).prefetch(tf.data.experimental.AUTOTUNE)

# PRE-TRAINING

amzAE = ae.AutoEncoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, bidirectional=False)
amzAE.compile()
#amzAE.load_weights(save_path + '/model_weights_ae/ae-epoch-last')
amzAE.load_weights(save_path + '/model_weights_ae/ae-epoch-23')


#ae_training.trainModel(model=amzAE, word2vec_model=word2vec_model, train_dataset=train_dataset, test_dataset=test_dataset, loss_function=tf.keras.losses.SparseCategoricalCrossentropy(), num_epochs=50)
#amzAE.save_weights(save_path + '/model_weights_ae/ae-epoch-last')



#input is a full text
#we send full text to Encoder (part of AE), to get such data to be able to feed Decoder
train_dataset_GAN = train_dataset_input
train_dataset_GAN = train_dataset_GAN.map(lambda x: tf.squeeze(amzAE.Encoder(tf.expand_dims(x, axis=0))))
#train_dataset_GAN = train_dataset_GAN.cache().batch(50).prefetch(tf.data.experimental.AUTOTUNE)

gan_dataset = tf.data.Dataset.zip((train_dataset_GAN, train_dataset_scores)).cache().batch(50).prefetch(tf.data.experimental.AUTOTUNE)





#GAN
LaTextGAN_Generator = latextgan.Generator()
LaTextGAN_Generator.compile()


eval.text_generator(generator=LaTextGAN_Generator, autoencoder=amzAE, word2vec_model=word2vec_model, num_texts=20)

LaTextGAN_Generator.load_weights(save_path + '/model_weights_latextgan/la-epoch-last')

LaTextGAN_Discriminator = latextgan.Discriminator()


'''
latextgan_training.train_GAN(
  generator=LaTextGAN_Generator, 
  discriminator=LaTextGAN_Discriminator, 
  autoencoder=amzAE, 
  word2vec_model=word2vec_model, 
  train_dataset_GAN=gan_dataset, 
  num_epochs=100
)

LaTextGAN_Generator.save_weights(save_path + "/model_weights_latextgan/la-epoch-last")


'''



print("success")


eval.text_generator(generator=LaTextGAN_Generator, autoencoder=amzAE, word2vec_model=word2vec_model, num_texts=20)

############ STOP ############
quit()
############ STOP ############