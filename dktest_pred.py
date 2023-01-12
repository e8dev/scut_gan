import json
#import gzip
#import tqdm
import re
import os
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import pandas as pd 

from tensorflow.keras.callbacks import ModelCheckpoint

#tf.debugging.set_log_device_placement(True)

### PARAMETERS

batch_size=128
learning_rate=0.01

max_length = 120
padding_type = 'post'
trunc_type = 'post'
embedding_dim = 16
num_epochs = 10


url = "/Users/dk/WOOFOLDER/SCUT_study/Thesis/code/gandk/data/musical_instruments_5.json"
data_lines = []
data_reviews = []
data_scores = []



df = pd.read_json(url, lines=True)
df['reviewText'].replace('', np.nan, inplace=True)
df.head()
df.info()

sentences = df['reviewText']
review_scores = df['overall']
#removing empty
for index, row in sentences.items():
    if(isinstance(row, (float))):
        sentences.drop(index, inplace=True)
        review_scores.drop(index, inplace=True)

sentences.reset_index(drop=True)
review_scores.reset_index(drop=True)

### DATA PREPARING
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
total_words = len(tokenizer.word_index) + 1
print("total_words")
print(total_words) #5952
input_sequences = []
print(len(sentences))

train_sequences = tokenizer.texts_to_sequences(sentences)
'''
k=0
for line in sentences:
	token_list = tokenizer.texts_to_sequences([line])[0]
	#print(len(token_list))
	k=k+1
	if k % 10000 == 0:
		print(k)
    #print()
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)
'''
max_sequence_length = max([len(x) for x in train_sequences])
print("max---")
print(max_sequence_length)

input_sequences = np.array(pad_sequences(train_sequences, maxlen=max_sequence_length, padding='pre'))


# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
print("test")
#print(xs)
print(xs.shape)
print("lab")
#print(labels)
print(labels.shape)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)
#print(ys)
print(ys.shape)

print("test5")
predicted = np.argmax(ys[120])
print(predicted)
for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break

print(output_word)

model = Sequential()
model.add(tf.keras.layers.Embedding(total_words, 80, input_length=max_sequence_length-1))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50))
model.add(tf.keras.layers.Dropout(0.1))
model.add(Dense(total_words/20))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              metrics=['accuracy'])

print("compiled")

#loss, acc = model.evaluate(xs, ys, verbose=2, steps=10)
#print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))





checkpoint_path_fixed = "training_1/cp-05.ckpt"
model.load_weights(checkpoint_path_fixed)
print("loaded")
loss, acc = model.evaluate(xs[:200], ys[:200], verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))





'''
predict
'''
seed_text = "what you can recommend?"
#next_words = []

for _ in range(10):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            #next_words = [""]
            break
    seed_text += " " + output_word

print(seed_text)








