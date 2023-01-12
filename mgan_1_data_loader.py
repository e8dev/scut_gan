#Author: Dmitrii Kuznetsov @e8dev
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
from sklearn.model_selection import train_test_split
import pandas as pd 


class DataLoader(object):
    def __init__(self):
        #init
        self.total_words=0
        self.max_sequence_length = 0
        
        df_s = pd.read_csv('./data/prepared_sentences.csv')
        self.sentences=df_s['reviewText']
        
        df_reviews=pd.read_csv('./data/prepared_reviews.csv')
        self.review_scores=df_reviews['overall']
        self.tokenizer = Tokenizer()

    def one_time_aws_prepare(self,url):
        df = pd.read_json(url, lines=True)
        df['reviewText'].replace('', np.nan, inplace=True)
        df.head()
        df.info()
        sentences = df['reviewText']
        #TODO: remove reviews longer 200 words
        review_scores = df['overall']
        #removing empty
        for index, row in sentences.items():
            if(isinstance(row, (float))):
                sentences.drop(index, inplace=True)
                review_scores.drop(index, inplace=True)
            else:
                #print(row)
                i=0
                for word in row.split():
                    #print(word)
                    i=i+1
                #print("number of words")
                #print(i)
                if(i<30 or i>300):
                    sentences.drop(index, inplace=True)
                    review_scores.drop(index, inplace=True)
            
        sentences.reset_index(drop=True)
        review_scores.reset_index(drop=True)
        #save to files
        sentences.to_csv('./data/prepared_sentences.csv',index=True) 
        review_scores.to_csv('./data/prepared_reviews.csv',index=True)  

        print("total examples--------")
        print(len(sentences))
        self.sentences=sentences
        self.review_scores=review_scores


    def tokenization(self):
        self.tokenizer.fit_on_texts(self.sentences)
        self.total_words = len(self.tokenizer.word_index) + 1
        input_sequences = []

        train_sequences = self.tokenizer.texts_to_sequences(self.sentences)
        self.max_sequence_length = max([len(x) for x in train_sequences])
        input_sequences = np.array(pad_sequences(train_sequences, maxlen=self.max_sequence_length, padding='pre'))
        return input_sequences


    def data_prepare(self, input_sequences):
        # create predictors and label
        xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=self.total_words)
        
        return xs, ys

    def word_from_index(self,w_index):
        predicted = np.argmax(ys[w_index])
        print(predicted)
        for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
        print(output_word)

        
#example usage dataloader class
#data_loader = DataLoader()
#input_sequences = data_loader.tokenization()
#xs,ys = data_loader.data_prepare(input_sequences)
#data_loader.word_from_index(125)
#print(data_loader.total_words)


