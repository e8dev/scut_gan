#Author: Dmitrii Kuznetsov @e8dev
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import pandas as pd 
import random

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DKDataLoader(object):
    def __init__(self):
        #init
        self.total_words=0
        self.max_sequence_length = 0
        '''
        df_s = pd.read_csv('./data/prepared_sentences.csv')
        self.sentences=df_s['reviewText']
        
        df_reviews=pd.read_csv('./data/prepared_reviews.csv')
        self.review_scores=df_reviews['overall']
         '''
        self.tokenizer = Tokenizer()

    def one_time_aws_prepare(self,url):
        df = pd.read_json(url, lines=True)
        df['reviewText'].replace('', np.nan, inplace=True)
        df.head()
        df.info()
        sentences = df['reviewText']
        review_scores = df['overall']

        #removing empty
        #remove reviews shorter than 30 and longer than 100 words 
        
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
                if(i<30 or i>100):
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
        #input_sequences = []

        train_sequences = self.tokenizer.texts_to_sequences(self.sentences)
        print("train_sequences")
        #print(train_sequences)
        self.max_sequence_length = max([len(x) for x in train_sequences])
        sequences = np.array(pad_sequences(train_sequences, maxlen=self.max_sequence_length, padding='pre'))
        
        return sequences

    def custom_pad_sequences(self,train_sequences):
        input_sequences = np.array(pad_sequences(train_sequences, maxlen=self.max_sequence_length-1, padding='pre'))
        return input_sequences

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def ngram_data_prepare(self, sequences):
        x=[]
        y=[]
        for i in range(len(sequences)):
            temp_x,temp_y = self.split_input_target(sequences[i])
            x.append(temp_x)
            y.append(temp_y)

        ###WRITE TO FILE SEQUENCES(./data/...) will be much faster
        return x,y

    def data_prepare(self, input_sequences):
        # create predictors and label
        print("data_prepare")
        print(input_sequences.shape)
        xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
        print(labels.shape)
        print(xs[0])
        print(labels[0])
        ys = tf.keras.utils.to_categorical(labels, num_classes=self.total_words)
        print(xs.shape)
        print(ys.shape)
        
        return xs, ys


    def word_to_index(self,word):
        return self.tokenizer.word_index[word]

    def word_from_index(self,predicted_index):
        #predicted = np.argmax(ys[w_index])
        #print(predicted)
        for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
        return output_word

    def sentences_from_words():
        print()

    def generate_and_save_text(self, epoch, seed_text):
        with open('saved_generated_text/saved.txt', 'a') as the_file:
            the_file.write(seed_text+' ---epoch:'+str(epoch)+' \n')



        
#example usage dataloader class
data_loader = DKDataLoader()
data_loader.one_time_aws_prepare("./data/musical_instruments_5.json")
#data_loader.one_time_sequences_save()
#input_sequences = data_loader.tokenization()
#xs,ys = data_loader.data_prepare(input_sequences)
#data_loader.word_from_index(125)
#print(data_loader.total_words)


