import json
#import gzip
#import tqdm
import re
import os
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import one_hot
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
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

from mgan_1_data_loader import DataLoader


### PARAMETERS

batch_size=16
learning_rate=0.01

#dataloader
data_loader = DataLoader()
input_sequences = data_loader.tokenization()
xs,ys = data_loader.data_prepare(input_sequences)
total_words = data_loader.total_words
max_sequence_length = data_loader.max_sequence_length


def getBatch(X, Y):
    indices = np.arange(len(X))
    batch=[]
    while True:
            # it might be a good idea to shuffle your data before each epoch
            np.random.shuffle(indices)
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    return X[batch], Y[batch]
                    batch=[]



### MODEL

# Create the generator

class GeneratorModel(tf.keras.Model):

    def __init__(self, total_words, max_sequence_length):
        super().__init__()
        #self.generator = generator
        self.l1 = tf.keras.layers.Embedding(total_words, 80, input_length=max_sequence_length-1)
        self.l11 = LSTM(100, return_sequences=True)
        self.l12 = LSTM(50)
        self.l2 = tf.keras.layers.Dropout(0.1)
        self.l3 = Dense(total_words/20)
        self.l4 = Dense(total_words, activation='softmax')


    def call(self, x):
        x = self.l1(x)
        x = self.l11(x)
        x = self.l12(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        x, y = data
        print(x.shape)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            print(y)
            #print()
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            print("loss")
            print(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}




modelG = GeneratorModel(total_words, max_sequence_length)
modelG.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['accuracy'])
#y = modelG(tf.ones(shape=xn.shape))
modelG.fit(xs, ys, epochs=3)
