# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers



class Encoder(Model):
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, bidirectional: bool=True, embedding_size: int=200):
    super(Encoder, self).__init__()

    self.bidirectional = bidirectional
    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    if bidirectional:
        self.bi_lstm = tfkl.Bidirectional(tfkl.LSTM(units=100))
    else:
        self.lstm = tfkl.LSTM(units=100)
    self.dense = tfkl.Dense(600)



  @tf.function(experimental_relax_shapes=True)
  def call(self, x, training: bool=True):
    x = self.embedding(x)
    if self.bidirectional:
        hs = self.bi_lstm(x, training=training)
    else:
        hs = self.lstm(x, training=training)
    dense_out = self.dense(hs, training=training) 

    return dense_out



class Decoder(Model):
  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, embedding_size: int=200):

    super(Decoder, self).__init__()

    self.embedding = tfkl.Embedding(input_dim=vocab_size+1, output_dim=embedding_size, weights=[embedding_matrix], trainable=False, mask_zero=True)
    self.lstm = tfkl.LSTM(units=600, return_sequences=True, return_state=True)
    self.dense = tfkl.Dense(units=vocab_size+1, activation="softmax")

  @tf.function(experimental_relax_shapes=True)  
  def call(self, x, states, training: bool=True):
    x = self.embedding(x)
    hidden_states, _, _ = self.lstm(x, initial_state=[states, tf.zeros_like(states)], training=training)
    dense_out = self.dense(hidden_states, training=training)
    return dense_out



  def inference_mode(self, states, training: bool=True):
    #Call Decoder in inference mode: Reconstructing the input using only start token and embeddings.

    predictions = []
    start_token = self.embedding(tf.constant([[2]]))
    _, hs, cs = self.lstm(start_token, initial_state=[states, tf.zeros_like(states)], training=training)
    dense_out = self.dense(hs, training=training)
    pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1)
    predictions.append(pred)

    max_seq_length = 162
    end_token = 3
    stopping_criterion = False

    while not stopping_criterion:

      last_pred = self.embedding(tf.expand_dims(pred, axis=0))
      _, hs, cs = self.lstm(last_pred, initial_state=[hs, cs], training=training) 
      dense_out = self.dense(hs, training=training)
      pred = tf.argmax(dense_out, output_type=tf.int32,  axis=1) 
      predictions.append(pred)

      if pred  == end_token or len(predictions) >= max_seq_length:
        stopping_criterion=True

    return predictions



class AutoEncoder(Model):

  def __init__(self, vocab_size: int, embedding_matrix: np.ndarray, bidirectional: bool=True, embedding_size: int=200):

    super(AutoEncoder, self).__init__()

    self.Encoder = Encoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, bidirectional=bidirectional, embedding_size=embedding_size)
    self.Decoder = Decoder(vocab_size=vocab_size, embedding_matrix=embedding_matrix, embedding_size=embedding_size)


  @tf.function(experimental_relax_shapes=True)    
  def call(self, input, teacher, training: bool=True):
    hs = self.Encoder(input, training=training)
    predictions = self.Decoder(teacher, states=hs, training=training)
    return predictions