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


### PARAMETERS

batch_size=64
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
labels = df['overall']
#removing empty
for index, row in sentences.items():
    if(isinstance(row, (float))):
        sentences.drop(index, inplace=True)
        labels.drop(index, inplace=True)

sentences.reset_index(drop=True)
labels.reset_index(drop=True)

### DATA PEPARING
train_sentences, val_sentences, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2, random_state=0)

for w in train_sentences:
    if(isinstance(w, (float))):
        print(type(w))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_sentences)
vocab_size = len(tokenizer.word_counts)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(sentences)
#print(train_sequences)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type, value=0.0)

def generate_batch(batch_size, train_padded, word_index):
    n_chunk = len(train_padded) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = train_padded[start_index:end_index]
        length = max(map(len, batches))
        x_data = np.full((batch_size, length), 1, np.int32)
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches

x, y = generate_batch(batch_size, train_padded, word_index)

#print(x[0])
#print(y[0])

'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_reviews)
vocab_size = len(tokenizer.word_counts)
word_index = tokenizer.word_index
'''

#train_sequences = tokenizer.texts_to_sequences(train_sentences)
#train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#val_sequences = tokenizer.texts_to_sequences(val_sentences)
#val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(max_length, activation='relu')
])


# Instantiate an optimizer.
#optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
# Instantiate a loss function.
#loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()

epochs = 20
n_chunk = len(train_padded) // batch_size

'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
'''

'''
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    n = 0
    for batch in range(n_chunk):
    #for step, (x_batch_train, y_batch_train) in enumerate(train_padded):
    #   print(step)
    # x[n], y[n]

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model_lstm(x[n], training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y[n], logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model_lstm.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model_lstm.trainable_weights))
        n += 1
        # Log every 200 batches.
        if n % 100 == 0:
            total_loss = tf.reduce_mean(loss_value, name = 'total_loss')
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (n, float(loss_value))
            )
            print(total_loss)
            print("Seen so far: %s samples" % ((n + 1) * batch_size))

    model_lstm.save(f'model-epoch-{epoch}')

'''

filepath = "model-epoch-1"
loaded_model = tf.keras.models.load_model(
    filepath, custom_objects=None, compile=True, options=None
)

seed_text = "I understand that "
token_list = tokenizer.texts_to_sequences([seed_text])[0]
print("token_list")
print(token_list)
token_list_pad = pad_sequences([token_list], maxlen=120, padding='post')
predicted = np.argmax(loaded_model.predict(token_list_pad), axis=-1)
print(predicted)
output_word = ""
for word, index in tokenizer.word_index.items():
    if index == predicted:
        output_word = word
        break
seed_text += " " + output_word

print(seed_text)







'''
model_lstm.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
#model_lstm.summary()

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history_lstm = model_lstm.fit(train_padded,
                    validation_data=(val_padded), 
                    epochs=num_epochs, 
                    verbose=2,
                    callbacks=callbacks_list
                    )

'''
