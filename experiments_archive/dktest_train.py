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



#checkpoint_path = "training_1/cp-{epoch:02d}-{loss:.4f}.ckpt"
checkpoint_path = "training_1/cp-{epoch:02d}.ckpt"
#checkpoint_path = F"/content/gdrive/My Drive/checkpoint.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_freq=100)
'''

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
'''


model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              metrics=['accuracy'])




checkpoint_path_fixed = "training_1/cp-01.ckpt"
model.load_weights(checkpoint_path_fixed)
print("loaded")
loss, acc = model.evaluate(xs[:200], ys[:200], verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


history = model.fit(xs, ys, epochs=20, verbose=1, callbacks=[cp_callback])





'''
predict


for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

'''










'''
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

print(x[0].shape)

print(y[0].shape)

#train_sequences = tokenizer.texts_to_sequences(train_sentences)
#train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

#val_sequences = tokenizer.texts_to_sequences(val_sentences)
#val_padded = pad_sequences(val_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


model_lstm = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length),

    #tf.keras.layers.LSTM(256, input_shape=(batch_size, max_length)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

    ##model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
    ##model.add(layers.Dense(len(chars), activation="softmax"))

    #tf.keras.layers.Dropout(.2),
    #tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(vocab_size, activation='relu')
])
'''


'''
# Instantiate an optimizer.
#optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
# Instantiate a loss function.
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

epochs = 30
n_chunk = len(train_padded) // batch_size

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
            print(loss_value)
            print("Seen so far: %s samples" % ((n + 1) * batch_size))

    model_lstm.save(f'2dk-model-epoch-{epoch}')

'''


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
