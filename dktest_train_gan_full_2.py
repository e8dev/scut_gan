import json
#import gzip
#import tqdm
import re
import time
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


######### DATA




######### MODEL

### PARAMETERS

batch_size=64
learning_rate=0.01

### LAYERS

generator = keras.Sequential(
    [
        #keras.Input(shape=(max_sequence_length-1,)),
        layers.Embedding(total_words, 80, input_length=max_sequence_length-1),
        LSTM(100, return_sequences=True),
        LSTM(50),
        tf.keras.layers.Dropout(0.1),
        Dense(total_words/20),
        Dense(total_words, activation='softmax'),
    ],
    name="generator",
)

discriminator = keras.Sequential(
    [
        #keras.Input(shape=(max_sequence_length-1,)),
        layers.Embedding(total_words, 80, input_length=max_sequence_length-1),
        layers.Bidirectional(tf.keras.layers.LSTM(64)),
        #layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
)

### OPTIMIZERS & LOSS FUNCTIONS

optimizer_generator=keras.optimizers.Adam(learning_rate=0.0003)
optimizer_discriminator=keras.optimizers.Adam(learning_rate=0.0003)

loss_fn_generator=keras.losses.CategoricalCrossentropy()
loss_fn_discriminator=keras.losses.BinaryCrossentropy()

### TRAIN


def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = loss_fn_generator(fake_output)
      disc_loss = loss_fn_discriminator(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizer_generator.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizer_discriminator.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


def generate_sentence(length):
    for _ in range(length):
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

'''
modelD.load_weights('2dk-discriminator-epoch-2')
modelD.compile(loss=loss_fn, 
              optimizer=optimizer, 
              metrics=['accuracy'])
# Check its architecture
#modelD.summary()
batch_x, batch_y = next(iter(val_dataset))
loss, acc = modelD.evaluate(batch_x, batch_y, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))



seed_text = "kids drink to we or not how wife impossible burn error beach honestly long wood burn go run instrument does good not"
#next_words = []
token_list = data_loader.tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_length, padding='pre')
x_test = tf.random.uniform(shape=token_list.shape, minval=0, maxval=total_words, dtype=tf.dtypes.int32)
print("token_list")
print(token_list.shape)
print("fake")
print(x_test)


res = modelD.predict(x_test)
res2 = modelD.predict(token_list)
print(res)
print(res2)

'''

#task 1 - with custom loop controll loss value. Later we use it to apply feedback score+++
#pre train generator to generate some samples. +++
#task 2 - build discriminator+++
#train discriminator on real and generated data+++
#combine both components to GAN+++
#train gan, test errors
#apply score/rewards to discriminator
#train
