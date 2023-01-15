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
data_loader = DataLoader("./data/musical_instruments_5.json")
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
        self.l2 = tf.keras.layers.Dropout(0.1)
        self.l3 = Dense(total_words/20)
        self.l4 = Dense(total_words, activation='softmax')

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x

    def compile(self):
        super(GeneratorModel, self).compile()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


modelG = GeneratorModel()
modelG.compile()







'''


# Prepare the metrics.
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

#checkpoint_path = "training_1/cp-{epoch:02d}-{loss:.4f}.ckpt"
checkpoint_path = "training_manual/cp-{epoch:02d}.ckpt"
#checkpoint_path = F"/content/gdrive/My Drive/checkpoint.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,save_freq=100)


model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
              metrics=['accuracy'])
'''





#checkpoint_path_fixed = "training_1/cp-01.ckpt"
#model.load_weights(checkpoint_path_fixed)
#print("loaded")
#loss, acc = model.evaluate(xs[:200], ys[:200], verbose=2)
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#history = model.fit(xs, ys, epochs=20, verbose=1, callbacks=[cp_callback])







# Instantiate an optimizer.
#optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
#optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
# Instantiate a loss function.
#loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
#loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)



epochs = 10
n_chunk = len(xs) // batch_size
print(n_chunk)



for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    n = 0
    for b_iter in range(n_chunk):
    #for step, (x_batch_train, y_batch_train) in enumerate(train_padded):
    #   print(step)
    # x[n], y[n]
        xn, yn = getBatch(xs,ys)


        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as gen_tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(xn, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(yn, logits)
            print(loss_value)
            

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = gen_tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric.
        train_acc_metric.update_state(yn, logits)


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

    model.save(f'2dk-model-epoch-{epoch}')

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()





'''
@tf.function
def train_step(xs_batch):
    noise = tf.random.normal([batch_size, total_words])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

'''

