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


### PARAMETERS

batch_size=64
learning_rate=0.01

#DATA FOR DISCRIMINATOR
data_loader = DataLoader()
input_sequences = data_loader.tokenization()
#xs,ys = data_loader.data_prepare(input_sequences)
total_words = data_loader.total_words #63,303
max_sequence_length = data_loader.max_sequence_length #568
#total samples 100,563

xs = input_sequences
ys = tf.ones(shape=(input_sequences.shape[0],), dtype=tf.dtypes.int32)

# loop start in range(#total_samples):
#1. take random number of words in range 30 > x > 300
#2. convert them to words / string
#3. add to array
#4. apply tokenizator
#5. return sequences
# loop end

fake_x = tf.random.uniform(shape=xs.shape, minval=0, maxval=total_words, dtype=tf.dtypes.int32)
fake_y = tf.zeros(shape=(input_sequences.shape[0],), dtype=tf.dtypes.int32)


#xs_combined = tf.concat([xs,fake_x], axis=0)
#ys_combined = tf.concat([ys,fake_y], axis=0)

xs_combined = [] # this will be the list of processed tensors
ys_combined = [] # this will be the list of processed tensors
i=0
for t in xs:
    # do whatever
    #result_tensor = t + 1
    xs_combined.append(xs[i])
    xs_combined.append(fake_x[i])
    ys_combined.append(ys[i])
    ys_combined.append(fake_y[i])

    i=i+1



# Reserve 5,000 samples for validation.
x_val = xs_combined[-5000:]
y_val = ys_combined[-5000:]
x_train = xs_combined[:-5000]
y_train = ys_combined[:-5000]


# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size)


# Prepare the metrics.
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()

### MODEL

# Create the Discriminator
modelD = keras.Sequential(
    [
        #keras.Input(shape=(max_sequence_length-1,)),
        layers.Embedding(total_words, 80, input_length=max_sequence_length),
        layers.Bidirectional(tf.keras.layers.LSTM(64)),
        #layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_fn = tf.keras.losses.BinaryCrossentropy()

def train_discriminator(modelD):

    


    epochs = 5
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                print("x_batch_train")
                print(x_batch_train.shape)
                logits = modelD(x_batch_train, training=True)  # Logits for this minibatch

                print("predd")
                pred_abs = tf.squeeze(logits)
                print(pred_abs)

                print("yyyy")
                print(y_batch_train)

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(y_batch_train, pred_abs)
                print("loss")
                print(loss_value)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, modelD.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, modelD.trainable_weights))

            #predicted = np.argmax(model.predict(token_list), axis=-1)
            # Update training metric.
            train_acc_metric.update_state(y_batch_train, pred_abs)

            # Log every 200 batches.
            if step % 20 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))
                # Display metrics at the end of each epoch.
                train_acc = train_acc_metric.result()
                print("Training acc over step: %.4f" % (float(train_acc),))


        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = modelD(x_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))
        #modelD.save(f'2dk-discriminator-epoch-{epoch}')
        modelD.save_weights(f'2dk-discriminator-epoch-{epoch}')
        print("saved")


#train_discriminator(modelD)



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




#task 1 - with custom loop controll loss value. Later we use it to apply feedback score+++
#pre train generator to generate some samples. 
#task 2 - build discriminator+++
#train discriminator on real and generated data+++
#combine both components to GAN
#train gan, test errors
#apply score/rewards to discriminator
#train
