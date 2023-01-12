#Author: Dmitrii Kuznetsov @e8dev
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



class Discriminator(object):
    def __init__(self, total_words, batch_size,
                 max_sequence_length, tokenizer):

        self.tokenizer = tokenizer
        self.total_words = total_words
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.learning_rate = 0.01 #tf.Variable(float(learning_rate), trainable=False)


    def model(self):

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(self.total_words, 80, input_length=self.max_sequence_length-1))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1))

        model.compile(loss='binary_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
              metrics=['accuracy'])

        return model

    def discriminator_loss(real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def checkpoint(ckpt_path):
        #checkpoint_path = "training_1/cp-{epoch:02d}.ckpt"
        #checkpoint_dir = os.path.dirname(ckpt_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                        save_weights_only=True,
                                                        verbose=1,save_freq=100)
        return cp_callback

    def model_with_loaded_weights(self,model,ckpt_path,xs,ys):
        model.load_weights(ckpt_path)
        print("loaded")
        loss, acc = model.evaluate(xs[:200], ys[:200], verbose=2)
        print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
        print("Loss")
        print(loss)
        return model

    def train_auto(model,cp_callback,xs,ys,epochs):
        history = model.fit(xs, ys, epochs=epochs, verbose=1, callbacks=[cp_callback])

    def train_manually():
        print("")

    def predict_generate(self,model,seed_text):

        #seed_text = "what you can recommend?"
        #next_words = []
        #generated_text = ""

        for _ in range(10):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_length - 1, padding='pre')
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    #next_words = [""]
                    break
            seed_text += " " + output_word

        return seed_text





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

