# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
import random
import time
import gensim
import matplotlib
matplotlib.use('macosx')


def timing(start):
  now = time.time()
  time_per_training_step = now - start
  return round(time_per_training_step, 4)
  
  
  
def visualization(word2vec_model, train_losses, test_losses, input_text, predicted_text, num_epochs, total_epochs): 

  print("Autoencoded Text (Training Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in input_text[0] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in tf.argmax(predicted_text[0], axis=2).numpy()[0] if i != 0])}")
  print()
  print("Autoencoded Text (Training Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in input_text[1] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in tf.argmax(predicted_text[1], axis=2).numpy()[0] if i != 0])}")
  print()
  
  print("Autoencoded Text (Test Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in input_text[2] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in tf.argmax(predicted_text[2], axis=2).numpy()[0] if i != 0])}")
  print()
  print("Autoencoded Text (Test Sample):")
  print(f"Input: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in input_text[3] if i != 0])}")
  print(f"Output: {' '.join([word2vec_model.wv.index_to_key[i-1] for i in tf.argmax(predicted_text[3], axis=2).numpy()[0] if i != 0])}")
  print()
  print()

  #set min max range
  minY = min([min(train_losses),min(test_losses)])
  minY = minY - minY*0.1
  maxY = max([max(train_losses),max(test_losses)])
  maxY = maxY + maxY*0.1

  if(num_epochs>=(total_epochs-1)):
    plt.style.use('ggplot')

    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
    ax1.set_ylim([minY,maxY])
    
    ax1.plot(train_losses, label='training')
    plt.pause(0.001)
    ax1.plot(test_losses, label='test')
    plt.pause(0.001)
    ax1.legend()
    plt.pause(0.001)
    #plt.ion()
    plt.show()


@tf.function(experimental_relax_shapes=True)
def train_step_ae(model, input, target, teacher, loss_function, optimizer):

  with tf.GradientTape() as tape:
    # 1.
    prediction = model(input, teacher)
    # 2.
    loss = loss_function(target, prediction)
    print("train step ae loss")
    print(loss)
    # 3.
    gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 
  return loss



def test_ae(model, test_data, loss_function):
  
  test_loss_aggregator = []
  
  for input, target, teacher, score in test_data:
    print("test ae")
    prediction = model(input, teacher)
    sample_test_loss = loss_function(target, prediction)
    print("test ae loss")
    print(sample_test_loss)
    test_loss_aggregator.append(sample_test_loss)
 
  test_loss = tf.reduce_mean(test_loss_aggregator)
  
  return test_loss



def trainModel(model, word2vec_model: gensim.models.word2vec.Word2Vec, train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, loss_function: tf.keras.losses, num_epochs: int=50, learning_rate: float=0.001, running_average_factor: float=0.95): 

  tf.keras.backend.clear_session()

  # Extract two fixed text
  for input, target, teacher, score in train_dataset.take(1):
    train_text_for_visualisation_1 = (input[0], teacher[0])
    train_text_for_visualisation_2 = (input[1], teacher[1])
    
  for input, target, teacher, score in test_dataset.take(1):
    test_text_for_visualisation_1 = (input[0], teacher[0])
    test_text_for_visualisation_2 = (input[1], teacher[1])
    
  # Initialize the optimizer: Adam with custom learning rate.
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  
  # Initialize lists for later visualization.
  train_losses = []
  test_losses = []
  
  # testing once before we begin on the test and train data
  '''
   test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
  test_losses.append(test_loss)

  train_loss = test_ae(model=model, test_data=train_dataset, loss_function=loss_function)
  train_losses.append(train_loss)
   
  
  '''

  


  for epoch in range(num_epochs):
    start = time.time()

    # Training and computing running average
    running_average = 0
    for input, target, teacher, score in train_dataset:
      train_loss = train_step_ae(model=model, input=input, target=target, teacher=teacher, loss_function=loss_function, optimizer=optimizer)
      print("train_loss")
      print(train_loss)
      running_average = running_average_factor * running_average  + (1 - running_average_factor) * train_loss
      #save every step
      model.save_weights(f'./_save/aws_music_tools/model_weights_ae/ae-epoch-{epoch}')
      print("model saved")
      
    
    train_losses.append(running_average)

    # Testing
    test_loss = test_ae(model=model, test_data=test_dataset, loss_function=loss_function)
    test_losses.append(test_loss)
    
    print(f"Epoch: {str(epoch+1)}")      
    print()
    print(f"This epoch took {timing(start)} seconds")
    print()
    print(f"Training loss for current epoch: {train_losses[-1]}")
    print()
    print(f"Test loss for current epoch: {test_losses[-1]}")
    #print()

    train_pred_text_1 = model(tf.expand_dims(train_text_for_visualisation_1[0], axis=0), tf.expand_dims(train_text_for_visualisation_1[1], axis=0))

    train_pred_text_2 = model(tf.expand_dims(train_text_for_visualisation_2[0], axis=0), tf.expand_dims(train_text_for_visualisation_2[1], axis=0))

    test_pred_text_1 = model(tf.expand_dims(test_text_for_visualisation_1[0], axis=0), tf.expand_dims(test_text_for_visualisation_1[1], axis=0), training=False)

    test_pred_text_2 = model(tf.expand_dims(test_text_for_visualisation_2[0], axis=0), tf.expand_dims(test_text_for_visualisation_2[1], axis=0), training=False)

    visualization(word2vec_model,
                  train_losses=train_losses, 
                  test_losses=test_losses, 
                  input_text=(train_text_for_visualisation_1[0],train_text_for_visualisation_2[0], test_text_for_visualisation_1[0], test_text_for_visualisation_2[0]), 
                  predicted_text=(train_pred_text_1, train_pred_text_2, test_pred_text_1, test_pred_text_2), 
                  num_epochs=epoch+1,
                  total_epochs=num_epochs
                  )

  


  #print()
  #model.summary()

