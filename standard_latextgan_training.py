# -*- coding: utf-8 -*-
#%matplotlib inline
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
tfkl = tf.keras.layers
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
plt.style.use('ggplot') # Change the style of the plots to a nicer theme
import random
import time
#From IPython.display we import clear_output() in order to be able to clear the print statements after each epoch
from IPython.display import clear_output
from tqdm import tqdm, tqdm_notebook # Show progress bar
import gensim
import sys
import matplotlib
matplotlib.use('macosx')

def timing(start):
  """Function to time the duration of each epoch

  Arguments:
    start (time): Start time needed for computation 
  
  Returns:
    time_per_training_step (time): Rounded time in seconds 
  """
  now = time.time()
  time_per_training_step = now - start
  return round(time_per_training_step, 4)



def discriminator_loss(real_text, fake_text, data_scores):
  """
  Calculate the Wasserstein loss for the discriminator but swapping the sign in order to apply gradient descent.
  Returns:
    x (tensor): Wasserstein Loss
  """
  real_text_sq = tf.squeeze(real_text)
  scores_f32 = tf.cast(data_scores, tf.float32)
  #convert scores to fractions, 5 is 1.0 (100%)
  scores_fraction = scores_f32 * 1/5

  tf.print("real_text.shape")

  tf.print(data_scores)
  tf.print( real_text_sq )
  tf.print( real_text_sq*(1/scores_fraction) )

  #ADD HERE REVIEW SCORE IF IT'S 0.8 WE MULTIPLY LABEL 1X0.8

  loss_real = - tf.reduce_mean(real_text)
  loss_real_with_score = - tf.reduce_mean(real_text_sq*(1/scores_fraction))

  tf.print("loss real")
  tf.print(loss_real)
  tf.print("loss_real_with_score")
  tf.print(loss_real_with_score)

  loss_fake = tf.reduce_mean(fake_text)
  result_loss_1 = loss_real + loss_fake
  result_loss_2 = loss_real_with_score + loss_fake

  tf.print("TOTAL 1-2")
  tf.print(result_loss_1)
  tf.print(result_loss_2)


  return result_loss_2



def generator_loss(fake_text):
  """Calculate the Wasserstein loss for the generator.

  Arguments:
    fake_text (tensor): Linear output from discriminator

  Returns:
    x (tensor): Wasserstein Loss
  """

  loss_fake = - tf.reduce_mean(fake_text)

  return loss_fake
  
  
  
@tf.function() 
def gradient_penalty(discriminator, real_text, generated_text):
  # Due to the stacked approach we chose for the Autoencoder we had to alter the gradient
  # penalty by interpolating twice and calculating an average penalty. 
  alpha = tf.random.uniform(shape=[real_text.shape[0], 1], minval=0, maxval=1)

  interpolate = alpha*real_text + (1-alpha)*generated_text

  output = discriminator(interpolate)

  gradients = tf.gradients(output, interpolate)

  print("gradients")
  print(gradients)

  gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients)))

  penalty = 10*tf.reduce_mean((gradient_norm-1.)**2)

  return penalty

@tf.function() 
def generator_penalty(generator,fixed_noise):
  rand_noise = tf.random.normal([1, 100])
  out_fixed=generator(fixed_noise)
  out_rand=generator(rand_noise)
  diff_expected = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(fixed_noise-rand_noise))))
  diff_real = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(out_fixed-out_rand))))
  tf.print("diff_real-diff_expected:", diff_real-diff_expected)
  penalty = 10*tf.sqrt(tf.square(diff_real-diff_expected))

  tf.print("GEN PENALTY:", penalty)
  

  return penalty




def visualize_GAN(train_losses_generator, train_losses_discriminator, num_epochs, total_epochs):

  minY = min([min(train_losses_generator),min(train_losses_discriminator)]) - 100
  maxY = max([max(train_losses_generator),max(train_losses_discriminator)]) + 100

  
  if(num_epochs>=(total_epochs-1)):
    plt.style.use('ggplot')
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize = (10, 6))
    ax1.set(ylabel='Loss', xlabel='Epochs', title=f'Average loss over {num_epochs} epochs')
    ax1.set_ylim([minY,maxY])
    
    ax1.plot(train_losses_generator, label='Generator')
    plt.pause(0.001)
    ax1.plot(train_losses_discriminator, label='Discriminator')
    plt.pause(0.001)
    ax1.legend()
    plt.pause(0.001)
    
    #plt.ion()
    plt.show()


@tf.function()  
def train_step_GAN(generator, discriminator, train_data, optimizer_generator, optimizer_discriminator, train_generator, fixed_noise, data_scores):
  # 1.
  noise = tf.random.normal([train_data.shape[0], 100])

  #print("train step start")

  # Two Gradient Tapes, one for the Discriminator and one for the Generator 
  with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
      # 2.
      generated_text = generator(noise)

      # 3.
      real = discriminator(train_data)
      #print("D(x)")
      #print(real)
      fake = discriminator(generated_text)
      #print("D(z)")
      #print(fake)

      # 4.
      loss_from_generator = generator_loss(fake)
      #gen_penalty = generator_penalty(generator,fixed_noise)
      #loss_from_generator = loss_from_generator
      #print("generator_penalty")
      #print(gen_penalty)
      # Add gradient penalty to enforce lipschitz continuity
      d_loss = discriminator_loss(real, fake, data_scores)
      #print("loss D total")
      #print(d_loss)
      gp = gradient_penalty(discriminator=discriminator, real_text=train_data, generated_text=generated_text)
      #print("GP")
      #print(gp)
      loss_from_discriminator = d_loss + gp
      #print("loss_from_discriminator + gp")
      #print(loss_from_discriminator)
      #tf.print("tensors:", loss_from_discriminator, output_stream=sys.stdout)

  # 5.
  gradients_from_discriminator = discriminator_tape.gradient(loss_from_discriminator, discriminator.trainable_variables)
  #print(gradients_from_discriminator)
  #tf.print("gradients disc:", gradients_from_discriminator, output_stream=sys.stdout)
  optimizer_discriminator.apply_gradients(zip(gradients_from_discriminator, discriminator.trainable_variables))

  # We update the generator once for ten updates to the discriminator
  if train_generator:
    #print("print gen")
    gradients_from_generator = generator_tape.gradient(loss_from_generator, generator.trainable_variables)
    #print(gradients_from_generator)
    tf.print("gradients disc:")
    optimizer_generator.apply_gradients(zip(gradients_from_generator, generator.trainable_variables))

  #print("train step end")

  return loss_from_generator, loss_from_discriminator
  


def train_GAN(generator, discriminator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, train_dataset_GAN: tf.data.Dataset, num_epochs: int=150, running_average_factor: float=0.95, learning_rate: float=0.0001):

  tf.keras.backend.clear_session()

  # Two optimizers one for the generator and of for the discriminator
  optimizer_generator=tf.keras.optimizers.Adam(learning_rate=0.00001)
  optimizer_discriminator=tf.keras.optimizers.Adam(learning_rate=0.00001)

  # Fixed, random vectors for visualization
  fixed_generator_input_1 = tf.random.normal([1, 100])
  #fixed_generator_input_2 = tf.random.normal([1, 100])

  # Initialize lists for later visualization.
  train_losses_generator = []
  train_losses_discriminator = []

  train_generator = False

  for epoch in range(num_epochs):

    start = time.time()
    running_average_gen = 0
    running_average_disc = 0

    #with tqdm(total=519) as pbar:
    batch_no = 0
    for input, data_scores in train_dataset_GAN:
    #for batch_no, input, data_scores in enumerate(train_dataset_GAN):

      print("scores gan")
      print(data_scores)

      # Boolean used to train the discriminator 10x more often than the generator
      train_generator = False
      if batch_no % 20 == 0:
        train_generator = True
        print("TRAIN GENERATOR")

      gen_loss, disc_loss = train_step_GAN(
        generator, 
        discriminator, 
        train_data=input, 
        optimizer_generator=optimizer_generator, 
        optimizer_discriminator=optimizer_discriminator, 
        train_generator=train_generator, 
        fixed_noise=fixed_generator_input_1, 
        data_scores=data_scores
      )
      
      running_average_gen = running_average_factor * running_average_gen + (1 - running_average_factor) * gen_loss
      running_average_disc = running_average_factor * running_average_disc + (1 - running_average_factor) * disc_loss

      print("gen_loss")
      print(gen_loss)
      print("disc_loss")
      print(disc_loss)

      #increase batch counter
      batch_no = batch_no+1

    train_losses_generator.append(float(running_average_gen))
    train_losses_discriminator.append(float(running_average_disc))

    #clear_output()
    print(f'Epoch: {epoch+1}')      
    print()
    print(f'This epoch took {timing(start)} seconds')
    print()
    print(f'The current generator loss: {round(train_losses_generator[-1], 4)}')
    print()
    print(f'The current discriminator loss: {round(train_losses_discriminator[-1], 4)}')
    print()

    visualize_GAN(train_losses_generator=train_losses_generator, 
                  train_losses_discriminator=train_losses_discriminator, 
                  num_epochs=epoch+1,
                  total_epochs=num_epochs
                  )