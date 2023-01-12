#Author: Dmitrii Kuznetsov @e8dev
import tensorflow as tf
from mgan_1_data_loader import DataLoader
from mgan_2_generator import Generator




#dataloader
data_loader = DataLoader("./data/musical_instruments_5.json")
input_sequences = data_loader.tokenization()
xs,ys = data_loader.data_prepare(input_sequences)

#generator
generator = Generator(data_loader.total_words, 128, data_loader.max_sequence_length, data_loader.tokenizer)
g_model = generator.model()
g_model_loaded = generator.model_with_loaded_weights(g_model, "./training_1/cp-05.ckpt", xs, ys)
#example
predicted_text = generator.predict_generate(g_model_loaded,"what you can recommend?")
#generated_image = we use it in descriminator
print(predicted_text)

#discriminator



#reward


#training
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

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

                

#train main
train(train_dataset, EPOCHS)