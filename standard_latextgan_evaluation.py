# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import gensim 
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction 

# Imports for latent space analysis
from bokeh.models import Title
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Colorblind3                  # Color palette
from bokeh.models import ColumnDataSource               # Allows creating a column dataset for convenient plotting
from bokeh.transform import factor_cmap                 # To apply color palette to our 2 classes


from transformers import AutoTokenizer
from bert_score import score


def text_generator(generator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, num_texts: int=1):
  """Function that generates a given amount of texts.
  
  Arguments:
    generator (Generator): Generator class instance
    autoencoder (Autoencoder): Autoencoder class instance
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    num_texts (int): Number of texts that should be generated
  """

  print("Generated:")
  print()
  average_length = 0
  for _ in range(num_texts):
    noise = tf.random.normal([1, 100])

    states=generator(noise)

    ae_out = autoencoder.Decoder.inference_mode(states=states, training=False)

    tokens = [word2vec_model.wv.index_to_key[i.numpy()[0] -1] for i in ae_out if i.numpy()[0] != 0]
    average_length += len(tokens)
    print(f"{' '.join(tokens)}")
    print()
  print()
  print(f"Average length of generated texts: {average_length/num_texts} tokens")
  print()



def bleu4_score(generator, autoencoder, word2vec_model: gensim.models.word2vec.Word2Vec, reference_data, num_texts: int=1):
  """Function that calculates the bleu4 score for a given amount of generated texts.
  
  Arguments:
    generator (Generator): Generator class instance
    autoencoder (Autoencoder): Autoencoder class instance
    word2vec_model (gensim.models.word2vec.Word2Vec): Pretrained word2vec model
    reference_data (list): List containing the reference data for the bleu score computation
    num_texts (int): Number of texts that should be generated
  """
  
  generated_text = []
  for _ in range(num_texts):
    noise = tf.random.normal([1, 100])
    generated_text.append([word2vec_model.wv.index_to_key[i.numpy()[0] -1] for i in autoencoder.Decoder.inference_mode(states=generator(noise), training=False) if i.numpy()[0] != 0])
  
  
  hyp = generated_text


  # sentence_bleu([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS
  #score_bleu = sentence_bleu([reference_data], hyp[0], weights=(.25, .25, .25, .25), smoothing_function=None)

  for i, gen_text in enumerate(generated_text):
    real = reference_data[i]
    real = real[1:-1]

    bl_real = [real]
    bl_gen = [gen_text]
    print("cycle")
    print(bl_real)
    print(bl_gen)

    score_bleu = corpus_bleu([bl_real], bl_gen, weights=(.25, .25, .25, .25), smoothing_function=SmoothingFunction(epsilon=1. / 10).method1)
    print("score_bleu")
    print(score_bleu)

    generated_str = ' '.join(gen_text)
    real_str = ' '.join(real)
    print("cycle")
    print(generated_str)
    print(real_str)

    P, R, F1 = score([generated_str], [real_str], lang='en', verbose=True)
    print("SCORE BERT")
    print(P, R, F1)

  
  
  return score_bleu

def latent_space_analysis(generator, autoencoder, train_dataset: tf.data.Dataset, name: str):
  """Plot 2D TSNE Embedding of Generator against Encoder.

  Arguments:
    generator (Generator): Generator class instance
    autoencoder (AutoEncoder): AutoEncoder class instance
    train_dataset (tf.data.Dataset): Dataset to be fed into the Encoder for latent space analysis 
    name (str): Used for in the title of the plot
  """

  # Create a list of real text encodings from Encoder
  train_texts_embeddings = [autoencoder.Encoder(text, training=False) for text, _, _ in train_dataset.take(500)]
  train_texts_embeddings = [text for text_batch in train_texts_embeddings for text in text_batch]

  # Create a list of fake text encodings from Generator  
  generator_texts_embeddings=[]
  for _ in range(500):
    noise = tf.random.normal([50, 100])
    generator_texts_embeddings.append(generator(noise))
  generator_texts_embeddings = [text for text_batch in generator_texts_embeddings for text in text_batch]


  pca = PCA(n_components=50, svd_solver="randomized", random_state=0)
  pca_embedding_enc = pca.fit_transform(train_texts_embeddings)
  print('Cumulative explained variation for enocder embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
  pca_embedding_gen = pca.fit_transform(generator_texts_embeddings)
  print('Cumulative explained variation for generator embedding: {}'.format(np.sum(pca.explained_variance_ratio_)))
  
  # We apply the TSNE algorithm from scikit to get a 2D embedding of our latent space
  # Once for the Encoder
  tsne = TSNE(n_components=2, perplexity=30., random_state=0)
  tsne_embedding_enc = tsne.fit_transform(pca_embedding_enc)
  
  # Once for the Generator
  tsne_embedding_gen = tsne.fit_transform(pca_embedding_gen)

  # Plotting the TSNE embeddings
  labels =  ["Encoder" for _ in range(len(train_texts_embeddings))]
  labels.extend(["Generator" for _ in range(len(generator_texts_embeddings))])

  p = figure(tools="pan,wheel_zoom,reset,save",
            toolbar_location="above",
            title=f"2D Encoder and Generator Embeddings.")
  p.title.text_font_size = "25px"
  p.add_layout(Title(text=name, text_font_size="15px"), 'above')

  x1=np.concatenate((tsne_embedding_enc[:,0], tsne_embedding_gen[:,0]))
  x2=np.concatenate((tsne_embedding_enc[:,1], tsne_embedding_gen[:,1]))

  # Create column dataset from the tsne embedding and labels
  source = ColumnDataSource(data=dict(x1=x1,
                                      x2=x2,
                                      names=labels))

  # Create a scatter plot from the column dataset above
  p.scatter(x="x1", y="x2", size=1, source=source, fill_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", "Generator"]), fill_alpha=0.3, line_color=factor_cmap('names', palette=Colorblind3, factors=["Encoder", "Generator"]), legend_field='names')  

  show(p)


