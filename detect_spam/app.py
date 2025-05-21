#spam detection V1

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np
import os
import glob
from colorama import init, Fore, Style

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

LABELS = [
    'NOT SPAM',
    'SPAM',
]

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)

  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(net)
  return tf.keras.Model(text_input, net)

def predict_class(s):
  label = int(np.round(bert.predict([s], verbose=0))[0,0])
  return LABELS[label]

if __name__ == '__main__':
  init()
  print(Fore.YELLOW +"Initializing.....")
  print(Style.RESET_ALL)
  if not 'bert_model.h5' in glob.glob("*"):
        print(Fore.YELLOW +"Downloading Resources.....")
        print(Style.RESET_ALL)
        os.system("gdown https://drive.google.com/uc?id=1-3ANxu3RryYX5kqIkqACwOvOuwfkVCFN")
  bert = build_classifier_model()
  bert.load_weights('./bert_model.h5')
  print(Fore.GREEN +"Model is Ready")
  print(Style.RESET_ALL)
  

  while True:
    print("")
    text = input("Enter text : ")
    print(predict_class(text))
  

