import theano.tensor as T
from unidecode import unidecode
from utils.network import NeuralNetwork
from utils.layers import FullyConnectedLayer, LSTMCell
from utils.losses import SparseCategoricalCrossentropy
from utils.embedding import OneHotEmbedding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def softmax(x):
    x_exp = np.exp(x)
    return x_exp/np.sum(x_exp)

def generate_text(phrase, n=2000, temp=1.0):
    encoded_phrase = np.array([[ohe._vocab[char]] for char in phrase])
    encoded_phrase = ohe(encoded_phrase)
    encoded_phrase = np.squeeze(encoded_phrase, axis=1)

    generated_text = net.predict(encoded_phrase, n=n, temp=temp)
    generated_text = [vocab[char_idx] for char_idx in generated_text]

    return phrase + "".join(generated_text)


df = pd.read_parquet("data/next-character.parquet")
vocab = list(set(unidecode("".join(df.context))))
vocab.sort()
print(len(vocab))
print(df.shape)

loss = SparseCategoricalCrossentropy()
ohe = OneHotEmbedding(vocab)

net = NeuralNetwork(len(vocab), loss, nepochs=10, nbatch=100, embedding=ohe)
net.add_layer(LSTMCell(512, 100))
net.add_layer(LSTMCell(512, 100))
net.add_layer(FullyConnectedLayer(len(vocab), softmax, seq2seq=True))

net.load("char-rnn.json")

net.inference_on()

phrase = "Mr. H"
temperature = 0.65
print(generate_text(phrase, temp=temperature))
