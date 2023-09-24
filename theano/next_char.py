import theano.tensor as T
from unidecode import unidecode
from utils.network import NeuralNetwork
from utils.layers import FullyConnectedLayer, LSTMCell
from utils.losses import SparseCategoricalCrossentropy
from utils.embedding import OneHotEmbedding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_X_y(df):
    X = df.context.to_list()
    y = df.label.to_list()

    X = np.array([[ohe._vocab[char] for char in x] for x in X])
    y = np.array([[ohe._vocab[char] for char in yy] for yy in y])

    return X, y

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

ntest = int(0.05 * df.shape[0])
df_train, df_test = df.head(df.shape[0] - ntest), df.tail(ntest)

X_train, y_train = get_X_y(df_train)
X_test, y_test = get_X_y(df_test)

net = NeuralNetwork(len(vocab), loss, nepochs=10, nbatch=100, alpha=0.002, lmbda=0.001, embedding=ohe)
net.add_layer(LSTMCell(512, 100))
net.add_layer(LSTMCell(512, 100))
net.add_layer(FullyConnectedLayer(len(vocab), T.nnet.softmax, seq2seq=True))

net.compile()

loss_train, loss_test, metric_scores_train, metric_scores_test = net.train(X_train, y_train, X_test, y_test)

plt.plot(loss_train, label="train")
plt.plot(loss_test, label="test")
plt.xlabel("epoch")
plt.ylabel("sparse categorical crossentropy")
plt.title("next-character task")
plt.legend()
plt.show()

