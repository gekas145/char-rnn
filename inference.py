from torch import nn
from torch import distributions
from torch.nn import functional as F
from sherlock_net import SherlockNet
from unidecode import unidecode
import config as c
import torch
import pickle

with open("data/sherlock.txt", "r") as f:
    vocab = f.read()

vocab = unidecode(vocab)

vocab = list(set(vocab))
vocab.sort()
vocab = "".join(vocab)

vocab2idx = dict(zip(vocab, range(len(vocab))))

with open("checkpoints/sherlock_net.pt", "rb") as f:
    state_dict = pickle.load(f)

net = SherlockNet(len(vocab), c.lstm_output_size, c.num_layers, c.dropout)
net.load_state_dict(state_dict)
net.eval()



init_text = "Mr. H"

inputs = torch.tensor([vocab2idx[chr] for chr in init_text])

inputs = F.one_hot(inputs, num_classes=len(vocab)).type(torch.float)

res = net.generate_text(inputs, temp=1.1)

text = "".join([vocab[idx] for idx in res])

print(init_text + text)








