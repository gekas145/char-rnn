import pickle
import torch
import config as c
from unidecode import unidecode
from torch import nn, optim
from torch.nn import functional as F
from sherlock_net import SherlockNet

def get_loss(X, y, h0, c0, nbatch):
    X = F.one_hot(X, num_classes=len(vocab)).type(torch.float)

    predicted, h0, c0 = net(X, h0, c0)
    predicted = torch.permute(predicted, (0, 2, 1))

    loss = criterion(predicted, y)

    return loss, h0.detach(), c0.detach()


def shuffle(X):
    return X[torch.randperm(X.shape[0]), :]


def get_zero_cell_state(nbatch):
    return torch.zeros((c.num_layers, nbatch, c.lstm_output_size)), torch.zeros((c.num_layers, nbatch, c.lstm_output_size))


with open("data/sherlock.txt", "r") as f:
    text = f.read()

text = unidecode(text)

vocab = list(set(text))
vocab.sort()
vocab = "".join(vocab)

vocab2idx = dict(zip(vocab, range(len(vocab))))

text = [vocab2idx[char] for char in text]

text = [text[i:i+c.main_seq_len+1] for i in range(0, len(text), c.main_seq_len)]
if len(text[-1]) < c.main_seq_len:
    text.pop(-1)

text = torch.tensor(text)
text = shuffle(text)
X_test, X_train = text[0:int(text.shape[0] * c.test_ratio), :], text[int(text.shape[0] * c.test_ratio):, :]
del text
print("Test shape: ", X_test.shape, "Train shape: ", X_train.shape)

net = SherlockNet(len(vocab), c.lstm_output_size, c.num_layers, c.dropout)
for param in net.parameters():
    param.register_hook(lambda grad: torch.clamp(grad, min=-c.clip_value, max=c.clip_value))

optimizer = optim.Adam(net.parameters())

scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, 
                                                lambda epoch: c.lr_decay if c.decay_after <= epoch <= c.stop_decay_after else 1.0)

criterion = nn.CrossEntropyLoss(reduction="mean")


for epoch in range(c.nepochs):
    X_train = shuffle(X_train)
    for i in range(0, X_train.shape[0], c.nbatch):
        X_batch = X_train[i:i + c.nbatch, :]
        h0, c0 = get_zero_cell_state(X_batch.shape[0])
        optimizer.zero_grad()
        for j in range(0, c.main_seq_len, c.sub_seq_len):

            loss, h0, c0 = get_loss(X_batch[:, j:j + c.sub_seq_len],
                                    X_batch[:, j + 1:j + c.sub_seq_len + 1],
                                    h0,
                                    c0,
                                    X_batch.shape[0])

            loss.backward()

        optimizer.step()

        iterations = epoch * (X_train.shape[0] // c.nbatch + int(X_train.shape[0] % c.nbatch > 0)) + i//c.nbatch
        if iterations % c.niter_info == 0:
            loss_test, loss_train = 0.0, 0.0

            with torch.inference_mode():
                for j in range(0, c.main_seq_len, c.sub_seq_len):

                    test, _, _ = get_loss(X_test[:, j:j + c.sub_seq_len],
                                          X_test[:, j + 1:j + c.sub_seq_len + 1],
                                          *get_zero_cell_state(X_test.shape[0]),
                                          X_test.shape[0])

                    train, _, _ = get_loss(X_train[:X_test.shape[0], j:j + c.sub_seq_len],
                                           X_train[:X_test.shape[0], j + 1:j + c.sub_seq_len + 1],
                                           *get_zero_cell_state(X_test.shape[0]),
                                           X_test.shape[0])
                    
                    loss_train += train
                    loss_test += test

            print(f"[iteration {iterations}] test loss {loss_test.data:.3f}, train loss {loss_train.data:.3f}")
            
            checkpoint_num = iterations//c.niter_info
            checkpoint_num = str(checkpoint_num) if checkpoint_num > 9 else f"0{checkpoint_num}"
            with open(f"checkpoints/{checkpoint_num}_checkpoint_test_loss_{loss_test.data:.3f}.pt", "wb") as f:
                pickle.dump(net.state_dict(), f)
    
    scheduler.step()








