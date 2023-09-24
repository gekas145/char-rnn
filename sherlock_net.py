from torch import nn
from torch import distributions
from torch.nn import functional as F
import torch


class SherlockNet(nn.Module):

    def __init__(self, input_size, lstm_output_size, lstm_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(input_size, lstm_output_size,
                            num_layers=lstm_layers, dropout=dropout, batch_first=True)

        self.dropout = nn.Dropout(p=dropout)

        self.linear = nn.Linear(lstm_output_size, input_size)

    def forward(self, inputs, h0, c0):
        outputs, (hn, cn) = self.lstm(inputs, (h0, c0))

        outputs = self.dropout(outputs)

        return self.linear(outputs), hn, cn
    
    @torch.inference_mode()
    def generate_text(self, inputs, temp=1.0, n=2000):
        
        h0 = torch.zeros((self.lstm.num_layers, self.lstm.hidden_size))
        c0 = torch.zeros((self.lstm.num_layers, self.lstm.hidden_size))
        outputs, h0, c0 = self.forward(inputs, h0, c0)

        res = []

        for i in range(n):
            probs = F.softmax(outputs[-1, :]/temp, dim=0)

            idx = distributions.categorical.Categorical(probs).sample()

            res.append(idx.item())
            
            next_input = torch.zeros((1, self.lstm.input_size))
            next_input[0, idx] = 1.0

            outputs, h0, c0 = self.forward(next_input, h0, c0)

        return res









