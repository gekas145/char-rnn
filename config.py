# Config file for SherlockNet training

# LSTM params
lstm_output_size = 512
num_layers = 2

# Training params
## regularisation
dropout = 0.5
## duration
nepochs = 30
## dataset
nbatch = 32
main_seq_len = 1000
sub_seq_len = 50
test_ratio = 0.15
## gradient
clip_value = 15
lr_decay = 0.8
decay_after = 3
stop_decay_after = 10
## logs
niter_info = 100





