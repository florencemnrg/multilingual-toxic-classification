import torch
import pandas as pd
import numpy as np

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

xlmr = torch.hub.load('pytorch/fairseq', 'xlmr.large')
xlmr.eval()
def preprocess(text):
    tokens = xlmr.encode(text)
    if len(tokens) > 512:
        tokens = torch.cat((tokens[:511], torch.Tensor([2]).long()), 0)

    last_layer_features = xlmr.extract_features(tokens)
    return last_layer_features[:, 0, :].data.numpy()


import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_dim = 1024
        prob_dropout = 0.2
        output_size = 1

        self.dropout_1 = nn.Dropout(p=prob_dropout)
        self.out_proj = nn.Linear(input_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, debug=False):
        x = x.squeeze()

        if debug:
            print("Init ", x.shape)

        x = self.dropout_1(x)
        x = self.out_proj(x)

        if debug:
            print("out_proj ", x.shape)

        x = self.sig(x)

        if debug:
            print("sig ", x.shape)

        return x

def test(data):
    model = Net()

    if train_on_gpu:
        model.cuda()

    model.load_state_dict(torch.load('../model/model.pt'))
    model.eval()

    if train_on_gpu:
        data = data.cuda()

    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)

    # convert output probabilities to predicted class
    pred = torch.round(output.squeeze()).int()  # rounds to the nearest integer

    return pred.detach().numpy(), output.squeeze().detach().numpy()

if __name__ == "__main__":
    while(1):
        test_string = input("Masukkan teks: ")
        preprocessed = preprocess(test_string)
        y_pred, y_pred_proba = test(torch.from_numpy(preprocessed))
        if y_pred:
            print("Hasil: TOXIC! (proba: {})\n\n".format(y_pred_proba))
        else:
            print("Hasil: Tidak Toxic (proba: {})\n\n".format(y_pred_proba))
