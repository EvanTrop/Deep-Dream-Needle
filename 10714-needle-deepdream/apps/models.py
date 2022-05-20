import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)



#######################################
#          Deep Dream - Model
#######################################

class DD_Model(ndl.nn.Module):
    def __init__(self,preTrained_model: [nn.Module] , output_layer: nn.Module):
        super().__init__()
      
      #Don't think passing device and dtype is neccessary since pretrained and output
      #should already hold device/dtype values in their parameter tensors
        self.preTrained = preTrained_model
        self.output_layer = output_layer

    def forward(self,x):
        for module in self.preTrained:
          print(module)
          if module  == self.output_layer: #Final layer is the output layer of interest
            return module(x)
          else:
            x = module(x)

#######################################
#          Deep Dream - Model
#######################################


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = self._conv_bn(3, 16, 7, 4, device, dtype)
        self.conv2 = self._conv_bn(16, 32, 3, 2, device, dtype)
        self.res1 = nn.Residual(
            nn.Sequential(
                self._conv_bn(32, 32, 3, 1, device, dtype),
                self._conv_bn(32, 32, 3, 1, device, dtype)
            )
        )
        self.conv5 = self._conv_bn(32, 64, 3, 2, device, dtype)
        self.conv6 = self._conv_bn(64, 128, 3, 2, device, dtype)
        self.res2 = nn.Residual(
            nn.Sequential(
                self._conv_bn(128, 128, 3, 1, device, dtype),
                self._conv_bn(128, 128, 3, 1, device, dtype)
            )
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def _conv_bn(self, in_channels, out_channels, kernel_size, stride, device, dtype):
        return nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device, dtype=dtype),
            nn.BatchNorm(dim=out_channels, device=device, dtype=dtype),
            nn.ReLU()
        )

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.

        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size, device, dtype)
        self.seq_model_type = seq_model
        if seq_model == 'rnn':
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        else: raise ValueError(seq_model)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).

        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        out, h = self.seq_model(x, h)
        out = out.reshape((x.shape[0] * x.shape[1], self.hidden_size))
        out = self.linear(out)
        if h is not None:
            if self.seq_model_type == 'rnn': h = h.data
            if self.seq_model_type == 'lstm': h = (h[0].data, h[1].data)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)
