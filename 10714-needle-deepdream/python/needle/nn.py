"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import needle.autograd as autograd
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    ### BEGIN YOUR SOLUTION
    children = []
    if isinstance(value, Module):
        children += [value]
        children += value._children()
    elif isinstance(value, dict):
        for k, v in value.items():
            children += _child_modules(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            children += _child_modules(v)
    return children
    ### END YOUR SOLUTION


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

#######################################
#       Deep Dream - Loss Function
#######################################

class DD_Loss(Module):

  def __init__(self):
    super().__init__()

  def forward(self, x):
    #Sum of the activations from the output of the layer of interest
    #Summing over all axes
    return x.sum()



class Flatten(Module):
    """
    Flattens the dimensions of a Tensor after the first into one dimension.

    Input shape: (bs, s_1, ..., s_n)
    Output shape: (bs, s_1*...*s_n)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x.reshape((x.shape[0], np.prod(x.shape[1:])))
        ### END YOUR SOLUTION


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.zeros((in_features, out_features)), device=device, dtype=dtype)
        a = (1/in_features)**0.5
        init.uniform(self.weight, low=-a, high=a)
        if bias:
            self.bias = Parameter(np.zeros(out_features), device=device, dtype=dtype)
            init.uniform(self.bias, low=-a, high=a)
        else:
            self.bias = Parameter(np.zeros(out_features), device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        cast_shape = np.array(x.shape)
        cast_shape[-1] = self.out_features
        return x @ self.weight + self.bias.broadcast_to(cast_shape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        return (-ops.logsoftmax(x) * ops.one_hot(y, num_classes=x.shape[1], device=x.device, dtype=x.dtype)).sum()/x.shape[0]
        ### END YOUR SOLUTION


class BatchNorm(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.running_mean = Tensor(np.zeros(dim), device=device, dtype=dtype, requires_grad=False)
        self.running_var = Tensor(np.ones(dim), device=device, dtype=dtype, requires_grad=False)
        self.weight = Parameter(np.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(np.zeros(dim), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        cast_shape = np.ones(len(x.shape), dtype=int)
        cast_shape[1] = self.dim
        if self.training:
            axes = tuple([i for i in range(len(x.shape)) if i!=1])
            n = int(np.prod(x.shape) / x.shape[1])
            mu = x.sum(axes=axes) / n
            self.running_mean.data = (1-self.momentum) * self.running_mean + self.momentum * mu.data
            x = x - mu.reshape(cast_shape).broadcast_to(x.shape)
            var = (x**2).sum(axes=axes) / n
            self.running_var.data = (1-self.momentum) * self.running_var + self.momentum * var.data *n/(n-1)
            sigma = ((var + self.eps)**0.5).reshape(cast_shape).broadcast_to(x.shape)
            return x * self.weight.reshape(cast_shape).broadcast_to(x.shape) / sigma + \
                self.bias.reshape(cast_shape).broadcast_to(x.shape)
        else:
            x = x.data - self.running_mean.reshape(cast_shape).broadcast_to(x.shape)
            sigma = ((self.running_var + self.eps)**0.5).reshape(cast_shape).broadcast_to(x.shape)
            return x * self.weight.data.reshape(cast_shape).broadcast_to(x.shape) / sigma + \
                self.bias.data.reshape(cast_shape).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm(Module):
    def __init__(self, dims, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else (dims,)
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(np.ones(dims), device=device, dtype=dtype)
        self.bias = Parameter(np.zeros(dims), device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        axes = tuple(np.arange(-1, -len(self.dims)-1, -1))
        cast_shape = np.array(x.shape)
        cast_shape[-len(axes):] = 1
        layer_mean = x.sum(axes=axes) / np.prod(self.dims)
        x = x - layer_mean.reshape(cast_shape).broadcast_to(x.shape)
        layer_var = (x**2).sum(axes=axes) / np.prod(self.dims)
        std = ((layer_var + self.eps)**0.5).reshape(cast_shape).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * x / std + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.p = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = ops.randb(x.shape, n=1, p=1-self.p, dtype=x.dtype)
            return x * mask / (1-self.p)
        else: return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION


class Identity(Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format

    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        ### BEGIN YOUR SOLUTION
        self.padding = int((self.kernel_size-1)/2)
        self.weight = Parameter(np.zeros((kernel_size, kernel_size, in_channels, out_channels)),
                                device=device, dtype=dtype)
        init.kaiming_uniform(self.weight)
        if bias:
            self.bias = Parameter(np.zeros(out_channels), device=device, dtype=dtype)
            a = 1.0/(in_channels * kernel_size**2)**0.5
            init.uniform(self.bias, low=-a, high=a)
        else:
            self.bias = Parameter(np.zeros(out_channels), device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1,2)).transpose((2,3))
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        cast_shape = np.ones(len(x.shape), dtype=int)
        cast_shape[-1] = self.out_channels
        x = x + self.bias.reshape(cast_shape).broadcast_to(x.shape)
        x = x.transpose((1,3)).transpose((2,3))
        return x
        ### END YOUR SOLUTION


class MaxPool(Module):
    """
    Multi-channel 2D max pooling layer
    Accepts inputs in NCHW format, outputs also in NCHW format
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return ops.max_pool(x, self.kernel_size, self.stride)


class AvgPool(Module):
    """
    Multi-channel 2D average pooling layer
    Accepts inputs in NCHW format, outputs also in NCHW format
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return ops.avg_pool(x, self.kernel_size, self.stride)


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        if nonlinearity not in ['tanh', 'relu']: raise ValueError(nonlinearity)
        self.W_ih = Parameter(np.zeros((input_size, hidden_size)), device=device, dtype=dtype)
        self.W_hh = Parameter(np.zeros((hidden_size, hidden_size)), device=device, dtype=dtype)
        a = (1/hidden_size) ** 0.5
        init.uniform(self.W_ih, low=-a, high=a)
        init.uniform(self.W_hh, low=-a, high=a)
        if bias:
            self.bias_ih = Parameter(np.zeros((hidden_size,)), device=device, dtype=dtype)
            self.bias_hh = Parameter(np.zeros((hidden_size,)), device=device, dtype=dtype)
            init.uniform(self.bias_ih, low=-a, high=a)
            init.uniform(self.bias_hh, low=-a, high=a)
        else:
            self.bias_ih = Parameter(np.zeros((hidden_size,)), device=device, dtype=dtype, requires_grad=False)
            self.bias_hh = Parameter(np.zeros((hidden_size,)), device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = Tensor(np.zeros((bs, self.hidden_size)), device=X.device, dtype=X.dtype, requires_grad=False)
        h_prime = getattr(ops, self.nonlinearity)(
            X @ self.W_ih + self.bias_ih.broadcast_to((bs, self.hidden_size)) + \
            h @ self.W_hh + self.bias_hh.broadcast_to((bs, self.hidden_size)))
        return h_prime
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias=bias, 
                                  nonlinearity=nonlinearity, 
                                  device=device, dtype=dtype)]
        for i in range(num_layers-1):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias=bias, 
                                  nonlinearity=nonlinearity, 
                                  device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h_n = []
        inputs = [X[i,:,:] for i in range(X.shape[0])]
        for i, ith_layer_rnn_cell in enumerate(self.rnn_cells):
            h = h0[i,:,:] if h0 is not None else None
            for j in range(len(inputs)):
                h = ith_layer_rnn_cell(inputs[j], h)
                inputs[j] = h
            h_n.append(h)
        out = ops.stack(inputs, axis=0)
        h_n = ops.stack(h_n, axis=0)
        return out, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.W_ih = Parameter(np.zeros((input_size, 4*hidden_size)), device=device, dtype=dtype)
        self.W_hh = Parameter(np.zeros((hidden_size, 4*hidden_size)), device=device, dtype=dtype)
        a = (1/hidden_size) ** 0.5
        init.uniform(self.W_ih, low=-a, high=a)
        init.uniform(self.W_hh, low=-a, high=a)
        if bias:
            self.bias_ih = Parameter(np.zeros((4*hidden_size,)), device=device, dtype=dtype)
            self.bias_hh = Parameter(np.zeros((4*hidden_size,)), device=device, dtype=dtype)
            init.uniform(self.bias_ih, low=-a, high=a)
            init.uniform(self.bias_hh, low=-a, high=a)
        else:
            self.bias_ih = Parameter(np.zeros((4*hidden_size,)), device=device, dtype=dtype, requires_grad=False)
            self.bias_hh = Parameter(np.zeros((4*hidden_size,)), device=device, dtype=dtype, requires_grad=False)
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        if h is None:
            h = (Tensor(np.zeros((bs, self.hidden_size)), device=X.device, dtype=X.dtype, requires_grad=False),) * 2
        h0, c0 = h
        tmp = X @ self.W_ih + self.bias_ih.broadcast_to((bs, 4*self.hidden_size)) + \
              h0 @ self.W_hh + self.bias_hh.broadcast_to((bs, 4*self.hidden_size))
        i = self.sigmoid(tmp[:, :self.hidden_size])
        f = self.sigmoid(tmp[:, self.hidden_size:2*self.hidden_size])
        g = self.tanh(tmp[:, 2*self.hidden_size:3*self.hidden_size])
        o = self.sigmoid(tmp[:, 3*self.hidden_size:])
        c_prime = f * c0 + i * g
        h_prime = o * self.tanh(c_prime)
        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias=bias, 
                                    device=device, dtype=dtype)]
        for i in range(num_layers-1):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias=bias, 
                                            device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        h_n, c_n = [], []
        inputs = [X[i,:,:] for i in range(X.shape[0])]
        for i, ith_layer_lstm_cell in enumerate(self.lstm_cells):
            h = (h0[0][i,:,:], h0[1][i,:,:]) if h0 is not None else None
            for j in range(len(inputs)):
                h = ith_layer_lstm_cell(inputs[j], h)
                inputs[j] = h[0]
            h_n.append(h[0])
            c_n.append(h[1])
        out = ops.stack(inputs, axis=0)
        h_n = ops.stack(h_n, axis=0)
        c_n = ops.stack(c_n, axis=0)
        return out, (h_n, c_n)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(np.zeros((num_embeddings, embedding_dim)), device=device, dtype=dtype)
        init.normal(self.weight, mean=0, std=1)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape[0], x.shape[1]
        x = x.reshape((seq_len * bs,))
        x = ops.one_hot(x, num_classes=self.num_embeddings, device=x.device, dtype=x.dtype)
        return (x @ self.weight).reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION
