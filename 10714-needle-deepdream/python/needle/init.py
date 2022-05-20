import math
import needle as ndl
import numpy as np

def uniform(x, low=0.0, high=1.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ops.randu(x.shape, low=low, high=high, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION


def normal(x, mean=0.0, std=1.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ops.randn(x.shape, mean=mean, std=std, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION


def constant(x, c=0.0):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ops.ones_like(x) * c
    ### END YOUR SOLUTION


def ones(x):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ops.ones_like(x)
    ### END YOUR SOLUTION


def zeros(x):
    ### BEGIN YOUR SOLUTION
    x.data = ndl.ops.zeros_like(x)
    ### END YOUR SOLUTION


def _calculate_fans(x):
    ### BEGIN YOUR SOLUTION
    if len(x.shape) == 2: # init linear layer
        fan_in, fan_out = x.shape[0], x.shape[1]
    elif len(x.shape) == 4: # init conv layer
        k = x.shape[0]
        fan_in, fan_out = k*k*x.shape[2], k*k*x.shape[3]
    else:
        raise NotImplementedError()
    return (fan_in, fan_out)
    ### END YOUR SOLUTION


def xavier_uniform(x, gain=1.0):
    ### BEGIN YOUR SOLUTION
    a = gain * (6/sum(_calculate_fans(x)))**0.5
    x.data = ndl.ops.randu(x.shape, low=-a, high=a, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION


def xavier_normal(x, gain=1.0):
    ### BEGIN YOUR SOLUTION
    std = gain * (2/sum(_calculate_fans(x)))**0.5
    x.data = ndl.ops.randn(x.shape, mean=0, std=std, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION


def kaiming_uniform(x, mode="fan_in", nonlinearity="relu"):
    ### BEGIN YOUR SOLUTION
    if nonlinearity == 'relu': gain = 2**0.5
    else: raise NotImplementedError()
    fan_in, fan_out = _calculate_fans(x)
    if mode == "fan_in": fan = fan_in
    elif mode == "fan_out": fan = fan_out
    else: raise ValueError(mode)
    bound = gain * (3/fan)**0.5
    x.data = ndl.ops.randu(x.shape, low=-bound, high=bound, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION


def kaiming_normal(x, mode="fan_in", nonlinearity="relu"):
    ### BEGIN YOUR SOLUTION
    if nonlinearity == 'relu': gain = 2**0.5
    else: raise NotImplementedError()
    fan_in, fan_out = _calculate_fans(x)
    if mode == "fan_in": fan = fan_in
    elif mode == "fan_out": fan = fan_out
    else: raise ValueError(mode)
    std = gain / (fan)**0.5
    x.data = ndl.ops.randn(x.shape, mean=0, std=std, device=x.device, dtype=x.dtype)
    ### END YOUR SOLUTION
