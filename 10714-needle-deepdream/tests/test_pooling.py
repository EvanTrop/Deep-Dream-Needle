import sys
sys.path.append('./python')
import needle as ndl
from needle import nn, ops
import numpy as np
import torch
import itertools
np.random.seed(0)

KERNEL_SIZE = [1,2,3,4,5]
STRIDE = [1,2,3,4,5]

def test_max_pool():
    for k, s in itertools.product(KERNEL_SIZE, STRIDE):
        arr = np.random.rand(2,5,16,16)
        A = ndl.Tensor(arr)
        B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
        max_poolA = nn.MaxPool(kernel_size=k, stride=s)
        max_poolB = torch.nn.MaxPool2d(kernel_size=k, stride=s)
        A_ = max_poolA(A)
        A_.sum().backward()
        B_ = max_poolB(B)
        B_.sum().backward()
        assert (A_.numpy() == B_.detach().numpy()).all()
        assert (A.grad.numpy() == B.grad.numpy()).all()

def test_avg_pool():
    for k, s in itertools.product(KERNEL_SIZE, STRIDE):
        arr = np.random.rand(2,5,16,16)
        A = ndl.Tensor(arr)
        B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
        avg_poolA = nn.AvgPool(kernel_size=k, stride=s)
        avg_poolB = torch.nn.AvgPool2d(kernel_size=k, stride=s)
        A_ = avg_poolA(A)
        A_.sum().backward()
        B_ = avg_poolB(B)
        B_.sum().backward()
        np.testing.assert_allclose(A_.numpy(), B_.detach().numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(A.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)
