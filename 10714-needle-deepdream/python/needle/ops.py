"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
import numpy as np
from .autograd import Op, Tensor, Value, Tuple
from .device import default_device

OP_TABLE = {}


def register_op(name: str, op: Op) -> Op:
    """Register an operator to the op table.

    Parameters
    ----------
    name : str
        The name of the op.

    Returns
    -------
    op : Op
        The registered op.
    """
    if name in OP_TABLE:
        raise ValueError("Op %s is already registered")
    OP_TABLE[name] = op
    return op


def register_op_attr(op_name, attr_name, attr_value=None):
    """Register additional attributes to an existing op by name.


    Parameters
    ----------
    op_name : str
        The name of the op

    attr_name : str
        The name of the attribute

    attr_value :
        The attribute value to be set.

    Returns
    -------
    The attr_value if attr_value is not None.
    Otherwise returns a decorator function.


    Note
    ----
    This function can be used to register additional attributes
    to an Op used by a specific backend.
    """

    def _register(value):
        if op_name not in OP_TABLE:
            raise ValueError("Op %s does not exist")
        op = OP_TABLE[op_name]
        setattr(op, attr_name, value)
        return op

    if attr_value is None:
        return _register
    return _register(attr_value)


class MakeTupleOp(Op):
    def __call__(self, *args: List[Value]) -> Tuple:
        return Tuple.make_from_op(self, list(args))

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, Tuple)
        return [out_grad[i] for i in range(len(out_grad))]


make_tuple = register_op("MakeTuple", MakeTupleOp())


class TupleGetItemOp(Op):
    def __call__(self, a: Tuple, index: int, *, fold_const=True) -> Tensor:
        assert isinstance(a, Tuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTupleOp):
            return a.inputs[index]
        return Tensor.make_from_op(self, [a], attrs={"index": index})

    def gradient(self, out_grad, node):
        index = node.attrs["index"]
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return [make_tuple(*in_grad)]


tuple_get_item = register_op("TupleGetItem", TupleGetItemOp())


class FusedAddScalarsOp(Op):
    def __call__(self, a: Tensor, c0: float, c1: float) -> Tuple:
        return Tuple.make_from_op(self, [a], attrs={"c0": c0, "c1": c1})

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


fused_add_scalars = register_op("FusedAddScalars", FusedAddScalarsOp())


class EWiseAddOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        return [out_grad, out_grad]


add = register_op("EWiseAdd", EWiseAddOp())


class AddScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad]


add_scalar = register_op("AddScalar", AddScalarOp())


class EWiseMulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs)


multiply = register_op("EWiseMul", EWiseMulOp())


class MulScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        return [out_grad * node.attrs["scalar"]]


multiply_scalar = register_op("MulScalar", MulScalarOp())


class PowerScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        scalar = node.attrs["scalar"]
        return [out_grad * scalar * node.inputs[0] ** (scalar-1)]
        ### END YOUR SOLUTION


power_scalar = register_op("PowerScalar", PowerScalarOp())


class EWiseDivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, -out_grad*lhs/(rhs*rhs))
        ### END YOUR SOLUTION


divide = register_op("EWiseDiv", EWiseDivOp())


class DivScalarOp(Op):
    def __call__(self, a: Tensor, scalar: Number) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"scalar": scalar})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad / node.attrs["scalar"]]
        ### END YOUR SOLUTION


divide_scalar = register_op("DivScalar", DivScalarOp())


class MatMulOp(Op):
    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = out_grad @ rhs.transpose(None)
        rgrad = lhs.transpose(None) @ out_grad
        if len(lgrad.shape) > len(lhs.shape):
            lgrad = lgrad.sum(axes=tuple(range(len(lgrad.shape) - len(lhs.shape))))
        if len(rgrad.shape) > len(rhs.shape):
            rgrad = rgrad.sum(axes=tuple(range(len(rgrad.shape) - len(rhs.shape))))
        return lgrad, rgrad
        ### END YOUR SOLUTION


matmul = register_op("MatMul", MatMulOp())


class SummationOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if len(out_grad.shape) == 0:
            return [out_grad.broadcast_to(node.inputs[0].shape)]
        shape = np.array(node.inputs[0].shape)
        shape[np.array(node.attrs["axes"])] = 1
        return [out_grad.reshape(shape).broadcast_to(node.inputs[0].shape)]
        ### END YOUR SOLUTION


summation = register_op("Summation", SummationOp())


class BroadcastToOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape, out_shape = node.inputs[0].shape, out_grad.shape
        if len(in_shape) == 0: 
            return [out_grad.sum()]
        cast_shape = np.ones_like(out_shape)
        cast_shape[-len(in_shape):] = in_shape
        axes = tuple(np.where(cast_shape==1)[0])
        return [out_grad.sum(axes=axes).reshape(in_shape)]
        ### END YOUR SOLUTION


broadcast_to = register_op("BroadcastTo", BroadcastToOp())


class ReshapeOp(Op):
    def __call__(self, a: Tensor, shape: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"shape": shape})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad.reshape(node.inputs[0].shape)]
        ### END YOUR SOLUTION


reshape = register_op("Reshape", ReshapeOp())


class NegateOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [-out_grad]
        ### END YOUR SOLUTION


negate = register_op("Negate", NegateOp())


class TransposeOp(Op):
    def __call__(self, a: Tensor, axes: Optional[tuple] = None) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"axes": axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad.transpose(node.attrs["axes"])]
        ### END YOUR SOLUTION


transpose = register_op("Transpose", TransposeOp())


class LogOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [out_grad / node.inputs[0]]
        ### END YOUR SOLUTION


log = register_op("Log", LogOp())


class ExpOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        return [exp(node.inputs[0]) * out_grad]


exp = register_op("Exp", ExpOp())


class ReLUOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [Tensor(node.inputs[0].numpy() > 0, device=out_grad.device, dtype=out_grad.dtype) * out_grad]
        ### END YOUR SOLUTION


relu = register_op("ReLU", ReLUOp())


class LogSoftmaxOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        cast_shape = np.array(node.shape)
        cast_shape[-1] = 1
        return [(out_grad.sum(axes=len(out_grad.shape)-1).reshape(cast_shape).broadcast_to(node.shape) * (-exp(node))) + out_grad]
        ### END YOUR SOLUTION


logsoftmax = register_op("LogSoftmax", LogSoftmaxOp())


class TanhOp(Op):
    def __call__(self, a: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a])

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [(-node ** 2 + 1) * out_grad]
        ### END YOUR SOLUTION


tanh = register_op("Tanh", TanhOp())


class GetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = zeros_like(node.inputs[0], device=node.inputs[0].device)
        grad[node.attrs["idxs"]] = out_grad
        return [grad]
        ### END YOUR SOLUTION

get_item = register_op("GetItem", GetItemOp())


class SetItemOp(Op):
    def __call__(self, a: Tensor, idxs: Tuple, b: Tensor) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={"idxs": idxs})

    def gradient(self, out_grad, node):
        raise NotImplementedError()

set_item = register_op("SetItem", SetItemOp())


class StackOp(Op):
    def __call__(self, args: List[Value], axis: int) -> Tensor:
        return Tensor.make_from_op(self, args, attrs={'axis': axis})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axis = node.attrs["axis"]
        slices = [slice(None, None, None)] * len(out_grad.shape)
        grads = []
        for i in range(out_grad.shape[axis]):
            slices[axis] = i
            grads.append(out_grad[tuple(slices)])
        assert grads[0].shape == node.inputs[0].shape
        return grads
        ### END YOUR SOLUTION

stack = register_op("Stack", StackOp())


class ConvOp(Op):
    def __call__(self, a: Tensor, b: Tensor, stride: Optional[int] = 1, padding: Optional[int] = 0) -> Tensor:
        return Tensor.make_from_op(self, [a, b], attrs={'stride': stride, 'padding': padding})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, weight = node.inputs[0], node.inputs[1]
        k, _, _, _ = weight.shape
        s, p = node.attrs["stride"], node.attrs["padding"]
        out_grad = dilate(out_grad, s-1, axes=(1,2))
        gradX = conv(out_grad, flip(weight.transpose(), axes=(0,1)), stride=1, padding=k-p-1)
        gradW = conv(X.transpose((0,3)), out_grad.transpose((0,1)).transpose((1,2)), stride=1, padding=p)
        gradW = gradW.transpose((0,1)).transpose((1,2))
        return (gradX, gradW)
        ### END YOUR SOLUTION

conv = register_op("Conv", ConvOp())


class MaxPoolOp(Op):
    def __call__(self, a: Tensor, kernel_size: Optional[int] = 2, stride: Optional[int] = 2) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'kernel_size': kernel_size, 'stride': stride})

    def gradient(self, out_grad, node):
        k, s = node.attrs['kernel_size'], node.attrs['stride']
        arr = node.inputs[0]
        N, C, _, _ = arr.shape
        grad = np.zeros(arr.shape, dtype=arr.dtype)
        for i in range(out_grad.shape[2]):
            for j in range(out_grad.shape[3]):
                grad_block = np.zeros((N*C, k*k), dtype=arr.dtype)
                arr_block = arr[:,:,s*i:s*i+k,s*j:s*j+k].numpy().reshape(-1,k*k)
                mask = arr_block.argmax(-1)
                grad_block[np.arange(N*C), mask] = out_grad[:,:,i,j].numpy().flatten()
                grad[:,:,s*i:s*i+k,s*j:s*j+k] += grad_block.reshape((N,C,k,k))
        grad = Tensor(grad, device=arr.device, dtype=arr.dtype, requires_grad=False)
        return [grad]

max_pool = register_op("MaxPool", MaxPoolOp())


class AvgPoolOp(Op):
    def __call__(self, a: Tensor, kernel_size: Optional[int] = 2, stride: Optional[int] = 2) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'kernel_size': kernel_size, 'stride': stride})

    def gradient(self, out_grad, node):
        k, s = node.attrs['kernel_size'], node.attrs['stride']
        arr = node.inputs[0]
        N, C, _, _ = arr.shape
        grad = np.zeros(arr.shape, dtype=arr.dtype)
        for i in range(out_grad.shape[2]):
            for j in range(out_grad.shape[3]):
                grad[:,:,s*i:s*i+k,s*j:s*j+k] += out_grad[:,:,i,j].numpy().reshape((N,C,1,1))/(k*k)
        grad = Tensor(grad, device=arr.device, dtype=arr.dtype, requires_grad=False)
        return [grad]

avg_pool = register_op("AvgPool", AvgPoolOp())


class FlipOp(Op):
    def __call__(self, a: Tensor, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'axes': axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return [flip(out_grad, node.attrs['axes'])]
        ### END YOUR SOLUTION

flip = register_op("Flip", FlipOp())


class DilateOp(Op):
    def __call__(self, a: Tensor, dilation: int, axes: tuple) -> Tensor:
        return Tensor.make_from_op(self, [a], attrs={'dilation': dilation, 'axes': axes})

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        dilation, axes = node.attrs['dilation'], node.attrs['axes']
        if dilation <= 0:
            return [out_grad]
        idxs = [slice(None, None, None)] * len(out_grad.shape)
        if isinstance(axes, int): axes = (axes,)
        for i in axes:
            if i >= len(out_grad.shape): continue
            idxs[i] = slice(None, None, dilation+1)
        grad = out_grad[tuple(idxs)]
        if grad.shape[-1] == 1:
            grad = grad.reshape(tuple(grad.shape[:-1]))
        return [grad]
        ### END YOUR SOLUTION

dilate = register_op("Dilate", DilateOp())


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    device = device if device else default_device()

    if not rand or "dist" not in rand:
        arr = device.empty(shape, dtype)
        device.fill(arr, fill_value)
    else:
        if rand["dist"] == "normal":
            arr = device.randn(shape, dtype, mean=rand["mean"], std=rand["std"])
        if rand["dist"] == "binomial":
            arr = device.randb(shape, dtype, ntrials=rand["trials"], p=rand["prob"])
        if rand["dist"] == "uniform":
            arr = device.randu(shape, dtype, low=rand["low"], high=rand["high"])

    return Tensor.make_const(arr, device, requires_grad=requires_grad)


def one_hot(labels: Tensor, *, num_classes=10, dtype="float32", device=None):
    device = device if device else default_device()
    arr = device.one_hot(labels.numpy(), num_classes=num_classes)
    return Tensor.make_const(arr, device, requires_grad=False)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
