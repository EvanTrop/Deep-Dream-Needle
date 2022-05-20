"""NDDArray backed computation backend.

This backend uses cuda backend_ndarray for cached data and computation.
"""
from needle import backend_ndarray as nd
from needle.device import Device, DLDeviceType
from needle.ops import register_op_attr
import needle.device
import numpy as np


class NDDevice(Device):
    def array(self, array, dtype):
        return nd.array(array, dtype=dtype, device=self.nd_device)

    def empty(self, shape, dtype):
        return nd.empty(shape, dtype=dtype, device=self.nd_device)

    def to_numpy(self, data):
        return data.numpy()

    def fill(self, array, fill_value):
        array.fill(fill_value)
        return array

    def randn(self, shape, dtype, mean=0.0, std=1.0):
        return nd.array(np.random.normal(loc=mean, scale=std, size=shape).astype(dtype), device=self.nd_device)

    def randb(self, shape, dtype, ntrials=1, p=0.5):
        return nd.array(np.random.binomial(ntrials, p, size=shape).astype(dtype), device=self.nd_device)

    def randu(self, shape, dtype, low=0, high=0):
        return nd.array(np.random.uniform(low=low, high=high, size=shape).astype(dtype), device=self.nd_device)

    def one_hot(self, y, num_classes=10):
        #TODO fix this
        y_one_hot = []
        for i in range(y.shape[0]):
            y_one_hot.append(np.eye(num_classes)[int(y[i])])
        y_one_hot = np.array(y_one_hot)
        return nd.array(y_one_hot, device=self.nd_device)

    def enabled(self):
        return self.nd_device.enabled()

    def compute(self, op, inputs, attrs):
        """Dispatch device specific computation"""
        # dispatch device specific compute to op.numpy_compute
        # these computation are registered below.
        return op.nd_compute(inputs, attrs)



class CUDADevice(NDDevice):
    def __init__(self, device_id: int = 0):
        assert device_id == 0
        self.nd_device = nd.cuda()
        self.device_id = device_id

    def __repr__(self):
        return "cuda(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CUDA, self.device_id)

    def __str__(self):
        return self.__repr__()


class CPUDevice(NDDevice):
    def __init__(self, device_id: int = 0):
        # self.nd_device = nd.cpu()
        self.nd_device = nd.numpy_device()
        self.device_id = device_id

    def __repr__(self):
        return "cpu(%d)" % self.device_id

    def __dlpack_device__(self):
        return (DLDeviceType.CPU, self.device_id)

    def __str__(self):
        return self.__repr__()



def cuda(device_id: int = 0) -> CUDADevice:
    return CUDADevice(device_id)


def cpu() -> CPUDevice:
    return CPUDevice()

# set default device to be cpu device.
needle.device._DEFAULT_DEVICE = CPUDevice

def register_nd_compute(name, value=None):
    """Register the compute property based on backend_ndarray
    nd computation can be shared across multiple backends.
    """
    return register_op_attr(name, "nd_compute", value)


# device specific computations
@register_nd_compute("EWiseAdd")
def add(inputs, attrs):
    return inputs[0] + inputs[1]


@register_nd_compute("AddScalar")
def add_scalar(inputs, attrs):
    return inputs[0] + attrs["scalar"]


@register_nd_compute("EWiseMul")
def mul(inputs, attrs):
    return inputs[0] * inputs[1]


@register_nd_compute("MulScalar")
def mul(inputs, attrs):
    return inputs[0] * attrs["scalar"]


@register_nd_compute("EWiseDiv")
def divide(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] / inputs[1]
    ### END YOUR SOLUTION


@register_nd_compute("DivScalar")
def divide_scalar(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] / attrs["scalar"]
    ### END YOUR SOLUTION


@register_nd_compute("PowerScalar")
def power_scalar(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] ** attrs["scalar"]
    ### END YOUR SOLUTION


@register_nd_compute("MatMul")
def matmul(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0] @ inputs[1]
    ### END YOUR SOLUTION


@register_nd_compute("Summation")
def summation(inputs, attrs):
    """
    Parameters:
    axes - int or tuple of ints or None

    If axes is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis.
    If axes is None, sum over all of the axes.

    Returns an array with the same shape, except with the specified axes removed.
    """
    ### BEGIN YOUR SOLUTION
    if attrs["axes"] is None or isinstance(attrs["axes"], int):
        res = inputs[0].sum(attrs["axes"])
    else:
        res = inputs[0]
        for i in attrs["axes"]:
            res = res.sum(axis=i)
    if attrs["axes"] is None: return res.reshape(())
    else:
        reduced_shape = np.delete(res.shape, attrs["axes"])
        return res.reshape(reduced_shape)
    ### END YOUR SOLUTION


@register_nd_compute("BroadcastTo")
def broadcast_to(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    if len(inputs[0].shape) < len(attrs["shape"]):
        cast_shape = np.ones_like(attrs["shape"])
        if len(inputs[0].shape) > 0: # if not scalar
            cast_shape[-len(inputs[0].shape):] = inputs[0].shape
        return inputs[0].reshape(cast_shape).broadcast_to(attrs["shape"])
    return inputs[0].broadcast_to(attrs["shape"])
    ### END YOUR SOLUTION


@register_nd_compute("Reshape")
def reshape(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].reshape(attrs["shape"])
    ### END YOUR SOLUTION


@register_nd_compute("Negate")
def negate(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return -inputs[0]
    ### END YOUR SOLUTION


@register_nd_compute("Transpose")
def transpose(inputs, attrs):
    """
    Parameters:
    axes - tuple of ints or None

    If axes is a tuple of ints, a sum is performed on all of the axes specified in the tuple instead of a single axis.
    If axes is None, permutes the last two axes.
    """
    ### BEGIN YOUR SOLUTION
    d = len(inputs[0].shape)
    if attrs["axes"]: axis0, axis1 = attrs["axes"][0], attrs["axes"][1]
    else: axis0, axis1 = -1, -2
    new_axes = np.arange(len(inputs[0].shape))
    new_axes[axis0], new_axes[axis1] = new_axes[axis1], new_axes[axis0]
    return inputs[0].permute(tuple(new_axes))
    ### END YOUR SOLUTION


@register_nd_compute("Log")
def log(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].log()
    ### END YOUR SOLUTION


@register_nd_compute("Exp")
def exp(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].exp()
    ### END YOUR SOLUTION


@register_nd_compute("ReLU")
def relu(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].maximum(0)
    ### END YOUR SOLUTION


@register_nd_compute("LogSoftmax")
def logsoftmax(inputs, attrs):
    """
    Computes log softmax along the last dimension of the array.
    """
    ### BEGIN YOUR SOLUTION
    last_axis = len(inputs[0].shape) - 1
    x, c = inputs[0], inputs[0].max(axis=last_axis).broadcast_to(inputs[0].shape)
    return x - c - ((x-c).exp().sum(axis=last_axis)).log().broadcast_to(x.shape)
    ### END YOUR SOLUTION


@register_nd_compute("Tanh")
def tanh(inputs, attrs):
    ### BEGIN YOUR SOLUTION
    return inputs[0].tanh()
    ### END YOUR SOLUTION


@register_nd_compute("GetItem")
def get_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Returns array indexed by idxs i.e. if array A has shape (5, 3, 2),
    then the shape of the A[0, :, :] would be (3, 2).
    """
    ### BEGIN YOUR SOLUTION
    if isinstance(attrs["idxs"], int):
        attrs["idxs"] = (attrs["idxs"],)
    shape = []
    res = inputs[0][attrs["idxs"]].compact()
    for s, i in zip(attrs["idxs"], res.shape):
        if isinstance(s, slice):
            shape.append(i)
    return res.reshape(tuple(shape))
    ### END YOUR SOLUTION


@register_nd_compute("SetItem")
def set_item(inputs, attrs):
    """
    Parameters:
    idxs - indices to index array; tuple of ints or slices

    Sets array A at idxs with array B and returns the array.
    """
    ### BEGIN YOUR SOLUTION
    inputs[0][attrs["idxs"]] = inputs[1]
    ### END YOUR SOLUTION


@register_nd_compute("Stack")
def stack(As, attrs):
    """
    Concatenates a sequence of arrays along a new dimension.

    Parameters:
    axis - dimension to concatenate along

    All arrays need to be of the same size.
    """
    ### BEGIN YOUR SOLUTION
    axis = attrs["axis"]
    shape = list(As[0].shape[:axis]) + [len(As)] + list(As[0].shape[axis:])
    res = nd.empty(tuple(shape), device=As[0].device)
    slices = [slice(None, None, None)] * len(shape)
    for i in range(len(As)):
        slices[axis] = i
        res[tuple(slices)] = As[i]
    return res
    ### END YOUR SOLUTION


@register_nd_compute("Flip")
def flip(inputs, attrs):
    """
    Flips the input along specified axes.

    Parameters:
    axes - Axes to flip.
    """
    ### BEGIN YOUR SOLUTION
    return inputs[0].flip(attrs["axes"])
    ### END YOUR SOLUTION


@register_nd_compute("Dilate")
def dilate(inputs, attrs):
    """
    Dilates the input by a dilation factor on specified axes.
    (i.e., inserts 0s between elements)

    Parameters:
    dilation - Dilation amount (number of 0s to insert)
    axes - Axes to dilate by this amount
    """
    ### BEGIN YOUR SOLUTION
    dilation, axes = attrs["dilation"], attrs["axes"]
    if dilation <= 0:
        return inputs[0]
    if isinstance(axes, int): axes = (axes,)
    out_shape = list(inputs[0].shape)
    idxs = [slice(None,None,None)] * len(out_shape)
    for i in axes: 
        if i >= len(out_shape): continue
        out_shape[i] *= dilation + 1
        idxs[i] = slice(None,None,dilation+1)
    out = nd.array(np.zeros(out_shape), device=inputs[0].device)
    out[tuple(idxs)] = inputs[0]
    return out
    ### END YOUR SOLUTION


@register_nd_compute("Conv")
def conv(inputs, attrs):
    """
    Multi-channel 2D convolution of two inputs (called input and weight, respectively).
    inputs[0]: "input", NHWC
    inputs[1]: "weight", (kernel_size, kernel_size, c_in, c_out)

    Parameters:
    padding - (int) Pad the HW axes of the input by this amount
    stride - (int) Stride of the convolution
    """
    ### BEGIN YOUR SOLUTION
    padding = attrs['padding']
    stride = attrs['stride']
    Z, weight = inputs[0].compact(), inputs[1].compact()
    N, H, W, C_in = Z.shape
    K, _, _, C_out = weight.shape
    Z = inputs[0].pad(axes=((0,0), (padding,)*2, (padding,)*2, (0,0)))
    out_h, out_w = int((H-K+2*padding)/stride+1), int((W-K+2*padding)/stride+1)
    Ns, Hs, Ws, Cs = Z.strides
    Z = Z.as_strided(shape=(N, out_h, out_w, K, K, C_in),
                     strides=(Ns, Hs*stride, Ws*stride, Hs, Ws, Cs)).compact()
    Z = Z.reshape((N*out_h*out_w, K*K*C_in))
    weight = weight.reshape((K*K*C_in, C_out))
    return (Z @ weight).reshape((N, out_h, out_w, C_out))
    ### END YOUR SOLUTION


@register_nd_compute("MaxPool")
def max_pool(inputs, attrs):
    """
    Applies 2D max pooling operation on input with NCHW form.

    Parameters:
    kernel_size - (int) kernel size of the max pooling
    stride - (int) Stride of the max pooling
    """
    assert len(inputs[0].shape) == 4
    K = attrs['kernel_size']
    S = attrs['stride']
    N, C, H, W = inputs[0].shape
    H_out, W_out = (H-K)//S + 1, (W-K)//S + 1
    out = inputs[0].as_strided(shape=(N, C, H_out, W_out, K, K),
                               strides=(C*H*W, H*W, W*S, S, W, 1)).compact()
    out = out.max(axis=5).max(axis=4).reshape((N, C, H_out, W_out))
    return out


@register_nd_compute("AvgPool")
def avg_pool(inputs, attrs):
    """
    Applies 2D average pooling operation on input with NCHW form.

    Parameters:
    kernel_size - (int) kernel size of the max pooling
    stride - (int) Stride of the max pooling
    """
    assert len(inputs[0].shape) == 4
    K = attrs['kernel_size']
    S = attrs['stride']
    N, C, H, W = inputs[0].shape
    H_out, W_out = (H-K)//S + 1, (W-K)//S + 1
    out = inputs[0].as_strided(shape=(N, C, H_out, W_out, K, K),
                               strides=(C*H*W, H*W, W*S, S, W, 1)).compact()
    out = out.sum(axis=5).sum(axis=4).reshape((N, C, H_out, W_out)) / (K*K)
    return out









