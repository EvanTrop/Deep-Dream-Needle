import torch
import numpy as np
import needle as ndl
from needle import nn
# from needle.nd_backend import NDDevice
# device = NDDevice()


def transfer_linear(torchLayer):
    """
    Recieves a linear layer in torch, outputs an equivalent linear layer
    in needle.
    
    Assumes numpy backend, float32 dtype, and the presence of a bias.
    """

    inDim,outDim = torchLayer.in_features,torchLayer.out_features
    ndlLayer = nn.Linear(inDim,outDim) # probably not the preferred way of indicating numpy device, by making a new one

    torchWeight = torchLayer.weight.detach().numpy()
    # transpose; it's stored in reverse in pytorch
    ndlLayer.weight = nn.Parameter(torchWeight.transpose())

    torchBias = torchLayer.bias.detach().numpy()
    ndlLayer.bias = nn.Parameter(torchBias)

    return ndlLayer


def transpose_torch_conv_weight(torchWeight):
    # Need to go from (out,in,k,k) to (k,k,in,out)
    return torchWeight.transpose(0,1).transpose(1,2).transpose(2,3).transpose(0,1).transpose(1,2)

def transfer_conv(torchLayer):
    """
    Recieves a Conv2d layer in torch, outputs an equivalent conv layer
    in needle.
    
    Assumes numpy backend, float32 dtype, and the presence of a bias.
    Assumes no grouped convolution, dilation, or non-square kernels, padding or
    stride.
    """

    in_channels,out_channels = torchLayer.in_channels,torchLayer.out_channels

    kernel_size,stride,padding,dilation = torchLayer.kernel_size,torchLayer.stride,torchLayer.padding,torchLayer.dilation
    assert(dilation == (1,1)) # this means "no dilation" in torch (I think), which we need
    assert(torchLayer.groups == 1) # we don't do grouped convolution

    assert(kernel_size[0] == kernel_size[1])
    assert(stride[0] == stride[1])
    assert(padding[0] == padding[1])

    kernel_size,stride,padding = kernel_size[0],stride[0],padding[0]

    ndlLayer = ndl.nn.Conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    if not (ndlLayer.padding == padding):
        print("The automatically computed padding in nn.Conv did not equal what was asked for")
        ndlLayer.padding = padding

    # transfer over weight
    torchWeight = torchLayer.weight
    # Need to go from (out,in,k,k) to (k,k,in,out)
    torchWeight = transpose_torch_conv_weight(torchWeight)
    assert(torchWeight.shape == ndlLayer.weight.shape)
    ndlLayer.weight = ndl.nn.Parameter(torchWeight.detach().numpy())

    # transfer over bias
    torchBias = torchLayer.bias.detach().numpy()
    ndlLayer.bias = ndl.nn.Parameter(torchBias)

    return ndlLayer


def transfer_MaxPool2d(torchLayer):
    """
    Recieves a MaxPool2d layer in torch, outputs an equivalent conv layer
    in needle.
    """

    assert(torchLayer.padding == 0) # only implemented for this case
    assert(torchLayer.dilation == 1) # only implemented for this case
    assert(torchLayer.ceil_mode == False) # only implemented for this case
    return nn.MaxPool(kernel_size = torchLayer.kernel_size, stride = torchLayer.stride)


# def transfer_AdaptiveAvgPool2d(torchLayer,input_size):
#     """
#     AdaptiveAvgPool2d() is a layer type in torch that acts like our AvgPool,
#     but rather than having a set kernel size and stride, it infers what those
#     should be based on the input dimensions and desired output dimensions. This
#     allows pytorch models to receive variable-sized images.

#     We will instead specify the height and width (inDim) of the input to this layer,
#     and will combine this with the layer's output_size to intuit the desired
#     parameters ourselves.

#     Algorithm described here (but not totally correct): https://discuss.pytorch.org/t/working-of-nn-nn-adaptiveavgpool2d/125389
#     Source code: https://github.com/pytorch/pytorch/blob/a08119afc27f44ff488cbf18c11659d255f378a8/aten/src/THNN/generic/SpatialAdaptiveAveragePooling.c
#     """

#     output_size = torchLayer.output_size
#     if type(output_size) == tuple:
#       assert(len(output_size) == 2 and output_size[0] == output_size[1])
#       output_size = output_size[0]

#     # old way:
#     stride = input_size // output_size
#     kernel_size = input_size - (output_size-1)*stride
#     print(input_size,output_size,kernel_size,stride)

#     """
#     We now need to figure out the stride and kernel_size that torch is choosing to use.
#     One way to do this is to feed it an input and just observe what the kernel size
#     seems to be, and then deduce stride from that
#     """
#     arr = np.random.rand(1,input_size,input_size)
#     torchOutput = torchLayer(torch.tensor(arr)).detach().numpy()
#     kernel_size = None
#     stride = None
#     for i in range(1,input_size+1):
#       "PADDING! WHAT IF IT'S HANDLING OVERFLOW WITH PADDING!"
#       "Or this: https://discuss.pytorch.org/t/what-is-adaptiveavgpool2d/26897/2"
#       if abs(arr[0,:i,:i].mean() - torchOutput[0,0,0]) < 1e-5:
#         print(i)
#         print(abs(arr[0,:i,:i].mean() - torchOutput[0,0,0]))
#         kernel_size = i
#         # now figure out the stride
#         for j in range(1,input_size-kernel_size):
#           print("*",abs(arr[0,j:j+i,:i].mean() - torchOutput[0,1,0]))
#           if abs(arr[0,j:j+i,:i].mean() - torchOutput[0,1,0]) < 1e-5:
#             print(j)
#             stride = j
#             break
#         break
#     if not kernel_size or not stride:
#       kernel_size,stride=5,6
#       print("********")
#       # raise ValueError("Kernel size could not be deduced")
#     # stride = int(np.ceil(input_size/output_size))
#     print(input_size,output_size,kernel_size,stride)

#     print()

#     return nn.AvgPool(stride=stride, kernel_size=kernel_size)




def transfer_model(torchModel,imgDim):
    """
    Currently assumes the model is the first part of VGG16. Transfers the model to needle layer by layer
    and outputs the needle version
    """

    ndlLayers = []
    # need to add in a Flatten layer before the linear layers for things to work in needle
    "FOR NOW we can only go up to but not including the adaptive pooling layer"
    # for layer in torchModel.features + [torchModel.avgpool,"Flatten"] + torchModel.classifier: # include a flatten?
    for layer in torchModel:
        s = str(layer)

        if s.startswith("Linear"):
          ndlLayers += [transfer_linear(layer)]
        elif s.startswith("Conv2d"):
          ndlLayers += [transfer_conv(layer)]
        elif s.startswith("ReLU"):
          ndlLayers += [nn.ReLU()]
        elif s.startswith("MaxPool2d"):
          ndlLayers += [transfer_MaxPool2d(layer)]
        elif s.startswith("AdaptiveAvgPool2d"):
          raise ValueError("Not implemented yet")
        elif s.startswith("Flatten"):
          ndlLayers += [nn.Flatten()]
        elif s.startswith("Dropout"):
          # not worrying about dropout layers because we aren't training the model
          pass 
        else:
          raise ValueError("Asked to transfer layer that isn't implemented yet:", s)

    return nn.Sequential(*ndlLayers)



















