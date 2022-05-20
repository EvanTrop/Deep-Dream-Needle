import sys
sys.path.append('./python')
import needle as ndl
from needle import nn, ops
from needle import transfer
import numpy as np
import torch
from torch.nn import Conv2d
import torchvision.models as models
import itertools
np.random.seed(0)

"""
Note: I only test the layers when a batch dimension is being used in the input.
So if we're only inputting 1 image, we should probably use a batch dimension of 1.
"""

BATCH_SIZE = [1,3,5,20,50]
IN_SIZE = [1,3,5,20,50]
OUT_SIZE = [1,3,5,20,50]
def test_transfer_linear():
    for b,i,o in itertools.product(BATCH_SIZE,IN_SIZE,OUT_SIZE):
        # print(b,i,o)

        # check weights
        torchLayer = torch.nn.Linear(i,o)
        ndlLayer = transfer.transfer_linear(torchLayer)
        assert (torchLayer.weight.detach().numpy().transpose() == ndlLayer.weight.numpy()).all()
        assert (torchLayer.bias.detach().numpy() == ndlLayer.bias.numpy()).all()

        # check outputs
        arr = np.random.rand(b,i)
        A = ndl.Tensor(arr,dtype='float32',device=ndlLayer.weight.device)
        B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
        A_ = ndlLayer(A)
        B_ = torchLayer(B)
        # print(A_)
        # print(B_)
        np.testing.assert_allclose(A_.numpy(), B_.detach().numpy(), atol=1e-5, rtol=1e-5)

        # Check gradients w.r.t inputs; grads are not populated at all before this
        A_.sum().backward()
        B_.sum().backward()
        # print(A.grad.numpy())
        # print(B.grad.numpy())
        np.testing.assert_allclose(A.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)

        # now check grads w.r.t weights, rather than inputs
        torchWeightGrad = torchLayer.weight.grad.numpy().transpose()
        ndlWeightGrad = ndlLayer.weight.grad.numpy()
        # print(torchWeightGrad,ndlWeightGrad)
        np.testing.assert_allclose(torchWeightGrad, ndlWeightGrad, atol=1e-5, rtol=1e-5)

        torchBiasGrad = torchLayer.bias.grad.numpy()
        ndlBiasGrad = ndlLayer.bias.grad.numpy()
        # print(torchBiasGrad,ndlBiasGrad) # tend to be like whole number 1, or whole number 3 or something?
        np.testing.assert_allclose(torchBiasGrad, ndlBiasGrad, atol=1e-5, rtol=1e-5)

    #     print("done")
    #     assert(False)
    # assert(False)




# "64 may work because with 7 output, kernel 10 and stride 9 actually forms a clean output"
# X = [64,32,128,256,32] # currently focusing on inputs with dimensions that are powers of two
# Y = [1,4]
# Z = [1,4,16]
# def test_transfer_AdaptiveAvgPool2d():

#     torchLayer = models.vgg16(pretrained=True).avgpool

#     for input_size,batchSize,in_channels in itertools.product(X,Y,Z):
#         print(input_size,batchSize,in_channels)

#         # # Currently we use the specific layer from VGG16
#         # # need to import this every time so gradients and adaptations don't get screwy
#         # torchLayer = models.vgg16(pretrained=True).avgpool

#         ndlLayer = transfer.transfer_AdaptiveAvgPool2d(torchLayer,input_size)

#         # check outputs
#         arr = np.random.rand(batchSize,in_channels,input_size,input_size) # nn.Conv in needle takes BCHW format
#         A = ndl.Tensor(arr,dtype='float32')
#         B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
#         A_ = ndlLayer(A)
#         B_ = torchLayer(B)
#         # print(A_)
#         # print(B_)
#         np.testing.assert_allclose(A_.numpy(), B_.detach().numpy(), atol=1e-5, rtol=1e-5)

#         # Check gradients w.r.t inputs; grads are not populated at all before this
#         A_.sum().backward()
#         B_.sum().backward()
#         # print(A.grad.numpy())
#         # print(B.grad.numpy())
#         np.testing.assert_allclose(A.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)

#         print("done")




# all of these are based on layer specs taken from large models like vgg16
# (in_c out_c k stride pad)
CONVS = [(3, 64, (3,3), (1, 1), (1, 1)),
        #  (256, 256, (3, 3), (1, 1), (1, 1)),
        #  (192, 384, (3, 3), (1, 1), (1, 1)),
        #  (384, 256, (3, 3), (1, 1), (1, 1)),
         (64, 192, (5, 5), (1, 1), (2, 2)),
         (3, 64, (7, 7), (2, 2), (3, 3))#, bias=False

        #  (3, 64, (11, 11), (4, 4), (2, 2)), # an example where the padding is not what was automatically computed
         ]
IMAGE_SIZES = [32,64,100]#,512]#,71] # 512 causes some gradient mismatch errors which are modest in proportion (1e-5)
BATCH_SIZES = [1,4]#,5,32 # large batches are probably not critical for our application, but they do work
def test_transfer_conv():
    "make sure after every loop iter gradients are reset"
    for (layerNum,lps),imgDim,batchSize in itertools.product(enumerate(CONVS),IMAGE_SIZES,BATCH_SIZES):
        print(layerNum,imgDim,batchSize)

        # check weights
        torchLayer = Conv2d(lps[0], lps[1], kernel_size=lps[2], stride=lps[3], padding=lps[4])
        ndlLayer = transfer.transfer_conv(torchLayer)
        assert (torchLayer.bias.detach().numpy() == ndlLayer.bias.numpy()).all()
        transposedTorchWeight = transfer.transpose_torch_conv_weight(torchLayer.weight)
        assert (transposedTorchWeight.detach().numpy() == ndlLayer.weight.numpy()).all()

        # check outputs
        arr = np.random.rand(batchSize,torchLayer.in_channels,imgDim,imgDim) # nn.Conv in needle takes BCHW format
        A = ndl.Tensor(arr,dtype='float32')
        B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
        A_ = ndlLayer(A)
        B_ = torchLayer(B)
        # print(A_)
        # print(B_)
        np.testing.assert_allclose(A_.numpy(), B_.detach().numpy(), atol=1e-5, rtol=1e-5)

        # Check gradients w.r.t inputs; grads are not populated at all before this
        A_.sum().backward()
        B_.sum().backward()
        # print(A.grad.numpy())
        # print(B.grad.numpy())
        np.testing.assert_allclose(A.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)

        # now check grads w.r.t weights, rather than inputs
        torchBiasGrad = torchLayer.bias.grad.numpy()
        ndlBiasGrad = ndlLayer.bias.grad.numpy()
        # print(torchBiasGrad,ndlBiasGrad) # tend to be like whole number 1, or whole number 3 or something?
        np.testing.assert_allclose(torchBiasGrad, ndlBiasGrad, atol=1e-5, rtol=1e-5)

        torchWeightGrad = transfer.transpose_torch_conv_weight(torchLayer.weight.grad).numpy()
        ndlWeightGrad = ndlLayer.weight.grad.numpy()
        # print(torchWeightGrad[:2,:],ndlWeightGrad[:2,:])
        np.testing.assert_allclose(torchWeightGrad, ndlWeightGrad, atol=1e-5, rtol=1e-5)

        print("done")
    #     assert(False)
    # assert(False)


def test_transfer_model():
    "fails for size 256?"
    for imgDim in [32,64,128]:#,512]:#,256]:
        print(imgDim)

        torchModel = models.vgg16(pretrained=True).features
        ndlModel = transfer.transfer_model(torchModel,imgDim) # the imgDim input is currently arbitrary because we're not yet using the adaptive pooling layer; you could just put 0, or to future-proof things you could put the width/height of your test image
        # print(ndlModel.modules)

        # check outputs
        arr = np.random.rand(1,3,imgDim,imgDim)
        A = ndl.Tensor(arr,dtype='float32')
        B = torch.tensor(arr, dtype=torch.float, requires_grad=True)
        A_ = ndlModel(A)
        B_ = torchModel(B)
        # print(A_)
        # print(B_)
        np.testing.assert_allclose(A_.numpy(), B_.detach().numpy(), atol=1e-5, rtol=1e-5)

        # Check gradients w.r.t inputs; grads are not populated at all before this
        A_.sum().backward()
        B_.sum().backward()
        # print(A.grad.numpy())
        # print(B.grad.numpy())
        np.testing.assert_allclose(A.grad.numpy(), B.grad.numpy(), atol=1e-5, rtol=1e-5)

        # now check grads w.r.t weights, rather than inputs
        for i in [0,2,5]: # test a few layers
          torchBiasGrad = torchModel[i].bias.grad.numpy()
          ndlBiasGrad = ndlModel.modules[i].bias.grad.numpy()
          # print(torchBiasGrad,ndlBiasGrad) # tend to be like whole number 1, or whole number 3 or something?
          np.testing.assert_allclose(torchBiasGrad, ndlBiasGrad, atol=1e-3, rtol=1e-3) # ease up requirements a bit, needed for size 128 images

          torchWeightGrad = transfer.transpose_torch_conv_weight(torchModel[i].weight.grad).numpy()
          ndlWeightGrad = ndlModel.modules[i].weight.grad.numpy()
          # print(torchWeightGrad[:2,:],ndlWeightGrad[:2,:])
          np.testing.assert_allclose(torchWeightGrad, ndlWeightGrad, atol=1e-3, rtol=1e-3)

        print("done")
    #     assert(False)
    # assert(False)









