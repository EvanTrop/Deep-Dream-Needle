# Deep Dream in Needle

## Contents
**10714-needle-deepdream **- a folder containing the source code to support Needle with most of the interesting code being located within python/needle <br>
**WalkThrough.ipynb** - a notebook which provides a walk through of the final project implementing deep dream in Needle

Throughout CMU's 10-714 course, Deep Learning Systems: Aglorthims and Implementations, we built a deep learning framework called Needle. Needle closesly follows Pytorch's implementation. Instructors provided a code base and through each homework we implemented different elements such as forward/backward computations for various operations, auto differentiation, network modules, optimizers, data, and backend NDarray classes. <br>

For the final course project we used the completed Needle framework to implement the DeepDream concept by Alexander Mordvintsev. Deep Dream involves taking a pretrained image classification model and performing gradient ascent on an input image so that it maximally activates a particular hidden layer (or group of layers) in the model. This creates “dream-like” images, and sheds light on what the network’s layers are focused on at various stages of the model.<br>

Specifically, code was developed to transfer a pretrained VGG16 net in Pytorch to Needle, implement the particular pooling layers found within VGG16 net, and finally a Deep Dream model class/loss function/optimizer.
