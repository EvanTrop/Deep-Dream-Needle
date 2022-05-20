import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from models import *
import time


### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
        correct, total, tot_loss = 0, 0, 0
        for inputs, labels in dataloader:
            opt.reset_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()
            tot_loss += loss.data.numpy() * inputs.shape[0]
            total += inputs.shape[0]
            preds = outputs.numpy().argmax(axis=1)
            correct += np.sum(preds == labels.numpy())
        return correct/total, tot_loss/total
    else:
        model.eval()
        correct, total, tot_loss = 0, 0, 0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            tot_loss += loss.data.numpy() * inputs.shape[0]
            total += inputs.shape[0]
            preds = outputs.numpy().argmax(axis=1)
            correct += np.sum(preds == labels.numpy())
        return correct/total, tot_loss/total
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs): 
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt)
        print(f"epoch {epoch}: avg_acc={avg_acc}, avg_loss={avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn(), opt=None)
    print(f"evaluate: avg_acc={avg_acc}, avg_loss={avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
        correct, total, tot_loss = 0, 0, 0
        # print(data.shape, seq_len)
        for i in range(0, data.shape[0]-1, seq_len):
            inputs, targets = ndl.data.get_batch(data, i, seq_len, device, dtype)
            opt.reset_grad()
            if i == 0:
                outputs, hstates = model(inputs, None)
            else:
                outputs, hstates = model(inputs, hstates)
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            tot_loss += loss.data.numpy() * targets.shape[0]
            total += targets.shape[0]
            preds = outputs.numpy().argmax(axis=1)
            correct += np.sum(preds == targets.numpy())
        return correct/total, tot_loss/total
    else:
        model.eval()
        correct, total, tot_loss = 0, 0, 0
        for i in range(0, data.shape[0]-1, seq_len):
            inputs, targets = ndl.data.get_batch(data, i, seq_len, device, dtype)
            if i == 0:
                outputs, hstates = model(inputs, None)
            else:
                outputs, hstates = model(inputs, hstates)
            loss = loss_fn(outputs, targets)
            tot_loss += loss.data.numpy() * targets.shape[0]
            total += targets.shape[0]
            preds = outputs.numpy().argmax(axis=1)
            correct += np.sum(preds == targets.numpy())
        return correct/total, tot_loss/total
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs): 
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), 
                                              opt, clip, device, dtype)
        print(f"epoch {epoch}: avg_acc={avg_acc}, avg_loss={avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), 
                                          opt=None, device=device, dtype=dtype)
    print(f"evaluate: avg_acc={avg_acc}, avg_loss={avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    # dataset = ndl.data.CIFAR10Dataset("../data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #          dataset=dataset,
    #          batch_size=128,
    #          shuffle=True,
    #          collate_fn=ndl.data.collate_ndarray,
    #          drop_last=False,
    #          device=device,
    #          dtype="float32"
    #          )
    #
    # model = ResNet9(device=device, dtype="float32")
    # train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #       lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("../data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=ndl.cpu(), dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=ndl.cpu())
    train_ptb(model, train_data, seq_len, n_epochs=10)
