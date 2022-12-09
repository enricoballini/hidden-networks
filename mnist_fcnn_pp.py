# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pdb
from prettytable import PrettyTable

def myprint(a):
    print(a+' = ', eval(a))
    


def count_parameters_1(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

    
def count_parameters_2(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Args():
    def __init__(self):
        """ """
  
args = Args()      
args.batch_size = 5 #64        
args.test_batch_size = 10000
args.epochs = 1000
args.lr = 0.01
args.momentum = 0.9
args.wd = 0.0005
args.no_cuda = False
args.seed = 1
args.log_interval = 1
args.save_model = False
args.data = '../data'
args.sparsity = 0.5
args.results_folder = './results'


use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int( (1-k)*scores.numel() )

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out # ehm... so what is flat_out? read comment above...

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None



class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x



class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), args.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)
        return x



# NOTE: not used here but we use NON-AFFINE Normalization!
# So there is no learned parameters for your nomralization layer.
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)



class Net(nn.Module):
    def __init__(self, n2):
        super(Net, self).__init__()
        self.fc1 = SupermaskLinear(28*28, n2, bias=False)
        self.fc2 = SupermaskLinear(n2, 10, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class Net_classic(nn.Module):
    def __init__(self, n2):
        super(Net_classic, self).__init__()
        self.fc1 = nn.Linear(28*28, n2, bias=False)
        self.fc2 = nn.Linear(n2, 10, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # anche loro devono essere spostati aqp...
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    return loss


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            i += 1

    test_loss_ave = test_loss/i #len(test_loader.dataset) ### mmm...

    accuracy = 100. * correct / len(test_loader.dataset)
    print( '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_ave, correct, len(test_loader.dataset),
        accuracy) )
    
    return accuracy, test_loss_ave






# training: ====================================================================
train_dataset= datasets.MNIST(os.path.join(args.data, 'mnist'), train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,)) # where do they come from?
               ]))
train_dataset = torch.utils.data.Subset(train_dataset, np.arange(100))
train_loader = torch.utils.data.DataLoader( train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
    
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(os.path.join(args.data, 'mnist'), train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)




loss_list = np.array([])
n2_list = np.array([10, 20, 40, 60, 80, 100, 2000, 6000])
n2_list = np.array([10, 20, 30, 40, 50, 75, 100, 150, 200,400 ])
accuracy_list = np.array([])
test_loss_ave_list = np.array([])

for n2 in n2_list:                            
    #model = Net(n2).to(device)
    model = Net_classic(n2).to(device)
    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important! ### altrimenti? i gradienti non sono richiesti
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs+1):
        myprint('n2')
        myprint('scheduler.get_last_lr()')
        loss = train(model, device, train_loader, optimizer, criterion, epoch)    
        accuracy, test_loss_ave = test(model, device, criterion, test_loader)
        scheduler.step()
    loss_list = np.append(loss_list, loss.to('cpu').detach().numpy())
    accuracy_list = np.append(accuracy_list, accuracy)
    test_loss_ave_list = np.append(test_loss_ave_list, test_loss_ave.to('cpu').detach().numpy()) # this is an estimation os the expected risk using the traind nn
    
    if args.save_model:
        torch.save(model, 'nn_'+str(n2))

n_params = (28*28+10)*n2_list
myprint('n_params')
myprint('test_loss_ave_list')
myprint('accuracy_list')

# save the main results:
np.savetxt(args.results_folder+'/n_params', n_params)
np.savetxt(args.results_folder+'/test_loss_ave_list', test_loss_ave_list)
np.savetxt(args.results_folder+'/accuracy_list', accuracy_list)    

# plots the main results:    
fig, ax = plt.subplots()
plt.plot(n_params, test_loss_ave_list)
plt.show()



