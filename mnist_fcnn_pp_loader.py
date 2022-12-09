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


def myprint(a):
    print(a+' = ', eval(a))


# load the main results:
n_params = np.loadtxt(args.results_folder+'/n_params')
test_loss_ave_list = np.savetxt(args.results_folder+'/test_loss_ave_list')
accuracy_list = np.savetxt(args.results_folder+'/accuracy_list')    

# plots the main results:    
fig, ax = plt.subplots()
plt.plot(n_params, test_loss_ave_list)
plt.show()



