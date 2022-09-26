#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch


class Block(nn.Module):
     def __init__(self, in_channels, num_filters, kernel_size, pool_size):
         super(Block, self).__init__()
         self.conv = nn.Conv2d(in_channels,
                               num_filters,
                               kernel_size=kernel_size)
         self.pool = nn.MaxPool2d(kernel_size=pool_size)
         self.relu = nn.ReLU()

     def forward(self, x):
         x = self.conv(x)
         x = self.pool(x)
         x = self.relu(x)
         return x

 # Define the network using the above blocks.
class Network(nn.Module):
     def __init__(self):
         super().__init__()
         self.layer1 = Block(1, 10, 5, 2)
         self.layer2 = Block(10, 20, 5, 2)
         self.layer3 = nn.Linear(320, 256)
         self.layer3_act = nn.ReLU()
         self.layer4 = nn.Linear(256, 10)

         self.softmax = nn.LogSoftmax(1)
         self.loss = nn.NLLLoss(reduction="mean")

     def forward(self, x, target=None):
         x = self.layer1(x)
         x = self.layer2(x)
         x = x.view(-1, 320)

         x = self.layer3_act(self.layer3(x))
         x = self.layer4(x)
         x = self.softmax(x)

         if target is not None:
             loss = self.loss(x, target)
             return x, loss
         return x


model = Network()

infer_opts = poptorch.Options()
infer_opts.deviceIterations(32) 
inference_model = poptorch.inferenceModel(model, options=infer_opts)

data = torch.ones(32,1, 28, 28)

output = inference_model(data)