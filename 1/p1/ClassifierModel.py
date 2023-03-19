r""" this file is to construct the handwrite Chinese 
    character classifier net"""

import math
import numpy as np
from BPLayer import BPLayer


class ClassifierNet(object):
    def __init__(self, layer_arch=[28*28,128,64,12], lr=0.01, random_range=0.15, 
                 train_data_size=8000, batch_size=20, task_kind="Classify"):
        assert len(layer_arch) >= 2, " ** Error!! 2 layers are needed at least!\n"

        self.layer_arch = layer_arch
        self.lr = lr
        self.random_range = random_range
        self.train_data_size = train_data_size
        self.train_data = []
        self.eval_data = []
        self.batch_size = batch_size
        self.task_kind = task_kind

        self.layers = []
        for i in range(0, len(self.layer_arch)-1):
            if i==len(self.layer_arch)-2:
                self.layers.append(BPLayer(self.layer_arch[i], self.layer_arch[i+1], 
                                           self.random_range, True, self.task_kind))
            else :
                self.layers.append(BPLayer(self.layer_arch[i], self.layer_arch[i+1], 
                                           self.random_range, False, self.task_kind))
    
    def forward(self, raw_input):
        for layer in self.layers:
            raw_input = layer.forward(raw_input)
        return raw_input
    
    def backward(self, loss):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def update_weight(self, lr):
        for layer in self.layers:
            layer.update_weight(lr)

    