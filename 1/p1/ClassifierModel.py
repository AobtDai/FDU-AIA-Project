r""" this file is to construct the handwrite Chinese 
    character classifier net"""

import math
import os
import numpy as np
from BPLayer import BPLayer


class ClassifierNet(object):
    def __init__(self, layer_arch=[28*28,128,64,12], lr=0.01, random_range=0.15, 
                 train_data_size=8000, batch_size=20, task_kind="Classifier", 
                 is_load=False, load_path=""):
        assert len(layer_arch) >= 2, " ** Error!! 2 layers are needed at least!\n"

        self.layer_arch = layer_arch
        self.lr = lr
        self.random_range = random_range
        self.train_data_size = train_data_size
        self.batch_size = batch_size
        self.task_kind = task_kind
        self.is_load = is_load
        self.load_path = load_path

        self.layers = []
        for i in range(0, len(self.layer_arch)-1):
            if i==len(self.layer_arch)-2:
                self.layers.append(BPLayer(self.layer_arch[i], self.layer_arch[i+1], 
                                           self.random_range, True, self.task_kind))
            else :
                self.layers.append(BPLayer(self.layer_arch[i], self.layer_arch[i+1], 
                                           self.random_range, False, self.task_kind))
        
        if is_load:
            self.load_model()
    
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

    def save_model(self):
        print("-------------saving best model--------------------\n")
        i = 0
        for layer in self.layers:
            save_file_w = os.path.join(self.load_path, "w%d%d.npy"%(i, i+1))
            save_file_b = os.path.join(self.load_path, "b%d%d.npy"%(i, i+1))
            w = layer.get_weight()
            b = layer.get_bias()
            np.save(save_file_w, w)
            np.save(save_file_b, b)
            i += 1
    
    def load_model(self):
        print("-------------loading existed model----------------\n")
        i = 0
        for layer in self.layers:
            load_file_w = os.path.join(self.load_path, "w%d%d.npy"%(i, i+1))
            load_file_b = os.path.join(self.load_path, "b%d%d.npy"%(i, i+1))
            w = np.load(load_file_w)
            b = np.load(load_file_b)
            layer.init_weight(w)
            layer.init_bias(b)
            i += 1
    