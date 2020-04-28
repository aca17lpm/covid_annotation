import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from .modelUlti import modelUlti

class NVDMUlti(modelUlti):
    def __init__(self, net, gpu=False):
        super(NVDMUlti, self).__init__(net, gpu=gpu)

    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None):
        self.cache_path = cache_path

        self.evaluation_history = []
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.criterion.cuda()
        for epoch in range(num_epohs):
            all_loss = []
            trainIter = self.pred(trainBatchIter, train=True)
            for pred, y in trainIter:
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                loss_value = float(loss.data.item())
                all_loss.append(loss_value)
            print("             ")
            if valBatchIter:
                output_dict = self.eval(valBatchIter)
                stop_signal = self.earlyStop(output_dict)
                if stop_signal:
                    print('stop signal received, stop training')
                    cache_load_path = os.path.join(self.cachePath, 'best_net.model')
                    print('finish training, load model from ', cache_load_path)
                    self.loadWeights(cache_load_path)

                
            print('epoch ', epoch, 'loss', sum(all_loss)/len(all_loss), ' val acc: ', output_dict['accuracy'])

