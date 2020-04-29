import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from .modelUlti import modelUlti

class modelUltiMapping(modelUlti):
    def __init__(self, net, gpu=False):
        super().__init__(net, gpu=gpu)

    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None):
        self.cache_path = cache_path
        output_dict = {}
        output_dict['accuracy'] = 'no val iter'

        self.evaluation_history = []
        self.optimizer_mapping = optim.Adam(self.net.bert_mapping.parameters())
        self.optimizer_classifier = optim.Adam(self.net.wv_classifier.parameters())
        self.criterion = nn.CrossEntropyLoss()
        #self.desc_criterion = nn.KLDivLoss()
        self.desc_criterion = torch.nn.MSELoss()
        if self.gpu:
            self.criterion.cuda()
        for epoch in range(num_epohs):
            all_loss = []
            trainIter = self.pred(trainBatchIter, train=True)
            for current_prediction in trainIter:
                self.optimizer_mapping.zero_grad()
                self.optimizer_classifier.zero_grad()
                pred = current_prediction['pred']
                y = current_prediction['y']
                atted = current_prediction['atted'] 
                y_desc_representation = current_prediction['y_desc_representation']

                class_loss = self.criterion(pred, y)
                desc_loss = self.desc_criterion(input=atted, target=y_desc_representation)
                loss = class_loss + desc_loss

                loss.backward()
                self.optimizer_mapping.step()
                self.optimizer_classifier.step()
                loss_value = float(loss.data.item())
                all_loss.append(loss_value)
            print("Finish batch")
            if valBatchIter:
                output_dict = self.eval(valBatchIter)
                stop_signal = self.earlyStop(output_dict, patience=20)
                if stop_signal:
                    print('stop signal received, stop training')
                    #cache_load_path = os.path.join(self.cachePath, 'best_net.model')
                    #print('finish training, load model from ', cache_load_path)
                    #self.loadWeights(cache_load_path)
                    break

            print('epoch ', epoch, 'loss', sum(all_loss)/len(all_loss), ' val acc: ', output_dict['accuracy'])
            
        
        cache_load_path = os.path.join(self.cache_path, 'best_net.model')
        print('finish training, load model from ', cache_load_path)
        self.loadWeights(cache_load_path)


    def pred(self, batchGen, train=False):
        if train:
            self.net.train()
            #self.optimizer.zero_grad()
        else:
            self.net.eval()
        i=0
        for x, mask, y, y_desc, y_desc_mask in batchGen:
            i+=1
            print("processing batch", i, end='\r')
            if self.gpu:
                x = x.type(torch.cuda.LongTensor)
                mask = mask.type(torch.cuda.LongTensor)
                y = y.type(torch.cuda.LongTensor)
                y_desc = y_desc.type(torch.cuda.LongTensor)
                y_desc_mask = y_desc_mask.type(torch.cuda.LongTensor)
                x.cuda()
                mask.cuda()
                y.cuda()
                y_desc.cuda()
                y_desc_mask.cuda()
            pred, atted = self.net(x, mask)
            with torch.no_grad():
                y_desc_representation = self.net.bert_embedding(y_desc, y_desc_mask)
            output_dict = {}
            output_dict['pred'] = pred
            output_dict['y'] = y
            output_dict['atted'] = atted
            output_dict['y_desc_representation'] = y_desc_representation[0][:,0]

            yield output_dict
