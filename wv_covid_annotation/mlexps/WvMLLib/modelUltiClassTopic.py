import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
from pathlib import Path
from .modelUltiVAEtm_noatt import ModelUltiVAEtmNOatt

class ModelUltiClass(ModelUltiVAEtmNOatt):
    def __init__(self, net, gpu=False):
        super().__init__(net, gpu=gpu)

    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None, earlyStopping='cls_loss', patience=5):
        self.cache_path = cache_path
        output_dict = {}
        output_dict['accuracy'] = 'no val iter'

        self.evaluation_history = []
        #classifier_paramters = list(self.net.wv_hidden.parameters()) + list(self.net.wv_classifier.parameters())
        #topic_model_paramters = list(self.net.mu.parameters())+ list(self.net.log_sigma.parameters()) + list(self.net.topics.parameters())
        #self.optimizer_classifier = optim.Adam(classifier_paramters)
        #self.optimizer_topic_modelling = optim.Adam(topic_model_paramters)

        self.optimizer = optim.Adam(self.net.parameters())
        #self.criterion = nn.CrossEntropyLoss()
        #if self.gpu:
        #    self.criterion.cuda()
        for epoch in range(num_epohs):
            all_loss = []
            trainIter = self.pred(trainBatchIter, train=True)
            for current_prediction in trainIter:
                self.optimizer.zero_grad()
                #self.optimizer_classifier.zero_grad()
                #self.optimizer_topic_modelling.zero_grad()

                pred = current_prediction['pred']
                y = current_prediction['y']
                atted = current_prediction['atted'] 
                #y_desc_representation = current_prediction['y_desc_representation']

                #class_loss = self.criterion(pred['pred'], y)
                #topic_loss = pred['loss']
                #print(class_loss)
                #desc_loss = self.desc_criterion(input=atted, target=y_desc_representation)
                loss = pred['loss']
                cls_loss = pred['cls_loss'].sum()

                loss.backward()
                self.optimizer.step()
                #self.optimizer_classifier.step()
                #self.optimizer_topic_modelling.step()
                #loss_value = float(loss.data.item())
                loss_value = float(cls_loss.data.item())
                all_loss.append(loss_value)
            if epoch % 20 == 0:
                self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess)
                print('===========')
                self.getClassTopics(trainBatchIter.dataIter.postProcessor.dictProcess)
                cache_last_path = os.path.join(self.cache_path, 'last_net.model')
                self.saveWeights(cache_last_path)
            #print("Finish Epoch ", epoch)
            if valBatchIter:
                output_dict = self.eval(valBatchIter)

            avg_loss = sum(all_loss)/len(all_loss)
            output_dict['cls_loss'] = -avg_loss
            if earlyStopping:
                stop_signal = self.earlyStop(output_dict, patience=patience, metric=earlyStopping)
                if stop_signal:
                    print('stop signal received, stop training')
                    cache_load_path = os.path.join(self.cache_path, 'best_net.model')
                    print('finish training, load model from ', cache_load_path)
                    self.loadWeights(cache_load_path)
                    break
            print('epoch ', epoch, 'loss', avg_loss, ' val acc: ', output_dict['accuracy'])
        cache_last_path = os.path.join(self.cache_path, 'last_net.model')
        self.saveWeights(cache_last_path)
        self.getTopics(trainBatchIter.dataIter.postProcessor.dictProcess)
        print('===========')
        self.getClassTopics(trainBatchIter.dataIter.postProcessor.dictProcess)
            
        
        #cache_load_path = os.path.join(self.cache_path, 'best_net.model')
        #print('finish training, load model from ', cache_load_path)
        #self.loadWeights(cache_load_path)


    def getClassTopics(self, dictProcess, ntop=10):
        termMatrix = self.net.get_class_topics()
        for each_topic in termMatrix:
            trans_list = list(enumerate(each_topic.cpu().numpy()))
            #print(trans_list)
            trans_list = sorted(trans_list, key=lambda k: k[1], reverse=True)
            #print(trans_list)
            topic_words = [dictProcess.get(item[0]) for item in trans_list[:ntop]]
            print(topic_words)

