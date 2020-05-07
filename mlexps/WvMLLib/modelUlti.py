import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

class modelUlti:
    def __init__(self, net, gpu=False):
        self.net = net
        self.gpu = gpu
        if self.gpu:
            self.net.cuda()

    def train(self, trainBatchIter, num_epohs=100, valBatchIter=None, cache_path=None):
        self.cache_path = cache_path
        output_dict = {}
        output_dict['accuracy'] = 'no val iter'

        self.evaluation_history = []
        self.optimizer = optim.Adam(self.net.parameters())
        self.criterion = nn.CrossEntropyLoss()
        if self.gpu:
            self.criterion.cuda()
        for epoch in range(num_epohs):
            all_loss = []
            trainIter = self.pred(trainBatchIter, train=True)
            for current_prediction in trainIter:
                pred = current_prediction['pred']
                y = current_prediction['y']
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                loss_value = float(loss.data.item())
                all_loss.append(loss_value)
            print("Finish batch")
            if valBatchIter:
                output_dict = self.eval(valBatchIter)
                stop_signal = self.earlyStop(output_dict)
                if stop_signal:
                    print('stop signal received, stop training')
                    cache_load_path = os.path.join(self.cachePath, 'best_net.model')
                    print('finish training, load model from ', cache_load_path)
                    self.loadWeights(cache_load_path)
                    break

            print('epoch ', epoch, 'loss', sum(all_loss)/len(all_loss), ' val acc: ', output_dict['accuracy'])
            if epoch % 20 == 0:
                cache_last_path = os.path.join(self.cache_path, 'last_net.model')
                self.saveWeights(cache_last_path)

        cache_last_path = os.path.join(self.cache_path, 'last_net.model')
        self.saveWeights(cache_last_path)

            
        

    def earlyStop(self, output_dict, metric='accuracy', patience=40):
        result = output_dict[metric]
        stop_signal = False
        self.evaluation_history.append(result)
        num_epochs = len(self.evaluation_history)
        max_result = max(self.evaluation_history)
        max_epoch = self.evaluation_history.index(max_result)
        max_passed = num_epochs - max_epoch
        if max_passed >= patience:
            stop_signal = True

        if max_passed == 1:
            print('caching best           ')
            cache_path = os.path.join(self.cache_path, 'best_net.model')
            self.saveWeights(cache_path)
        return stop_signal


    def pred(self, batchGen, train=False):
        pre_embd = False
        if train:
            self.net.train()
            self.optimizer.zero_grad()
        else:
            self.net.eval()
        i=0
        for x, y in batchGen:
            i+=1
            print("processing batch", i, end='\r')
            if self.gpu:
                y = y.type(torch.cuda.LongTensor)
                y.cuda()
                if batchGen.dataIter.postProcessor.embd_ready:
                    pre_embd = True
                    x = x.type(torch.cuda.FloatTensor).squeeze(1)
                    x.cuda()
                else:
                    x = x.type(torch.cuda.LongTensor)
                    x.cuda()

            pred = self.net(x, None, pre_embd = pre_embd)
            output_dict = {}
            output_dict['pred'] = pred
            output_dict['y'] = y
            yield output_dict

    def eval(self, batchGen):
        output_dict = {}
        all_prediction = []
        all_true_label = []
        print(len(batchGen))
        print(len(batchGen.dataIter))
        for current_prediction in self.pred(batchGen):
            pred = current_prediction['pred']
            y = current_prediction['y']
            current_batch_out = F.softmax(pred, dim=-1)
            label_prediction = torch.max(current_batch_out, -1)[1]
            current_batch_out_list = current_batch_out.to('cpu').detach().numpy()
            label_prediction_list = label_prediction.to('cpu').detach().numpy()
            y_list = y.to('cpu').detach().numpy()
            all_prediction.append(label_prediction_list)
            all_true_label.append(y_list)

        all_prediction = np.concatenate(all_prediction)
        all_true_label = np.concatenate(all_true_label)
        print(len(all_true_label))
        num_correct = (all_prediction == all_true_label).sum()
        accuracy = num_correct / len(all_prediction)
        output_dict['accuracy'] = accuracy
        output_dict['f-measure'] = {}
        num_classes = len(batchGen.dataIter.postProcessor.labelsFields)
        for class_id in list(range(num_classes)):
            f_measure_score = self.fMeasure(all_prediction, all_true_label, class_id)
            output_dict['f-measure']['class '+str(class_id)] = f_measure_score

        return output_dict


    def saveWeights(self, save_path):
        torch.save(self.net.state_dict(), save_path)

    def loadWeights(self, load_path, cpu=True):
        if cpu:
            self.net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')), strict=False)
        else:
            self.net.load_state_dict(torch.load(load_path), strict=False)
        self.net.eval()


    def fMeasure(self, all_prediction, true_label, class_id, ignoreid=None):
        #print(class_id)
        mask = [class_id] * len(all_prediction)
        mask_arrary = np.array(mask)
        pred_mask = np.argwhere(all_prediction==class_id)
        #print(pred_mask)
        true_mask = np.argwhere(true_label==class_id)
        #print(true_mask)
        #print(len(true_mask))
        

        total_pred = 0
        total_true = 0
        pc = 0
        for i in pred_mask:
            if all_prediction[i[0]] == true_label[i[0]]:
                pc+=1
            if true_label[i[0]] != ignoreid:
                total_pred += 1


        rc = 0
        for i in true_mask:
            if all_prediction[i[0]] == true_label[i[0]]:
                rc+=1
            if true_label[i[0]] != ignoreid:
                total_true += 1

        if total_pred == 0:
            precision = 0
        else:
            precision = float(pc)/total_pred
        if total_true == 0:
            recall = 0
        else:
            recall = float(rc)/total_true
        if (precision+recall)==0:
            f_measure = 0
        else:
            f_measure = 2*((precision*recall)/(precision+recall))
        #print(total_true)
        return precision, recall, f_measure, total_pred, total_true, pc, rc
