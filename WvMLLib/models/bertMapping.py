from transformers import BertModel
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .miscLayer import BERT_Embedding, SingleHeadAttention, EncoderLayer

class BERT_Mapping_mapping(nn.Module):
    def __init__(self, bert_dim):
        super().__init__()
        self.encoder1 = EncoderLayer(768, 12)
        self.encoder2 = EncoderLayer(768, 12)
        self.att = SingleHeadAttention(bert_dim, bert_dim)

    def forward(self,x, mask):
        encoded = self.encoder1(x, None)
        encoded = self.encoder2(encoded, None)
        atted = self.att(encoded)
        return atted


class WVClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        hidden_dim = 300
        self.hidden1 = nn.Linear(input_dim, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.layer_output = torch.nn.Linear(100, n_classes)

    def forward(self, x):
        hidden = F.leaky_relu(self.hidden1(x))
        hidden = F.leaky_relu(self.hidden2(hidden))
        out = self.layer_output(hidden)
        return out


class BERT_Mapping(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.target_desc = None
        self.bert_embedding = BERT_Embedding(config)
        self.n_classes = len(config['TARGET'].get('labels'))
        bert_dim = 768

        self.bert_mapping = BERT_Mapping_mapping(bert_dim)
        self.wv_classifier = WVClassifier(bert_dim, self.n_classes)



    def forward(self, x, mask):
        #print(x.shape)
        #print(mask.shape)
        #print(x)
        bert_rep = self.bert_embedding(x, mask)
        #print(bert_rep[0].shape)
        #print(bert_rep[1].shape)
        bert_rep = bert_rep[0]
        atted = self.bert_mapping(bert_rep, mask)
        #print(atted.shape)
        #out = self.wv_classifier(atted)
        out = atted.matmul(self.target_desc)
        return out, atted


    def set_target_desc(self, label_desc, label_desc_mask):
        with torch.no_grad():
            self.target_desc = self.bert_embedding(label_desc, label_desc_mask)[0][:,0]
        self.target_desc = self.target_desc.transpose(0,1)
        print(self.target_desc.shape)
        





