from transformers import BertModel
from .miscLayer import BERT_Embedding
from .NVDM import BERT_Mapping_mapping
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class BERT_ATT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = 768
        self.n_classes = len(config['TARGET'].get('labels'))
        hidden_dim = 300
        self.bert_mapping = BERT_Mapping_mapping(bert_dim)

        self.hidden1 = nn.Linear(bert_dim, 300)
        self.hidden2 = nn.Linear(300, 50)
        self.layer_output = torch.nn.Linear(50, self.n_classes)


    def forward(self, x, mask=None, pre_embd=False):
        if pre_embd:
            bert_rep = x
        else:
            bert_rep = self.bert_embedding(x, mask)
            bert_rep = bert_rep[0]

        atted = self.bert_mapping(bert_rep)

        hidden = F.leaky_relu(self.hidden1(atted))
        hidden = F.leaky_relu(self.hidden2(hidden))
        out = self.layer_output(hidden)
        return out

