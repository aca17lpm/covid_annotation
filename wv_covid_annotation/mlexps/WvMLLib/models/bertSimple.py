from transformers import BertModel
from .miscLayer import BERT_Embedding
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class BERT_Simple(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert_embedding = BERT_Embedding(config)
        bert_dim = 768
        self.n_classes = len(config['TARGET'].get('labels'))
        hidden_dim = 300
        self.hidden1 = nn.Linear(bert_dim, 300)
        self.hidden2 = nn.Linear(300, 100)
        self.layer_output = torch.nn.Linear(100, self.n_classes)


    def forward(self, x, mask=None, pre_embd=False):
        if pre_embd:
            bert_rep = x
        else:
            bert_rep = self.bert_embedding(x, mask)
            bert_rep = bert_rep[0]
        bert_rep = bert_rep[:,0]

        hidden = F.leaky_relu(self.hidden1(bert_rep))
        hidden = F.leaky_relu(self.hidden2(hidden))
        out = self.layer_output(hidden)
        return out

