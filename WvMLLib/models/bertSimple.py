from transformers import BertModel
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class BERT_Simple(nn.Module):
    def __init__(self, config):
        super().__init__()

        bert_model_path = os.path.join(config['BERT'].get('bert_path'), 'model')
        bert_dim = int(config['BERT'].get('bert_dim'))
        self.n_classes = len(config['TARGET'].get('labels'))
        #print(self.n_classes)
        #self.n_classes = 11
        self.bert = BertModel.from_pretrained(bert_model_path)
        for p in self.bert.parameters():
            p.requires_grad = False

        hidden_dim = 300
        self.hidden1 = nn.Linear(bert_dim, 300)
        #self.nonlin1 = torch.nn.Tanh()
        self.hidden2 = nn.Linear(300, 100)
        #self.nonlin2 = torch.nn.Tanh()
        self.layer_output = torch.nn.Linear(100, self.n_classes)


    def forward(self, x, mask):
        #print(x.shape)
        #print(mask.shape)
        bert_rep = self.bert(x, attention_mask=mask)[0]
        bert_rep = bert_rep[:,0]
        hidden = F.leaky_relu(self.hidden1(bert_rep))
        hidden = F.leaky_relu(self.hidden2(hidden))
        out = self.layer_output(hidden)
        return out

