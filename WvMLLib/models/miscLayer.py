from transformers import BertModel
import os
import torch.nn.functional as F
import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_output, dropout = 0.1):
        super().__init__()

        self.q = nn.Parameter(torch.randn([d_output, 1]).float())
        self.v_linear = nn.Linear(d_model, d_output)
        self.dropout_v = nn.Dropout(dropout)
        self.k_linear = nn.Linear(d_model, d_output)
        self.dropout_k = nn.Dropout(dropout)
        self.softmax_simi = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)
        #self.out = nn.Linear(d_output, d_output)

    def forward(self, x, mask=None):
        k = self.k_linear(x)
        k = F.relu(k)
        k = self.dropout_k(k)
        v = self.v_linear(x)
        v = F.relu(v)
        v = self.dropout_v(v)

        dotProducSimi = k.matmul(self.q)
        normedSimi = self.softmax_simi(dotProducSimi)
        attVector = v.mul(normedSimi)
        weightedSum = torch.sum(attVector, dim=1)
        #output = self.out(weightedSum)
        return weightedSum



def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        # bs, sl, d_model --> bs, sl, heads, sub_d_model
        # d_model = heads * sub_d_model
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class BERT_Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        bert_model_path = os.path.join(config['BERT'].get('bert_path'), 'model')
        bert_dim = int(config['BERT'].get('bert_dim'))
        self.bert = BertModel.from_pretrained(bert_model_path)
        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, x, mask):
        bert_rep = self.bert(x, attention_mask=mask)
        return bert_rep

