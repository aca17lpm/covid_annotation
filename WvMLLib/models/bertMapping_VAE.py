from transformers import BertModel
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from .miscLayer import BERT_Embedding, SingleHeadAttention

class BERT_Mapping_mapping(nn.Module):
    def __init__(self, bert_dim):
        super().__init__()
        self.att = SingleHeadAttention(bert_dim, bert_dim)

    def forward(self,x):
        atted = self.att(x)
        return atted


class WVHidden(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        hidden = F.leaky_relu(self.hidden1(x))
        return hidden

class WVClassifier(nn.Module):
    def __init__(self, n_gaussian, n_classes):
        super().__init__()
        self.layer_output = torch.nn.Linear(n_gaussian, n_classes)

    def forward(self, x):
        out = self.layer_output(x)
        return out



def logQ(mu, log_sigma):
    """
    log Q (Gaussian distribution).
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class BERT_Mapping_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert_embedding = BERT_Embedding(config)
        self.n_classes = len(config['TARGET'].get('labels'))
        bert_dim = 768
        hidden_dim = 300
        num_gaussian = 100

        self.bert_mapping = BERT_Mapping_mapping(bert_dim)
        self.wv_hidden = WVHidden(bert_dim, hidden_dim)

        self.mu = nn.Linear(hidden_dim, num_gaussian)
        self.log_sigma = nn.Linear(hidden_dim, num_gaussian)

        self.wv_classifier = WVClassifier(num_gaussian, self.n_classes)


    def forward(self, x, mask, n_samples=10, true_y=None, train=False):
        #print(true_y.shape)
        bert_rep = self.bert_embedding(x, mask)
        bert_rep = bert_rep[0]
        atted = self.bert_mapping(bert_rep)
        #print(atted.shape)
        hidden = self.wv_hidden(atted)
        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        if train:

            log_q = logQ(mu, log_sigma)
            rec_loss = 0
            for i in range(n_samples):
                z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
                y = self.wv_classifier(z) 
                logy = torch.log_softmax(y, dim=-1)
                rec_loss += (logy * true_y).sum(dim=-1)
            rec_loss =rec_loss / n_samples

            minus_elbo = - (rec_loss - log_q) ## trying to maximise elbo, so we need to minimise minus elbo
            #minus_elbo = minus_elbo.mean()
            minus_elbo = minus_elbo.sum()


            loss = {
                    'loss': minus_elbo,
                    'rec_loss': rec_loss
                    }
            y = loss
        else:
            z = mu 
            #z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            y = self.wv_classifier(z)


        return y, atted

