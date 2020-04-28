import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math

class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        #print('hey')
        #print(self.topic.weight)
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


class NVDM(nn.Module):
    def __init__(self, cust_config={}):
        super(NVDM, self).__init__()
        default_config = {}
        default_config['ntopics'] = 50
        default_config['vocab_dim'] = 1994
        default_config['hidden_him'] = 500
        default_config['n_samples'] = 1
        default_config['n_classes'] = 12

        config = dict(list(default_config.items()) + list(cust_config.items()))

        ntopics = config['ntopics']
        vocab_dim = config['vocab_dim']
        hidden_him = config['hidden_him']
        n_classes = config['n_classes']


        self.n_samples = default_config['n_samples']
        self.hidden = nn.Linear(vocab_dim, hidden_him)
        self.nonlin = torch.nn.Tanh()

        self.mu = nn.Linear(hidden_him, ntopics)
        self.log_sigma = nn.Linear(hidden_him, ntopics)
        self.h_to_z = Identity()
        self.topics = Topics(ntopics, vocab_dim) # decoder
        self.layer_output = torch.nn.Linear(hidden_dim, n_classes)
        self.reset_parameters()


    def forward(self,x):
        h = self.hidden(x)
        h = self.nonlin(h)

        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        kld = kld_normal(mu, log_sigma)
        rec_loss = 0
        for i in range(self.n_samples):
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            z = self.h_to_z(z)
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        rec_loss = rec_loss / self.n_samples

        minus_elbo = rec_loss + kld

        out = self.layer_output(h)

        return {
            'loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld,
            'output': out
        }


    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


    def get_topics(self):
        return self.topics.get_topics()
