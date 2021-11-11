import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import math
from .miscLayer import BERT_Embedding, SingleHeadAttention

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
    def __init__(self, n_hidden, n_classes):
        super().__init__()
        self.layer_output = torch.nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        out = self.layer_output(x)
        return out


class HANsentEncoder(nn.Module):
    def __init__(self, config):
        super(HANsentEncoder, self).__init__()
        bert_dim = 768
        self.sent_level_att = BERT_Mapping_mapping(bert_dim)

    def forward(self, x):
        return self.sent_level_att(x)

        


class HANvae(nn.Module):
    def __init__(self, vocab_dim, config):
        super(HANvae, self).__init__()
        default_config = {}


        ntopics = 50
        self.bert_embedding = BERT_Embedding(config)
        self.n_classes = len(config['TARGET'].get('labels'))
        bert_dim = 768
        self.bert_dim = bert_dim
        hidden_dim = 300

        #self.sent_encoder = HANsentEncoder(config)

        self.loss_weight_lambda = float(math.ceil(vocab_dim/self.n_classes))*10



        self.doc_level_att = BERT_Mapping_mapping(bert_dim)

        self.wv_hidden = WVHidden(bert_dim, hidden_dim)
        self.wv_classifier = WVClassifier(ntopics, self.n_classes)

        self.mu = nn.Linear(hidden_dim, ntopics)
        self.log_sigma = nn.Linear(hidden_dim, ntopics)
        self.h_to_z = Identity()
        self.topics = Topics(ntopics, vocab_dim) # decoder
        self.reset_parameters()


    def forward(self,x, mask=None, n_samples=1, bow=None, train=False, true_y=None, pre_embd=False):
        input_shape = x.shape
        num_batch = input_shape[0]
        num_sent = input_shape[1]
        num_words = input_shape[2]
        if pre_embd:
            bert_rep = x
        else:
            reshape_for_embed = x.reshape(num_batch*num_sent, num_words)
            bert_rep = self.bert_embedding(reshape_for_embed, mask)
            bert_rep = bert_rep[0]
            bert_rep = bert_rep.reshape(num_batch, num_sent, num_words, self.bert_dim)
            #sent_level_att = bert_rep[:,0]
            #sent_level_att_reshaped = sent_level_att.reshape(num_batch, num_sent, self.bert_dim)

        #sent_level_att = self.sent_encoder(bert_rep)
        sent_level_att = bert_rep[:,:,0]
        #sent_level_att_reshaped = sent_level_att.reshape(num_batch, num_sent, self.bert_dim)

        atted = self.doc_level_att(sent_level_att)
        hidden = self.wv_hidden(atted)

        mu = self.mu(hidden)
        log_sigma = self.log_sigma(hidden)
        if train:
            kld = kld_normal(mu, log_sigma)
            rec_loss = 0
            class_loss = 0
            for i in range(n_samples):
                z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
                z = self.h_to_z(z)
                log_y_hat = torch.log_softmax(self.wv_classifier(z), dim=-1)
                log_prob = self.topics(z)
                rec_loss = rec_loss - (log_prob * bow).sum(dim=-1)
                class_loss = class_loss - (log_y_hat * true_y).sum(dim=-1)
            rec_loss = rec_loss / n_samples
            class_loss = class_loss / n_samples


            minus_elbo = rec_loss + kld + self.loss_weight_lambda*class_loss
            minus_elbo = minus_elbo.sum()
            #minus_elbo = minus_elbo.mean()
            total_loss = minus_elbo

            y = {
                'loss': total_loss,
                'minus_elbo': minus_elbo,
                'rec_loss': rec_loss,
                'kld': kld,
                'cls_loss': class_loss
            }

        else:
            y = self.wv_classifier(mu)
            #y = out

        return y, atted


    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


    def get_topics(self):
        return self.topics.get_topics()
