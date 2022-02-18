import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from blitz.modules import BayesianLinear, BayesianConv1d
from blitz.utils import variational_estimator
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

@variational_estimator
class BBNHConvBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=1, output_attentions=False, keep_multihead_output=False):
        super(BBNHConvBertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        # for params in list(self.bert.parameters())[:133]:
        #     params.requires_grad = False

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

        self.conv = nn.ModuleList([BayesianConv1d(config.hidden_size, config.hidden_size, 
                                   kernel_size=f, prior_sigma_1 = 0.1,
                                   prior_sigma_2 = 0.02,
                                   prior_pi = 1,
                                   posterior_mu_init = 0,
                                   posterior_rho_init = -7.0,) for f in [2,3,4]])                 
        self.conv2 = BayesianConv1d(config.hidden_size, 128, kernel_size=2, prior_sigma_1 = 0.1,
                 prior_sigma_2 = 0.02,
                 prior_pi = 1,
                 posterior_mu_init = 0,
                 posterior_rho_init = -7.0)
        self.classifier = BayesianLinear(config.hidden_size, num_labels)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, encoded_layers, pooled_output = outputs
        else:
            encoded_layers, pooled_output = outputs
        
        # conv layer
        x_in = encoded_layers.transpose(1, 2)  # if not channel first

        # Conv outputs
        out = []
        for i in range(3):
            y = self.conv[i](x_in)
            z = self.conv2(y)
            z_max = F.max_pool1d(z, z.size(2)).squeeze(2)
            z_avg = F.avg_pool1d(z, z.size(2)).squeeze(2)
            out.append(z_avg)
            out.append(z_max)
        # Concat conv outputs
        z = torch.cat(out, 1)  # * 2(ngram[2,3]) * 2(avg, max)
        x = self.dropout(z)

        logits = self.classifier(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits


@variational_estimator
class BBNBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BBNBertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.classifier = BayesianLinear(config.hidden_size, num_labels)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits