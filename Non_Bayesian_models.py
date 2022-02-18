import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel

# Add Two layer Conv1d, first layer [2,3,4] kernal size and 768 Co; the sencond one kernel size=2, Co=128
# Model ID: 2
class HConvBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=1, output_attentions=False, keep_multihead_output=False):
        super(HConvBertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        for params in list(self.bert.parameters())[:133]:
            params.requires_grad = False
        self.conv = nn.ModuleList([nn.Conv1d(config.hidden_size, config.hidden_size, 
                                   kernel_size=f) for f in [2,3,4]])
        self.conv2 = nn.Conv1d(config.hidden_size, 128, kernel_size=2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.fc = nn.Linear(config.hidden_size * 6, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)


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

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
        super(BertForSequenceClassification, self).__init__(config)
        self.output_attentions = output_attentions
        self.num_labels = num_labels
        self.bert = BertModel(config, output_attentions=output_attentions,
                                      keep_multihead_output=keep_multihead_output)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
        if self.output_attentions:
            all_attentions, _, pooled_output = outputs
        else:
            _, pooled_output = outputs
        # pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif self.output_attentions:
            return all_attentions, logits
        return logits

# BERT 12 encoders output is considered as 12 channels of Conv2d
# In Conv2d, Ci = 12, Co = 128, ks = [2,3,4], max_pooling and avg_pooling: 2 
# concatenate: 3 * 2 * 128 = 768, sentence pair representation [cnn output, cls]
# classifier: 2 x 768 --> 1
# Model ID: 3
# class CNN_Text(nn.Module):
#     def __init__(self):
#         super(CNN_Text, self).__init__()
#         D = 768
#         Ci = 12
#         Co = 128
#         Ks = [2,3,4]
#         self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
#         '''
#         self.conv12 = nn.Conv2d(Ci, Co, (2, D))
#         self.conv13 = nn.Conv2d(Ci, Co, (3, D))
#         self.conv14 = nn.Conv2d(Ci, Co, (4, D))
#         '''
#         self.dropout = nn.Dropout(0.1)
    
#     def forward(self, x_in):
#         x = x_in  # (N, Layers, 128, 768)  Requirment: (N, Ci, W, D)
#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
#         max_list = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
#         avg_list = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in x]
#         x = max_list + avg_list
#         x = torch.cat(x, 1)
#         return x # [N, 128*2*3]
        

# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config, num_labels=1, output_attentions=False, keep_multihead_output=False):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.output_attentions = output_attentions
#         self.num_labels = num_labels
#         self.bert = BertModel(config, output_attentions=output_attentions,
#                                       keep_multihead_output=keep_multihead_output)
#         for params in list(self.bert.parameters()):
#             params.requires_grad = False
#         self.conv = CNN_Text()
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)


#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
#         outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, head_mask=head_mask)
#         if self.output_attentions:
#             all_attentions, encoded_layers, pooled_output = outputs
#         else:
#             encoded_layers, pooled_output = outputs

#         # Convert a list of [N, 128, 768] into [N, 12, 128, 768]
#         out = []
#         for i in range(12):
#             x = encoded_layers[i].unsqueeze(1)
#             out.append(x)
#         x = torch.cat(out, dim = 1)
#         # conv layer
#         x = self.conv(x)

#         # y = torch.cat([x, pooled_output], dim = 1)
#         out = self.dropout(x)
#         logits = self.classifier(out)

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         elif self.output_attentions:
#             return all_attentions, logits
        # return logits



# only add fixed bert weight, compared with the original one
# Model ID: 1
# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config, num_labels=2, output_attentions=False, keep_multihead_output=False):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.output_attentions = output_attentions
#         self.num_labels = num_labels
#         self.bert = BertModel(config, output_attentions=output_attentions,
#                                       keep_multihead_output=keep_multihead_output)
#         for params in list(self.bert.parameters()):
#             params.requires_grad = False
        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)


#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, head_mask=None):
#         outputs = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, head_mask=head_mask)
#         if self.output_attentions:
#             all_attentions, _, pooled_output = outputs
#         else:
#             _, pooled_output = outputs
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return loss
#         elif self.output_attentions:
#             return all_attentions, logits
#         return logits