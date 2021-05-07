import numpy as numpy
import pandas as pd 
from transformers import AutoModel, BertPreTrainedModel
import torch
import torch.nn as nn 
import torch.nn.functional as F
import label_attention_layer as label_attention


d_l = 2 # number class
d_proj = 64

if torch.cuda.is_available():
  device = torch.device('cuda')
  # print(torch.cuda.get_device_name())
else:
  device = torch.device('cpu')
#label attention
attention = label_attention.LabelAttentionEncoder(d_l=d_l, d_proj=d_proj)

########### Model Basic Bert #############
class BertBase(BertPreTrainedModel):
    def __init__(self, conf):
        super(BertBase, self).__init__(conf)
        self.config = conf
        self.num_labels = conf.num_labels
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, config=self.config)

        self.out = nn.Linear(self.config.hidden_size*2, self.num_labels)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, _, hidden_outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = False
        )

        avg_pool = torch.mean(sequence_output, 1)
        max_pool, _ = torch.max(sequence_output, 1)
        h_conc = torch.cat((max_pool, avg_pool), 1)
        output = self.dropout(h_conc)
        logits = self.out(output)

        return torch.sigmoid(logits)


########## Model Bert+CNN+label attention ######
# get 4 hidden state in transformer + CNN, concat label attention in after CNN
class BertCNN(BertPreTrainedModel):
    def __init__(self, conf, MODEL_NAME, drop_out, lb_availible, max_len, max_len_tfidf, kernel_sizes):
        super(BertCNN, self).__init__(conf)
        self.config = conf
        self.lb_availible = lb_availible
        self.num_labels = conf.num_labels
        self.backbone = AutoModel.from_pretrained(MODEL_NAME, config=self.config)

        self.convs = nn.ModuleList([nn.Conv1d(max_len+max_len_tfidf, 256, kernel_size) for kernel_size in kernel_sizes])
        self.dropout = nn.Dropout(drop_out)
        self.out = nn.Linear(self.config.hidden_size, self.num_labels)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        sequence_output, _, hidden_outputs = self.backbone(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            return_dict = False
        )
        # sequence_output = torch.stack([hidden_outputs[-1], hidden_outputs[-2], hidden_outputs[-3]])
        sequence_output = torch.stack([hidden_outputs[-1]])
        sequence_output = torch.mean(sequence_output, dim=0)

        ## label attention
        if self.lb_availible == True:
          lb = attention(sequence_output.to('cpu'), attention_mask.to('cpu')).to(device)
          sequence_output = torch.cat((sequence_output, lb), axis=2)
        ## CNN
        cnn = [F.relu(conv(sequence_output)) for conv in self.convs]
        max_pooling = []
        for i in cnn:
          max, _ = torch.max(i, 2)
          max_pooling.append(max)
        output = torch.cat(max_pooling, 1)
        
        output = self.dropout(output)
        logits = self.out(output)

        return torch.sigmoid(logits)
