# from models import RobertaForEventTriggerPrediction
from transformers import AutoTokenizer, AdamW
import torch
from tqdm import tqdm, trange
import random
import numpy as np
import os
import argparse
import time
import logging
from transformers import RobertaConfig
from config import Config
from transformers import RobertaPreTrainedModel
from transformers import RobertaModel
import torch.nn as nn
import torch


from transformers import RobertaPreTrainedModel, RobertaConfig
import torch
import torch.nn as nn


class RobertaForEventTriggerPrediction(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # self.config.num_labels = 2
        self.num_labels = 2
        self.roberta = RobertaModel(self.config, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            # etok_idxs=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        etok_idxs = (input_ids == 50265).nonzero(as_tuple=True)[1]

        sequence_output = self.dropout(sequence_output)
        logits_raw = self.classifier(sequence_output)
        logits = logits_raw[range(len(etok_idxs)), etok_idxs]
        # return logits
        # return logits, logits_raw
        # res_raw = nn.functional.sigmoid(logits)
        res = nn.functional.softmax(logits, dim=-1)

        outputs = (res,) + outputs[2:]  # add hidden states and attention if they are here
        # return outputs
        # return attention_mask

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = logits.view(-1, self.num_labels)
        #         active_labels = torch.where(
        #             active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        #         )
        #         return active_logits, active_labels
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        # return outputs, labels

        return outputs



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHKPT_PATH = Config.event_trigger_path

# tokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
special_tokens_dict = {"additional_special_tokens": ["<E>"]}        # we add as special token <E> here
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print("We have added", num_added_toks, "tokens")

# model
model = RobertaForEventTriggerPrediction.from_pretrained('roberta-base')
model.resize_token_embeddings(len(tokenizer))
chkpt_path = os.path.join(CHKPT_PATH, 'model')
chkpt = torch.load(chkpt_path, map_location='cpu')
model.load_state_dict(chkpt['model'])
model.to(device)
model.eval()
print("Loaded the event trigger predictor!")
def get_event_triggering_score(txt):
    #e.g. We played football and then eat <E>
    if txt[-3:] != "<E>":
        txt = txt.strip(" ") + " <E>"
    input_sentences = [txt]   # list of sentences where <E> denotes the next token to be predicted
    model_input = tokenizer(input_sentences, return_tensors='pt', padding=True).to(device)
    res = model(**model_input)[0][:, 1].detach().cpu().numpy()
    return res[0]

