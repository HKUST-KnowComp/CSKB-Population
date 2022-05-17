import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, AutoModel, PretrainedConfig)

class KGBERTClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.model_type = self.model.config.model_type
        
        try:
            self.emb_size = self.model.config.d_model # bart
        except:
            self.emb_size = self.model.config.hidden_size # roberta/bert

        self.linear = nn.Linear(self.emb_size, 2)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(tokens['input_ids'],
                             attention_mask=tokens['attention_mask'])

        if self.model_type == "bart":
            # embedding of [EOS] in the decoder
            eos_mask = tokens['input_ids'].eq(self.model.config.eos_token_id)

            if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            sentence_representation = outputs[0][eos_mask, :].view(outputs[0].size(0), -1, outputs[0].size(-1))[
                :, -1, :
            ]
        else:
            # embedding of the [CLS] tokens
            sentence_representation = outputs[0][:, 0, :]
        
        return sentence_representation

    def forward(self, tokens):
        """
            tokens: 
        """
        
        embs = self.get_lm_embedding(tokens) # (batch_size, emb_size)

        logits = self.linear(embs) # (batch_size, 2)

        return logits