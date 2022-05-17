import random
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

cr_loss = torch.nn.functional.cross_entropy

def evaluate(tokenizer, model, device, loader, class_break_down=False, model_type="kgbert"):
    # evaluate CSKB Population

    model.eval()

    predicted_scores = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    classes = []

    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            if model_type == "kgbert":
                outputs_logits = model(tokens)

                logits = torch.softmax(outputs_logits, dim=1)            
                values = logits[:, 1]
            elif model_type == "gpt2":
                outputs = model(input_ids = ids, attention_mask = mask, labels=ids)

                shift_logits = outputs[1][..., :-1, :].contiguous().view(-1,outputs[1].size(-1))
                shift_labels = ids[..., 1:].contiguous().view(-1)
                
                losses = cr_loss(shift_logits, shift_labels, 
                    ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)

                losses = torch.div(torch.sum(losses, dim=1), 
                    torch.sum(mask[:, 1:], dim=1)) # (batch_size, ) get the loss after removing PAD_TOKEN

                values = -losses

            predicted_scores = torch.cat((predicted_scores, values))
            labels = torch.cat((labels, y))
            classes.extend(data["clss"])

    if class_break_down:
        # calculate the break-down scores given different classes.
        classes = pd.Series(classes)
        predicted_scores = pd.Series((predicted_scores).tolist())
        labels = pd.Series(labels.tolist())

        clss_scores = {}
        clss_num = {}
        for clss in ["test_set", "cs_head", "all_head"]:
            idx = classes == clss
            clss_num[clss] = sum(idx)
            try:
                clss_scores[clss] = roc_auc_score(labels[idx], predicted_scores[idx])
            except:
                clss_scores[clss] = 0
        return roc_auc_score( labels, predicted_scores ), len(labels), clss_scores, clss_num
    else:
        # return the overall AUC scores, with the number of examples
        return roc_auc_score( labels.tolist(), (predicted_scores).tolist()), len(labels)

def score_triples(tokenizer, model, device, loader, model_type="kgbert"):
    """
        return: predicted_scores (list) The scores predicted by the model.
                for KG-BERT, the returned score is the softmax score for the triple being true.
                    GPT2, the returned score is the negative GPT2 loss.
    """
    model.eval()

    predicted_scores = torch.tensor([]).to(device)

    with torch.no_grad():
        for _, data in tqdm(enumerate(loader, 0)):
            y = data['label'].to(device, dtype=torch.long)
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)

            tokens = {"input_ids":ids, "attention_mask":mask}

            if model_type == "kgbert":
                outputs_logits = model(tokens)

                logits = torch.softmax(outputs_logits, dim=1)            
                values = logits[:, 1]
            elif model_type == "gpt2":
                outputs = model(input_ids = ids, attention_mask = mask, labels=ids)

                shift_logits = outputs[1][..., :-1, :].contiguous().view(-1,outputs[1].size(-1))
                shift_labels = ids[..., 1:].contiguous().view(-1)
                
                losses = cr_loss(shift_logits, shift_labels, 
                    ignore_index=tokenizer.pad_token_id, reduction="none").view(ids.size(0), -1)

                losses = torch.div(torch.sum(losses, dim=1), 
                    torch.sum(mask[:, 1:], dim=1)) # (batch_size, ) get the loss after removing PAD_TOKEN

                values = -losses

            predicted_scores = torch.cat((predicted_scores, values))

    return predicted_scores.tolist()
