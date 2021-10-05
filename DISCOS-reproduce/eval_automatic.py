import warnings
warnings.filterwarnings("ignore")
import torch
import os
from dataloader import *
from model import *
from utils import get_logger
import argparse
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import json
import pandas as pd
import math
from functools import partial

import argparse

def eval(data_loader,
         model,
         test_batch_size,
         device,
         mode="test",):
    pred_y = []
    gt_y = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader.get_batch(batch_size=test_batch_size, mode=mode)):
            edges, labels = batch
            # allocate to right device
            edges = edges.to(device)
            labels = labels.to(device)

            logits = model(edges, edges.shape[0])  # (batch_size, 2)
            predicted = torch.max(logits, dim=1)[1]

            pred_y.extend(predicted.cpu().tolist())
            gt_y.extend(labels.cpu().tolist())

    
    report = classification_report(gt_y, pred_y, output_dict=True)
    return report["accuracy"], report["macro avg"]["f1-score"]


test_batch_size = 512
MAX_NODE_LENGTH=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_relations = [
        "xWant", "oWant", "general Want",
        "xEffect", "oEffect", "general Effect",
        "xReact", "oReact", "general React",
        "xAttr",
        "xIntent",
        "xNeed",
        "Causes", "xReason",
        "isBefore", "isAfter",
        'HinderedBy',
        'HasSubEvent',
    ]
rel_dict_convert = {"gReact":"general React",
                    "gEffect":"general Effect",
                    "gWant":"general Want"}

# train all, eval all
# graph_cache = "../../data/graph_cache/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge.pickle"
# trained on ATOMIC, test set evaled on ATOMIC
graph_cache = "../../data/graph_cache/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-atomic-relations_rel_in_edge.pickle"

model_name = "kgbert_va"

# train all
# model_path = "../../data/models/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge/kgbert_va_single_cls_trans_best_bert_bs32_opt_ADAM_lr5e-05_decay1.0_500_f1_seed401.pth.step.{}"
# train atomic
model_path = "../../data/models/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-atomic-relations_rel_in_edge/kgbert_va_single_cls_trans_best_bert_bs32_opt_ADAM_lr5e-05_decay1.0_500_f1_seed401.pth.step.{}"

with open(graph_cache, "rb") as reader:
    graph_dataset = pickle.load(reader)


neighnum = 4

if "simple" in model_name:
    model = SimpleClassifier(encoder="bert",
                             adj_lists=graph_dataset.get_adj_list(),
                             nodes_tokenized=graph_dataset.get_nodes_tokenized(),
                             nodes_text=graph_dataset.get_nid2text(),
                             device=device,
                             enc_style="single_cls_trans",
                             num_class=2,
                             include_rel="relational" in model_name,
                             relation_tokenized=graph_dataset.get_relations_tokenized()
                             )
elif 'graphsage' in model_name:
    model = LinkPrediction(encoder="bert",
                           adj_lists=graph_dataset.get_adj_list(),
                           nodes_tokenized=graph_dataset.get_nodes_tokenized(),
                           id2node=graph_dataset.get_nid2text(),
                           device=device,
                           num_layers=1,
                           num_neighbor_samples=4,
                           enc_style="single_cls_trans",
                           agg_func="MEAN",
                           num_class=2,
                           include_rel="relational" in model_name,
                           relation_tokenized=graph_dataset.get_relations_tokenized()
                           )
elif 'kgbert' in model_name:
    model = KGBertClassifier(encoder="bert",
            adj_lists=graph_dataset.get_adj_list() if "sage" in model_name else None,
            nodes_tokenized=graph_dataset.get_nodes_tokenized(),
            relation_tokenized=graph_dataset.get_relations_tokenized(),
            id2node=graph_dataset.get_nid2text(),
            enc_style="single_cls_trans",
            agg_func="MEAN",
            num_neighbor_samples=neighnum,
            device=device,
            version=model_name)

test_acc_dict = dict([(step, 0) for step in range(250, 56251, 250)])

for step in range(250, 3971, 250):
    model.load_state_dict(torch.load(model_path.format(step)))
    model = model.to(device)
    model.eval()

    acc, _ = eval(graph_dataset,
         model,
         test_batch_size,
         device,
         mode="test")
    test_acc_dict[step] = acc
