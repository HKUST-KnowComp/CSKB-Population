import warnings
warnings.filterwarnings("ignore")
import torch
import os
import sys
sys.path.append("../model/BertSAGE")
from dataloader import *
from model import *
from utils import get_logger
import argparse
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import json
import pandas as pd
import math
from functools import partial

import argparse

rel_dict_convert = {"gReact":"general React",
                    "gEffect":"general Effect",
                    "gWant":"general Want"}

r = "HinderedBy"

test_batch_size = 512
MAX_NODE_LENGTH=10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

r_id = graph_dataset.rel2id[rel_dict_convert.get(r, r)]
candidates = np.load("../../data/DISCOS_infer_candidates_filter.npy", allow_pickle=True)[()]
# step = 32250 # kgbert trained on all
step = 22000 # kgbert trained on atomic

neighnum = 4

def truncate(node):
    strs = node.split()
    if len(strs) <= MAX_NODE_LENGTH:
        return node
    else:
        return " ".join(strs[:MAX_NODE_LENGTH])


def infer(infer_list,
         r_id,
         model,
         test_batch_size,
         device,
         mode="test",
         get_accuracy=False):

    model.eval()
    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(infer_list), test_batch_size)):
            edges = infer_list[i:i+test_batch_size]
            edges = [[truncate(head), truncate(tail), r_id] for head, tail in edges]

            # allocate to right device

            logits = model(edges, len(edges))  # (batch_size, 2)
            logits = torch.softmax(logits, dim=1)
            values = logits[:, 1]
            scores.extend(values.cpu().tolist())

    return scores


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

atomic_rels = ['oEffect', 'xEffect', 'oWant','xWant', 
               'oReact', 'xReact', 'xAttr',  'xNeed', 'xIntent',
              'isBefore', 'isAfter', 'HinderedBy']

# for r in atomic_rels:

model.load_state_dict(torch.load(model_path.format(step)))
model = model.to(device)
model.eval()

scores = infer(candidates[r],
    r_id,
     model,
     test_batch_size,
     device)

# np.save("../../data/DISCOS_inference/scores_{}".format(r), scores)

df_triple_scores = pd.DataFrame({"head":[h for h, _ in candidates[r]], "tail":[t for _, t in candidates[r]], "score":scores})
df_triple_scores.to_csv("../../data/DISCOS_inference/scores_{}.csv".format(r), index=False)





