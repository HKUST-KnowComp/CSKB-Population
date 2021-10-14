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
from sklearn.metrics import roc_auc_score
import json
import pandas as pd
import math
from functools import partial
from evaluate_utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='kgbert_va', type=str, required=False,
                    choices=["graphsage_relational","simple_relational",
                    "kgbert_va", "kgbertsage_va"],
                    help="Name of the model.")
parser.add_argument("--encoder", default='bert', type=str, required=False,
                    choices=["bert", "bert_large", "roberta", "roberta_large"],
                    help="Pretrained encoder.")
parser.add_argument("--graph_cache_path", 
                    default="data/graph_cache/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge.pickle", 
                    type=str, required=False,
                    help="Path to the graph cache.")
parser.add_argument("--node_token_path", 
                    default="", 
                    type=str, required=False,
                    help="Path to the node_tokens.")
parser.add_argument("--neigh_num", 
                    default=4, 
                    type=int, required=False,
                    help="Number of neighbors that are aggregated in KGBertSAGE or BertSAGE.")
parser.add_argument("--model_path", 
                    # default="data/models/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge/kgbertsage_va_single_cls_trans_best_bert_bs32_opt_ADAM_lr5e-05_decay1.0_500_layer1_neighnum_{neighnum}_graph_ASER_f1_aggfuncMEAN_seed401.pth.step.{step}", 
                    default="data/models/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge/kgbert_va_single_cls_trans_best_bert_bs32_opt_ADAM_lr1e-06_decay1.0_500_f1_human_annotation_seed401.pth",
                    type=str, required=False,
                    help="Path to the saved models.")
parser.add_argument("--evaluation_file_path", 
                    default="data/evaluation_set.csv", 
                    type=str, required=False,
                    help="Path to the evaluation set csv.")

args = parser.parse_args()
test_batch_size = 128
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

graph_cache = args.graph_cache_path
node_token_path = args.node_token_path
model_name = args.model_name
neighnum = args.neigh_num
model_path = args.model_path
encoder = args.encoder

with open(graph_cache, "rb") as reader:
    graph_dataset = pickle.load(reader)

infer_file = pd.read_csv(args.evaluation_file_path)


step_range = range(250, 34751, 250)
steps_dict = dict([(step, dict([(rel, { "auc":0,})  for rel in all_relations]), 
                        ) for step in step_range])





if "simple" in model_name:
    model = SimpleClassifier(encoder=encoder,
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
    model = LinkPrediction(encoder=encoder,
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
    model = KGBertClassifier(encoder=encoder,
            adj_lists=graph_dataset.get_adj_list() if "sage" in model_name else None,
            nodes_tokenized=graph_dataset.get_nodes_tokenized(),
            relation_tokenized=graph_dataset.get_relations_tokenized(),
            id2node=graph_dataset.get_nid2text(),
            enc_style="single_cls_trans",
            agg_func="MEAN",
            num_neighbor_samples=neighnum,
            device=device,
            version=model_name)

model.load_state_dict(torch.load(model_path))
model = model.to(device)

###########################################################################
# 1. Select best models on the dev set
###########################################################################

dataset = get_dataset(graph_dataset, infer_file)

dataset_dev = pd.DataFrame(dataset["dev"])

val_auc = get_val_auc(model, dataset_dev)

print("validation auc:", val_auc)

###########################################################################
# 2. Test on dev set
###########################################################################



dataset_tst = pd.DataFrame(dataset["tst"])
dataset_tst.insert(len(dataset_tst.columns), "prediction_value", np.zeros((len(dataset_tst), 1)))
dataset_tst.insert(len(dataset_tst.columns), "final_label", np.zeros((len(dataset_tst), 1), dtype=np.int64))

test_auc, relation_break_down_auc, main_result_auc = get_test_auc_scores(model, dataset_tst)
print("relational break down: " + relation_break_down_auc)
print("class break down: All / Ori Test Set / CSKB head + ASER tail / ASER edges")
print(main_result_auc)