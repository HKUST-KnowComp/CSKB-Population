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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='kgbert_va', type=str, required=False,
                    choices=["graphsage_relational","simple_relational",
                    "kgbert_va", "kgbertsage_va"],
                    help="Name of the model.")
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
                    default="/home/data/tfangaa/CKGP/model_data/models/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge+neighbor/kgbertsage_release/kgbertsage_va_single_cls_trans_best_bert_bs32_opt_ADAM_lr5e-05_decay1.0_500_layer1_neighnum_{neighnum}_graph_ASER_acc_aggfuncMEAN_seed401.pth.step.{step}",
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

with open(graph_cache, "rb") as reader:
    graph_dataset = pickle.load(reader)

infer_file = pd.read_csv(args.evaluation_file_path)

def get_dataset(infer_file):
    
    dataset = {"tst":{"i":[], "head_input":[], "relation_input":[], "tail_input":[],  "votes":[], "relation":[], "class":[], "rev_aser":[]}, 
               "dev":{"i":[], "head_input":[], "relation_input":[], "tail_input":[],  "votes":[], "relation":[], "class":[], "rev_aser":[]}}

    for i, (head, relation, tail, votes, split, clss, rev_aser) in \
        enumerate(zip(infer_file["head"], infer_file["relation"], 
            infer_file["tail"], infer_file["worker_vote"], 
            infer_file["split"], infer_file["class"], infer_file["reverse_aser_edge"])): #

        votes = json.loads(votes)
        
        if len(head.split()) <= MAX_NODE_LENGTH and len(tail.split()) <= MAX_NODE_LENGTH:
            dataset[split]["i"].append(i)
            dataset[split]["head_input"].append(graph_dataset.node2id.get(head, head))
            dataset[split]["tail_input"].append(graph_dataset.node2id.get(tail, tail))
            dataset[split]["relation_input"].append(graph_dataset.rel2id[rel_dict_convert.get(relation, relation)])
            dataset[split]["votes"].append(votes)
            dataset[split]["relation"].append(relation)
            dataset[split]["class"].append(clss)
            dataset[split]["rev_aser"].append(rev_aser)
    return dataset


def get_prediction(dataset):

    with torch.no_grad():
        all_predictions = []
        all_values = []
        all_hids = []
        for i in range(0, len(dataset), test_batch_size):
            batch = dataset[i:min(i+test_batch_size, len(dataset))]
            b_s = len(batch)

            logits = model(batch, b_s) # (batch_size, 2)

            logits = torch.softmax(logits, dim=1)
            values = logits[:, 1]
            _, predicted = torch.max(logits, dim=1)
            predicted = predicted.tolist()
            values = values.tolist()
            all_predictions.extend(predicted)
            all_values.extend(values)
    return all_predictions, all_values

def get_labels(votes_list):
    labels = []
    for votes in votes_list:
        label = 1 if sum(votes) >=3 else 0
        labels.append(label)

    return labels


step_range = range(250, 20001, 250)
steps_dict = dict([(step, dict([(rel, { "auc":0,})  for rel in all_relations]), 
                        ) for step in step_range])





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


###########################################################################
# 1. Select best models on the dev set
###########################################################################

dataset = get_dataset(infer_file)

dataset_dev = pd.DataFrame(dataset["dev"])

for step in tqdm(step_range):

    if "kgbertsage" in model_path:
        model.load_state_dict(torch.load(model_path.format(neighnum=neighnum, step=step)))
    else:
        model.load_state_dict(torch.load(model_path.format(step)))
    model = model.to(device)
    model.eval()

    
    labels = get_labels( dataset_dev["votes"])
    preds, vals = get_prediction( 
        dataset_dev.loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist() 
        )

    for rel in all_relations:
        
        rel_idx = pd.Series(map(lambda x: x == rel, dataset_dev["relation"] ))

        try:
            steps_dict[step][rel]["auc"] = roc_auc_score(np.array(labels)[rel_idx], np.array(vals)[rel_idx])
        except ValueError:
            # Cases where all labels are 1. 
            steps_dict[step][rel]["auc"] = 0

# if not os.path.exists("eval_stat/"):
#     os.mkdir("eval_stat/")
# np.save("eval_stat/"+model_name+str(neighnum)+"dev", steps_dict)

###########################################################################
# 2. Test on dev set
###########################################################################

def calc_test_auc(group_by, rev_aser=False, clss="test_set"):
    """
        group_by: ["relation", "class", "rev_aser"]
                "relation": check auc grouped by relations. Main result.
                "class": check the auc scores divided by different classes of test edges 
                        (CSKB edges, CSKB head + ASER tail, ASER head + ASER tail)
                "rev_aser": Whether the edges are reversed edges in ASER.
    """
    best_test_results = dict([(rel, {"auc":0}
        ) for rel in all_relations])

    group_sum = {}

    for rel in all_relations:
        
        if group_by == "relation":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"] )))
        elif group_by == "class":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"] ))) \
            & (pd.Series(map(lambda x: x == clss, dataset_tst["class"] )))
        elif group_by == "rev_aser":
            rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"] ))) \
            & (pd.Series(map(lambda x: x == rev_aser, dataset_tst["rev_aser"] ))) \
            & (pd.Series(map(lambda x: x != "test_set", dataset_tst["class"] )))


        group_sum[rel] = sum(rel_idx)

        labels = dataset_tst[rel_idx]["final_label"]

        vals = dataset_tst.loc[rel_idx, "prediction_value"]

        try:
            best_test_results[rel]["auc"] = roc_auc_score(labels, vals)
        except:
            best_test_results[rel]["auc"] = 0
    return best_test_results, group_sum


dataset_tst = pd.DataFrame(dataset["tst"])
dataset_tst.insert(len(dataset_tst.columns), "prediction_value", np.zeros((len(dataset_tst), 1)))
dataset_tst.insert(len(dataset_tst.columns), "final_label", np.zeros((len(dataset_tst), 1), dtype=np.int64))


group_sum = {}
# write results on the csv
for rel in all_relations:
    if steps_dict[step_range[0]][rel]["auc"] == 0:
        best_step = max(step_range)
    else:
        best_step, best_dev_auc = sorted([(step, steps_dict[step][rel]["auc"]) for step in steps_dict],
                                     key=lambda x:x[1], reverse=True)[0]
    if "kgbertsage" in model_path:
        model.load_state_dict(torch.load(model_path.format(neighnum=neighnum, step=step)))
    else:
        model.load_state_dict(torch.load(model_path.format(step)))

    model.eval()
    

    rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"] )))

    group_sum[rel] = sum(rel_idx)

    labels = get_labels( dataset_tst[rel_idx]["votes"])

    preds, vals = get_prediction( 
        dataset_tst[rel_idx].loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist() 
        )
    dataset_tst.loc[rel_idx, "prediction_value"] = vals
    dataset_tst.loc[rel_idx, "final_label"] = labels

# dataset_tst.loc[:, ["votes", "relation", "class", "rev_aser", "prediction_value",  "final_label" ]].\
#     to_csv("eval_stat/"+model_name+str(neighnum)+"tst.csv", index=False)

# 1. get scores grouped by different relations
best_test_results, group_sum = calc_test_auc("relation")

total_test_tuple = sum(group_sum.values())
print("avg auc", sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum]) )
print(" & ".join( [str(round(best_test_results[g]['auc'], 3)) for g in best_test_results]))


# 2. Check the overall AUC for different classes
print("2. Check the overall AUC for different classes")

class_types = ["test_set", "cs_head", "all_head"]


for clss in class_types:
    best_test_results, group_sum = calc_test_auc("class", clss=clss)
    total_test_tuple = sum(group_sum.values())
    print("class", clss, round(sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum]), 3))

# 3. Check the performance of ASER edge / Reversed ASER edge. CSKB test set are excluded
print("3. Check the performance of ASER edge / Reversed ASER edge. CSKB test set are excluded")

for rev_aser in [True, False]:
    best_test_results, group_sum = calc_test_auc("rev_aser", rev_aser=rev_aser)
    total_test_tuple = sum(group_sum.values())
    print("rev aser", rev_aser, round(sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum]), 3))

