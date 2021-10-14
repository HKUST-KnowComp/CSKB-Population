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

def get_dataset(graph_dataset, infer_file):
    
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


def get_prediction(model, dataset):

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

def get_val_auc(model, dataset_dev):
    model.eval()
    score_by_rel = {}
    labels = get_labels(dataset_dev["votes"])
    preds, vals = get_prediction( 
        model,
        dataset_dev.loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist() 
        )

    for rel in all_relations:
        
        rel_idx = pd.Series(map(lambda x: x == rel, dataset_dev["relation"] ))

        try:
            score_by_rel[rel] = sum(rel_idx) / len(dataset_dev["relation"]) * roc_auc_score(np.array(labels)[rel_idx], np.array(vals)[rel_idx])
        except ValueError:
            score_by_rel[rel] = 0
    return sum([score_by_rel[rel] for rel in all_relations])

def get_test_auc_scores(model, dataset_tst):
    model.eval()

    group_sum = {}
    # write results on the csv
    for rel in all_relations:

        rel_idx = (pd.Series(map(lambda x: x == rel, dataset_tst["relation"] )))

        group_sum[rel] = sum(rel_idx)

        labels = get_labels( dataset_tst[rel_idx]["votes"])

        preds, vals = get_prediction( 
            model,
            dataset_tst[rel_idx].loc[:, ["head_input", "tail_input", "relation_input"]].values.tolist() 
            )
        dataset_tst.loc[rel_idx, "prediction_value"] = vals
        dataset_tst.loc[rel_idx, "final_label"] = labels
    best_test_results, group_sum = calc_test_auc(dataset_tst, "relation")

    total_test_tuple = sum(group_sum.values())
    auc = sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum]) 
    relation_break_down_auc = " & ".join( [str(round(best_test_results[g]['auc']*100, 1)) for g in best_test_results])

    class_types = ["test_set", "cs_head", "all_head"]
    
    class_scores = {}

    for clss in class_types:
        best_test_results, group_sum = calc_test_auc(dataset_tst, "class", clss=clss)
        total_test_tuple = sum(group_sum.values())
        # print("class", clss, round(sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum]), 3))
        class_scores[clss] = round(sum([group_sum[g] / total_test_tuple * best_test_results[g]["auc"]  for g in group_sum])*100, 1)
    main_result_auc = " & ".join([str(round(auc*100, 1))] + [str(class_scores[clss]) for clss in class_types])
    return auc, relation_break_down_auc, main_result_auc

def calc_test_auc(dataset_tst, group_by, rev_aser=False, clss="test_set"):
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


