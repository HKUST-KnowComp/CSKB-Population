import os
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from copy import deepcopy
from itertools import chain
import pickle

from transformers import BertTokenizer, RobertaTokenizer
from tqdm import tqdm

# debug
import time

MAX_NODE_LENGTH=10
# for xWant ,neg_prop=4, all_in_aser, when this equals 10, filter out 0.6%. when 11, filter out 0.1%.
# Commonsense relationships
CS_RELATIONS_2NL = {
    "oEffect" : "then, PersonY will",
    "xEffect" : "then, PersonX will",
    "general Effect" : "then, other people or things will",
    "oWant" : "then, PersonY wants to ",
    "xWant" : "then, PersonX wants to",
    "general Want" : "then, other people or things want to",
    "oReact" : "then, PersonY feels",
    "xReact" : "then, PersonX feels",
    "general React" : "then, other people or things feel",
    "xAttr" : "PersonX is seen as",
    "xNeed" : "but before, PersonX needed",
    "xIntent" : "because PersonX wanted",
    "isBefore" : "happens before",
    "isAfter" : "happens after",
    "HinderedBy" : "can be hindered by",
    "xReason" : "because",
    "Causes" : "causes ",
    "HasSubEvent" : "includes the event or action",
}

CS_RELATIONS = {
    "all": ['xAttr', 'xReact', 'xWant', 'xEffect', 'xNeed', 'oWant', 'oReact',
            'xIntent', 'oEffect', 'HinderedBy', 'Causes', 'isBefore', 'isAfter',
            'general Effect', 'HasSubEvent', 'general Want', 'general React',
            'xReason'],
    "atomic": ['xAttr', 'xReact', 'xWant', 'xEffect', 'xNeed', 'oWant', 'oReact',
               'xIntent', 'oEffect', 'HinderedBy', 'isBefore', 'isAfter'],
    "glucose": ['xAttr', 'xReact', 'xWant', 'xEffect', 'oWant', 'oReact', 'xIntent',
                'oEffect', 'Causes', 'general Effect', 'general Want', 'general React'],
    "cn": ['Causes', 'xReason', 'HasSubEvent'],
}

ASER_RELATIONS = ['Co_Occurrence', 'Conjunction', 'Contrast', 'Synchronous',
                  'Condition', 'Reason', 'Result', 'Precedence', 'Succession',
                  'Alternative', 'Concession', 'Restatement', 'ChosenAlternative',
                  'Exception', 'Instantiation']

np.random.seed(229)


class MultiGraphDataset():
    """A dataset class for multi task setting: loads all relations.
    """

    def __init__(self, graph_data_path,
        device,
        encoder,
        split=[0.8, 0.1, 0.1],
        max_train_num=np.inf,
        target_relation="all",
        target_dataset="all",
        eval_dataset="all",
        edge_include_rel=False,
        neighbor_include_rel=False,
        load_edge_types="ASER",
        negative_sample="prepared_neg",
        neg_prop=1.0,
        highest_aser_rel=False,
        use_nl_rel=False):
        """
        graph_data_path: the new graph
        device: cuda device
        encoder: Bert or RoBerta
        split: train, dev, test split
        max_train_num: max number of train examples
        target_relation: whether you want specific relation graph or not.
            'all' or any of the commonsense relations.
        edge_include_rel: would the edges include relation data or not.
            see _make_edges() function for usage.
            if True, edge = [head, tail, relation]  # used for kgbert_va model
            if False, edge = [head, tail]
        load_edge_types controls the edges to be loaded to self.adj_list
        negative_sample: the way of negative sampling
            **This is deprecated with the new version of the graph. Always using
            prepared negative samples only.
        neg_pop: the ratio [neg: pos] of negative sampling
            **This is deprecated with the new version of the graph. Always using
            all the prepared negative samples only.
        highest_aser_rel: whether to use select the aser relation with the highest weight
        use_nl_rel: whether to use natural language to encode relations.
        """
        del negative_sample, neg_prop  # unused (deprecated)
        assert load_edge_types in ["CS", "ASER", "CS+ASER"], \
            "should be in [\"CS\", \"ASER\", \"CS+ASER\"]"

        # set up specific evaluate dataset only for all relation training
        eval_dataset = eval_dataset if target_dataset == "all" else target_dataset
        # 1. Load graph
        G = nx.read_gpickle(graph_data_path)

        print("dataset statistics:\nnumber of nodes:{}\nnumber of edges:{}\n".format(len(G.nodes()), len(G.edges())))

        # filter extra large nodes
        filter_nodes = []
        for node in G.nodes():
            if len(node.split()) > MAX_NODE_LENGTH:
                filter_nodes.append(node)
        print("num of removing nodes:", len(filter_nodes))
        G.remove_nodes_from(filter_nodes)
        # filter empty strings
        if "" in G:
            G.remove_node("")

        # some encoder dictionaries
        self.rel2id = {rel: i for i, rel in enumerate(CS_RELATIONS["all"], 1)}
        self.id2rel = dict(enumerate(CS_RELATIONS["all"], 1))
        if neighbor_include_rel and load_edge_types != "CS":
            self.rel2id.update({rel: i for i, rel in enumerate(ASER_RELATIONS, len(CS_RELATIONS["all"]) + 1)})
            self.id2rel.update(dict(enumerate(ASER_RELATIONS, len(CS_RELATIONS["all"]) + 1)))

        self.node2id = {node: i for i, node in enumerate(G.nodes())}
        self.id2node = dict(enumerate(G.nodes()))

        # 2. Prepare training and testing edges
        all_edges_shuffle = list(G.edges.data())
        np.random.shuffle(all_edges_shuffle)

        # get the positive (commonsense) edges first: all or only target relation
        positive_edges = {
            "trn": [],
            "dev": [],
            "tst": [],
        }
        positive_labels = {
            "trn": [],
            "dev": [],
            "tst": [],
        }
        for head, tail, feat in all_edges_shuffle:
            if isinstance(feat["relation"], dict):  # ASER + some negative edges
                continue
            feat_dataset = "atomic" if feat["dataset"] == "atomic2020" else feat["dataset"]
            if target_dataset not in ("all", feat_dataset):
                continue  # not the target commonsense dataset
            for rel in feat["relation"]:
                if rel.startswith("neg_"):  # negative relation
                    continue
                if target_relation not in ("all", rel):
                    continue  # not the target relation
                if feat["split"] == "trn":
                    if target_dataset != "all" and rel not in CS_RELATIONS[target_dataset]:
                        continue
                else:
                    if eval_dataset != "all" and rel not in CS_RELATIONS[eval_dataset]:
                        continue
                if edge_include_rel:
                    edge = [self.node2id[head], self.node2id[tail], self.rel2id[rel]]
                    positive_labels[feat["split"]].append(1)
                else:
                    edge = [self.node2id[head], self.node2id[tail]]
                    label = self.rel2id[rel] if target_relation == "all" else 1
                    positive_labels[feat["split"]].append(label)

                positive_edges[feat["split"]].append(edge)


        print('Number of positive training examples:{}, validating:{}, testing:{}'.format(
            len(positive_edges["trn"]), len(positive_edges["dev"]), len(positive_edges["tst"])))

        # adjust data size if there are too many examples
        if len(positive_edges["trn"]) > max_train_num:
            for i, mode in enumerate(["trn", "dev", "tst"]):
                trim_idx = np.random.permutation(len(positive_edges[mode]))[:int(max_train_num * split[i]/split[0])]
                positive_edges[mode] = list(np.array(positive_edges[mode])[trim_idx])
                positive_labels[mode] = list(np.array(positive_labels[mode])[trim_idx])
            print('Number of positive training examples after trucating:{}, validating:{}, testing:{}'.format(
                len(positive_labels["trn"]), len(positive_labels["dev"]), len(positive_labels["tst"])))

        # 3. Prepare negative edges: use the prepared negative edges
        negative_edges = {
            "trn": [],
            "dev": [],
            "tst": [],
        }
        for head, tail, feat in all_edges_shuffle:
            if "dataset" not in feat:  # pure ASER edges
                continue
            feat_dataset = "atomic" if feat["dataset"] == "atomic2020" else feat["dataset"]
            if target_dataset not in ("all", feat_dataset):
                continue  # not the target commonsense dataset
            
            # NOTE: consider this for ASER / negative overlap edges: they do
            # not have "split"
            if "split" not in feat:
                # NOTE: this operations actually changes all_edges_shuffle
                feat["split"] = np.random.choice(["trn", "dev", "tst"], p=split)
            for rel in feat["relation"]:
                if not rel.startswith("neg_"):  # positive relation
                    continue
                rel = rel[4:]  # remove "neg_"
                if target_relation not in ("all", rel):
                    continue  # not the target relation
                if feat["split"] == "trn":
                    if target_dataset != "all" and rel not in CS_RELATIONS[target_dataset]:
                        continue
                else:
                    if eval_dataset != "all" and rel not in CS_RELATIONS[eval_dataset]:
                        continue
                if not edge_include_rel:
                    edge = [self.node2id[head], self.node2id[tail]]
                else:
                    edge = [self.node2id[head], self.node2id[tail], self.rel2id[rel]]

                negative_edges[feat["split"]].append(edge)

        print("num of prepared neg for train:{}, dev:{}, test:{}".format(
            len(negative_edges["trn"]), len(negative_edges["dev"]), len(negative_edges["tst"])))
        # negative labels == all 0
        negative_labels = {
            "trn": [0] * len(negative_edges["trn"]),
            "dev": [0] * len(negative_edges["dev"]),
            "tst": [0] * len(negative_edges["tst"]),
        }

        # NOTE: this function is deprecated with the new graph: left here for 
        # the record
        '''
        def _sample_negative_edges(G, negative_sample, neg_prop):
            # create edge dictionary of all edges: used for negative sampling.
            # Sample negative edge such that is not included in this edge_dict
            edge_dict = {}
            for head, tail, feat in G.edges.data():
                atomic_rel = list(set(ATOMIC_RELATIONS) & set(feat["relation"]))
                edge = (self.node2id[head], self.node2id[tail])
                if edge not in edge_dict:
                    edge_dict[edge] = atomic_rel
                else:
                    edge_dict[edge].extend(atomic_rel)

            if negative_sample == "fix_head":
                # bipartite graph
                all_heads = [self.node2id[node] for node, out_degree in G.out_degree if out_degree>0]
                all_tails = [self.node2id[node] for node, out_degree in G.out_degree if out_degree==0]
                neg_edges = []
                num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
                for i in range( int(num_neg * neg_prop) ):
                    hd_idx = np.random.randint(0, len(all_heads))
                    tl_idx = np.random.randint(0, len(all_tails))
                    while (all_heads[hd_idx], all_tails[tl_idx]) in edge_dict:
                        hd_idx = np.random.randint(0, len(all_heads))
                        tl_idx = np.random.randint(0, len(all_tails))
                    neg_edges.append([all_heads[hd_idx], all_tails[tl_idx]])
            elif negative_sample == "from_all":
                neg_edges = []
                num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
                for i in range( int(num_neg * neg_prop) ):
                    rnd = np.random.randint(0, len(self.node2id), 2)
                    tmp_edge = (rnd[0], rnd[1])
                    while tmp_edge in edge_dict or tmp_edge[0] == tmp_edge[1]:
                        rnd = np.random.randint(0, len(self.node2id), 2)
                        tmp_edge = (rnd[0], rnd[1])
                    neg_edges.append(list(tmp_edge))
            elif negative_sample == "prepared_neg":
                # some of the negative samples are pre-prepared
                neg_train = [[self.node2id[head], self.node2id[tail]] \
                    for head, tail, feat in G.edges.data() if "neg_trn" in feat["relation"]]
                neg_val = [[self.node2id[head], self.node2id[tail]] \
                    for head, tail, feat in G.edges.data() if "neg_dev" in feat["relation"]]
                neg_test = [[self.node2id[head], self.node2id[tail]] \
                    for head, tail, feat in G.edges.data() if "neg_tst" in feat["relation"]]
                print("num of prepared neg for train:{}, dev:{}, test:{}".format(len(neg_train), len(neg_val), len(neg_test)))
                neg_edges = []
                num_neg = len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos)
                for i in range(int(num_neg * neg_prop) - len(neg_train) - len(neg_val) - len(neg_test) ):
                    rnd = np.random.randint(0, len(self.node2id), 2)
                    tmp_edge = (rnd[0], rnd[1])
                    while tmp_edge in edge_dict or tmp_edge[0] == tmp_edge[1]:
                        rnd = np.random.randint(0, len(self.node2id), 2)
                        tmp_edge = (rnd[0], rnd[1])
                    neg_edges.append(list(tmp_edge))
                trn_val_idx = int(len(train_edges_pos)*neg_prop) - len(neg_train)
                val_tst_idx = trn_val_idx + int(len(val_edges_pos)*neg_prop)-len(neg_val)
                neg_edges = neg_train + neg_edges[:trn_val_idx]\
                        +neg_val + neg_edges[trn_val_idx:val_tst_idx]\
                        +neg_test + neg_edges[val_tst_idx:]
                assert len(neg_edges) >= int((len(train_edges_pos) + len(val_edges_pos) + len(test_edges_pos))*neg_prop)

            train_edges_neg = neg_edges[:int(len(train_edges_pos)*neg_prop)]
            val_edges_neg = neg_edges[int(len(train_edges_pos)*neg_prop):int((len(train_edges_pos)+len(val_edges_pos))*neg_prop)]
            test_edges_neg = neg_edges[int((len(train_edges_pos)+len(val_edges_pos))*neg_prop):]
            print('Number of negative examples after trucating:{}, validating:{}, testing:{}'.format(
                len(train_edges_neg), len(val_edges_neg), len(test_edges_neg)))    
        '''
        # NOTE: this function is deprecated with the new graph: left here for 
        # the record 
        '''
        def _make_edges(edges_neg, edges_pos):
            """
            Some unlisted arguments:

            edge_include_rel: would the edges include relation data or not.
                if True, edge = [head, tail, relation]  # used for kgbert_va model
                if False, edge = [head, tail]
            target_relation: whether you want specific relation graph or not.
                'all' or any of the 9 ATOMIC relations.
            """
            edges = []
            labels = [0] * len(edges_neg)
            for head, tail, rel in edges_neg:
                if target_relation != "all":
                    assert self.rel2id[target_relation] == rel  # final check
                if edge_include_rel:
                    edges.append([head, tail, rel])
                else:
                    edges.append([head, tail])
            for head, tail, rel in edges_pos:
                if target_relation != "all":
                    assert self.rel2id[target_relation] == rel  # final check
                if edge_include_rel:
                    edges.append([head, tail, rel])
                    labels.append(1)
                else:
                    edges.append([head, tail])
                    label = rel if target_relation == "all" else 1
                    labels.append(label)

            labels = np.array(labels)
            edges = np.array(edges)

            shuffle_idx = np.random.permutation(len(edges))
            return labels[shuffle_idx], edges[shuffle_idx]
        '''
        # self.train_labels, self.train_edges = _make_edges(train_edges_neg, train_edges_pos)
        # self.val_labels, self.val_edges = _make_edges(val_edges_neg, val_edges_pos)
        # self.test_labels, self.test_edges = _make_edges(test_edges_neg, test_edges_pos)
        for mode in ["trn", "dev", "tst"]:
            mode_edges = negative_edges[mode] + positive_edges[mode]
            mode_labels = negative_labels[mode] + positive_labels[mode]

            shuffle_idx = np.random.permutation(len(mode_edges))
            mode_edges = np.array(mode_edges)[shuffle_idx]
            mode_labels = np.array(mode_labels)[shuffle_idx]
            if mode == "trn":
                self.train_labels, self.train_edges = mode_labels, mode_edges
            elif mode == "dev":
                self.val_labels, self.val_edges = mode_labels, mode_edges
            else:
                self.test_labels, self.test_edges = mode_labels, mode_edges

        print('finish preparing neg samples')

        self.mode_edges = {
            "train":torch.tensor(self.train_edges).to(device),
            "valid":torch.tensor(self.val_edges).to(device),
            "test":torch.tensor(self.test_edges).to(device)
        }
        self.mode_labels = {
            "train":torch.tensor(self.train_labels).to(device),
            "valid":torch.tensor(self.val_labels).to(device),
            "test":torch.tensor(self.test_labels).to(device)
        }

        # Prepare a sparse adj matrix, mask all the valid and test set
        # adj list that contains all the training edges
        self.adj_list = [[] for i in range(len(self.id2node))]

        # Edges are all the edges except for those in test/val set
        val_edges_dict = {(edge[0], edge[1]) for edge in positive_edges["dev"]}
        test_edges_dict = {(edge[0], edge[1]) for edge in positive_edges["tst"]}

        for head, tail, feat in G.edges.data():
            if load_edge_types == "CS+ASER":
                raise NotImplementedError
            elif load_edge_types == "CS":
                raise NotImplementedError
            elif load_edge_types == "ASER":
                if isinstance(feat["relation"], dict):
                    edge = (self.node2id[head], self.node2id[tail])
                    if edge not in val_edges_dict and edge not in test_edges_dict:
                        if not neighbor_include_rel:
                            self.adj_list[self.node2id[head]].append(self.node2id[tail])
                        else:
                            relations = []
                            aser = list(set(ASER_RELATIONS) & set(feat["relation"]))
                            relations.extend([(self.rel2id[rel], feat["relation"][rel]) for rel in aser])
                            # [(rel_id, weight_in_aser)]
                            if len(relations) == 0:
                                continue
                            if highest_aser_rel:
                                # select the aser relation with the highest weight as the relation
                                # discard other relations
                                rel, weight = sorted(relations, key=lambda x:x[1], reverse=True)[0]
                                self.adj_list[self.node2id[head]].append((self.node2id[tail], rel))
                            else:
                                # include all relations
                                for rel, weight in relations:
                                    self.adj_list[self.node2id[head]].append((self.node2id[tail], rel))
                else:
                    pass


            # if load_edge_types == "CS+ASER":
            #     pass
            # else:
            #     is_aser = isinstance(feat["relation"], dict)
            #     if load_edge_types == "ASER" and not is_aser:
            #         continue
            #     if load_edge_types == "CS":
            #         if is_aser:
            #             continue
            #         if target_relation not in ["all"] + feat["relation"]:
            #             continue
                

            # edge = (self.node2id[head], self.node2id[tail])
            # if edge not in val_edges_dict and edge not in test_edges_dict:
            #     if not neighbor_include_rel:
            #         self.adj_list[self.node2id[head]].append(self.node2id[tail])
            #     else:
            #         relations = []
            #         if load_edge_types in ["CS+ASER", "CS"]:
            #             commonsense = list(set(CS_RELATIONS["all"]) & set(feat["relation"]))
            #             relations.extend([self.rel2id[rel] for rel in commonsense])
            #         if load_edge_types in ["CS+ASER", "ASER"]:
            #             aser = list(set(ASER_RELATIONS) & set(feat["relation"]))
            #             relations.extend([self.rel2id[rel] for rel in aser])
            #         for rel in relations:
            #             self.adj_list[self.node2id[head]].append((self.node2id[tail], rel))

        # 4. Tokenize nodes

        if encoder == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif encoder == "roberta":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.id2nodestoken = None
        # NOTE: id2reltoken is used for KG Bert type a. However, it might also be
        # useful in later application, so keeping it alive.
        if use_nl_rel:
            # use the natural language explanations to encode the relation
            # CS_RELATIONS_2NL
            self.id2reltoken = { self.rel2id[rel]:torch.tensor(self.tokenizer.encode(
                CS_RELATIONS_2NL.get(rel, rel), add_special_tokens=False)).to(device) for rel in tqdm(self.rel2id)}
        else:
            self.id2reltoken = {self.rel2id[rel]: torch.tensor(self.tokenizer.encode(
                rel, add_special_tokens=False)).to(device) for rel in tqdm(self.rel2id)}
        

    def tokenize_nodes(self, node_token_path):
        if os.path.exists(node_token_path):
            with open(node_token_path, "rb") as reader:
                self.id2nodestoken = pickle.load(reader)
            print("after loading id2nodestoken cache from", node_token_path)
        else:
            print("no cache found, tokenizing nodes. This may take up to an hour..")
            self.id2nodestoken = {self.node2id[line]: torch.tensor(self.tokenizer.encode(line, 
                add_special_tokens=True)) for line in tqdm(self.node2id)}
            with open(node_token_path, "wb") as writer:
                pickle.dump(self.id2nodestoken, writer, pickle.HIGHEST_PROTOCOL) 
            

    def get_adj_list(self):
        return self.adj_list
    def get_nid2text(self):
        return self.id2node

    def get_nodes_tokenized(self):
        if self.id2nodestoken is None:
            print("nodes are not tokenized yet. Please tokenize first by calling"
                  "\nMultiGraphDataset.tokenize_nodes(node_token_path, device)")
        return self.id2nodestoken
    
    def get_relations_tokenized(self):
        return self.id2reltoken

    def get_batch(self, batch_size=16, mode="train"):
        assert mode in ["train", "valid", "test"], f"invalid mode: {mode}"

        for i in range(0, len(self.mode_edges[mode]), batch_size):
            yield self.mode_edges[mode][i:min(i+batch_size, len(self.mode_edges[mode]))], \
                self.mode_labels[mode][i:min(i+batch_size, len(self.mode_edges[mode]))]
