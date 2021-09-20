import sys
import numpy as np
import networkx as nx
from random import sample
from tqdm import tqdm, trange
from itertools import chain
from collections import Counter

G_atomic =  nx.read_gpickle("../data/final_graph_file/CSKG/G_atomic_agg_node.pickle")
G_Glucose = nx.read_gpickle('../data/final_graph_file/CSKG/G_glucose_agg_node.pickle')  # new glucose
G_cn = nx.read_gpickle("../data/final_graph_file/CSKG/G_cn_agg_node.pickle") # new conceptnet
G_atomic2020 = nx.read_gpickle("../data/final_graph_file/CSKG/G_atomic2020_agg_node.pickle")

cskg_dict = {
  "atomic":G_atomic,
  "glucose":G_Glucose,
  "atomic2020":G_atomic2020,
  "cn":G_cn,
}
G_aser_norm = nx.read_gpickle("../data/final_graph_file/ASER/G_aser_norm_nodefilter.pickle")
aser_norm_dict = dict([(node, i) for i, node in enumerate(G_aser_norm.nodes())])

# commonsense relations
cs_rels = ['oEffect', 'xEffect', 'general Effect',
           'oWant','xWant','general Want',
           'oReact', 'xReact',   'general React',
           'xAttr',  'xNeed', 'xIntent', 
           'isBefore',  'isAfter', 'HinderedBy', 
           'Causes',   'xReason', 'HasSubEvent',]
###############################################################
# 1. Select Heads&Tails from Aggregated graph
###############################################################

G_cs = nx.DiGraph()
for cskg_name, G_cskg in cskg_dict.items():
    for head_agg, tail_agg, feat in tqdm(G_cskg.edges.data()):
        pairs = [(head, tail) for head in head_agg.split("\t") for tail in tail_agg.split("\t")]
        # 1. if there are edge matches in ASER, select the longest one
        pairs_matched = [(head, tail) for head, tail in pairs if head in aser_norm_dict and tail in aser_norm_dict ]
        if len(pairs_matched) > 0:
            selected_pair = pairs_matched[np.argmax(
                [len(head.split())+len(tail.split()) for head, tail in pairs_matched]
            )]
        else:
            # 2. If no pair match, select the head match
            pairs_head_matched = [(head, tail) \
                   for head, tail in pairs if head in aser_norm_dict]
            if len (pairs_head_matched) > 0:
                selected_pair = pairs_head_matched[np.argmax(
                    [len(head.split())+len(tail.split()) for head, tail in pairs_head_matched]
                )]
            else:
                # 3. if no head match, select tail match.
                pairs_tail_matched = [(head, tail) \
                     for head, tail in pairs if tail in aser_norm_dict]
                if len(pairs_tail_matched) > 0:
                    selected_pair = pairs_tail_matched[np.argmax(
                        [len(head.split())+len(tail.split()) for head, tail in pairs_tail_matched]
                    )]
                else:
                    # if no tail match, simply select the longest
                    selected_pair = pairs[np.argmax(
                        [len(head.split())+len(tail.split()) for head, tail in pairs]
                    )]
        G_cs.add_edge(selected_pair[0], selected_pair[1], dataset=cskg_name,
                     relation=feat["relation"],split=feat["split"],hid=feat["hid"])
        
###############################################################
# 2. Merge and sample negative
###############################################################

cs_rels_dict = dict([(r, True) for r in cs_rels])
# all_aser_nodes = list(G_aser.nodes())

all_candi_cs_edge = [(feat["split"], (head, tail, feat["relation"])) for head, tail, feat in tqdm(G_cs.edges.data()) \
   if any(rel in cs_rels_dict for rel in feat["relation"])]

all_candi_cs_edge_dict = {"trn":[], "dev":[], "tst":[]}
for spl, (head, tail, rels) in all_candi_cs_edge:
    all_candi_cs_edge_dict[spl].append((head, tail, rels))

neg_prop = 1
o_prop = 0.1
i_prop = 0.1
s_prop = 0.1

neg_edges = dict([(dataset_name, dict([(r, 
                   {"trn":[], "dev":[], "tst":[]}) 
                  for r in cs_rels])) \
                  for dataset_name in ["atomic", "atomic2020", "glucose", "cn"]])

for spl in ["trn", "tst", "dev"]:
    for r in tqdm(cs_rels):
        # 1. all (h, r, t) tuples with relations other than 3.
        all_candi_other = []
        all_candi_inv = []
        for head, tail, feat in G_cs.edges.data():
            if feat["split"]==spl:
                if all(rel != r for rel in feat["relation"]):
                    all_candi_other.append((head, tail))
                else:
                    all_candi_inv.append((head, tail))
        all_candi_other, all_candi_inv = np.array(all_candi_other), np.array(all_candi_inv)  

        existing_edges_dict = dict([(tuple(item), True) for item in all_candi_inv])
        # edges of this relation
        all_candi_nodes = list(set(chain(*all_candi_inv)))
        num_this_rel = len(all_candi_inv)
        
        for dataset in ["atomic", "atomic2020", "glucose", "cn"]:
        
            # sample negative examples under the 3 hierarchies.
            num_this_dataset = len([1 for _, _, feat in G_cs.edges.data()\
                                   if feat["dataset"]==dataset and feat["split"]==spl\
                                   and any(rel == r for rel in feat["relation"])])

            # O edges
            o_edges = all_candi_other[
              np.random.choice(list(range(len(all_candi_other))), 
                                int(num_this_dataset * neg_prop * o_prop), replace=False)]

            # I edges
            i_edges = all_candi_inv[np.random.choice(list(range(len(all_candi_inv))), 
                      int(num_this_dataset * neg_prop * i_prop), replace=False)]
            i_edges = [(tail, head) for head, tail in i_edges]
            # S edges

            s_edges = np.random.choice(all_candi_nodes, [int(num_this_dataset * neg_prop * s_prop), 2])
            for i in range(len(s_edges)):
                while tuple(s_edges[i]) in existing_edges_dict:
                    h_n, t_n = np.random.choice(all_candi_nodes, 2)
                    s_edges[i] = [h_n, t_n]

            rand_edges = []
            num_rand = int((1 - o_prop - i_prop - s_prop) * neg_prop * num_this_dataset)

            rand_edges = np.random.choice(all_candi_nodes, [num_rand, 2])
            for i in range(len(rand_edges)):
                while tuple(rand_edges[i]) in existing_edges_dict:
                    h_n, t_n = np.random.choice(all_candi_nodes, 2)
                    rand_edges[i] = [h_n, t_n]
            neg_edges[dataset][r][spl] = [o_edges, i_edges, s_edges, rand_edges]

G_aser_all = G_aser_norm


for head, tail, feat in tqdm(G_cs.edges.data()):
    G_aser_all.add_edge(head, tail, relation=feat["relation"], split=feat["split"], dataset=feat["dataset"])
for dataset in neg_edges:
    for r in neg_edges[dataset]:
        for spl in neg_edges[dataset][r]:
            negs = list(chain(*neg_edges[dataset][r][spl]))
            for i, (head, tail) in enumerate(negs):
                if (head, tail) in G_aser_all.edges():
                    if isinstance(G_aser_all[head][tail]["relation"], dict):
                        # Aser edge
                        G_aser_all[head][tail]["relation"] = {
                          **G_aser_all[head][tail]["relation"],
                          **{"neg_" + r:1},
                        }
                        G_aser_all[head][tail]["dataset"] = dataset
                        G_aser_all[head][tail]["split"] = spl
                    elif isinstance(G_aser_all[head][tail]["relation"], list):
                        G_aser_all[head][tail]["relation"] = \
                          list(set(G_aser_all[head][tail]["relation"] + ["neg_" + r]))
                        G_aser_all[head][tail]["dataset"] = dataset
                        G_aser_all[head][tail]["split"] = spl
                else:
                    G_aser_all.add_edge(head, tail, relation=["neg_" + r], split=spl,
                                          dataset=dataset)
                    
if not os.path.exist("../data/final_graph_file/merge/"):
    os.mkdir("../data/final_graph_file/merge/")
nx.write_gpickle(G_aser_all, 
"../data/final_graph_file/merge/"
"G_nodefilter_aser_all_inv_{}_shuffle_{}_other{}_negprop_{}.pickle".format(
  int(i_prop*100),
  int(s_prop*100),
  int(o_prop*100),
  str(neg_prop)))