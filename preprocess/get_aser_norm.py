from random import sample

import networkx as nx
import numpy as np
from tqdm import tqdm, trange
import networkx as nx
import sys
sys.path.append("..")
from utils.utils import subject_list, object_list, group_list
from utils.aser_to_glucose import generate_aser_to_glucose_dict

G_aser = nx.read_gpickle("../data/G_aser_core_nofilter.pickle")

def reverse_px_py(original: str):
    return original.replace("PersonX", "[PX]").replace("PersonY", "[PY]").replace("[PX]", "PersonY").replace(
        "[PY]", "PersonX")
  
def merge_rel_dict(d1: dict, d2: dict):
    d_merge = {}
    for key in set(d1.keys()) |  set(d2.keys()):
        d_merge[key] = d1.get(key, 0) + d2.get(key, 0)
    return d_merge

def get_normalized_graph(G: nx.DiGraph):
    G_conceptualized = nx.DiGraph()
    for head, tail, feat_dict in tqdm(G.edges.data()):
        head_split = head.split()
        tail_split = tail.split()
        head_subj = head_split[0]
        tail_subj = tail_split[0]
        relations = feat_dict["relations"]
        
        _, re_head, re_tail, _ = generate_aser_to_glucose_dict(head, tail, True)
        re_head_reverse, re_tail_reverse = reverse_px_py(re_head), reverse_px_py(re_tail)
        if len(re_head) > 0 and len(re_tail) > 0:
            if G_conceptualized.has_edge(re_head, re_tail):
                G_conceptualized.add_edge(re_head, re_tail, 
                    relation=merge_rel_dict(G_conceptualized[re_head][re_tail]["relation"],
                                    relations)
                )
            else:
                G_conceptualized.add_edge(re_head, re_tail, relation=relations)
        if len(re_head_reverse) > 0 and len(re_tail_reverse) > 0:
            if G_conceptualized.has_edge(re_head_reverse, re_tail_reverse):
                G_conceptualized.add_edge(re_head_reverse, re_tail_reverse, 
                relation=merge_rel_dict(G_conceptualized[re_head_reverse][re_tail_reverse]["relation"],
                                    relations))
            else:
                G_conceptualized.add_edge(re_head_reverse, re_tail_reverse, relation=relations)
    return G_conceptualized

def filter_event(event):
    tokens = event.split()
    if len(tokens) <= 2 and any(kw in tokens for kw in ["say", "do", "know", "tell", "think", ]):
        return True
    if tokens[0] in ["who", "what", "when", "where", "how", "why", "which", "whom", "whose"]:
        return True
    return False  
to_be_filtered = [node for node in tqdm(G_aser.nodes()) if filter_event(node)]
print("proportion of filtered nodes", len(to_be_filtered)/len(G_aser.nodes()))
G_aser.remove_nodes_from(to_be_filtered)

G_aser_conceptualized = get_normalized_graph(G_aser)
print("Before Conceptualization:\nNumber of Edges: {}\tNumber of Nodes: {}\n".format(len(G_aser.edges), len(G_aser.nodes)))
print("After Conceptualization:\nNumber of Edges: {}\tNumber of Nodes: {}\n".format(len(G_aser_conceptualized.edges),
                                                                                    len(G_aser_conceptualized.nodes)))

nx.write_gpickle(G_aser_conceptualized, '../data/G_aser_nodefilteronly_norm.pickle')

