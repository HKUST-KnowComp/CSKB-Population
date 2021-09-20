import sys

import networkx as nx
import numpy as np
from tqdm import trange

sys.path.append('../../')
from Glucose.utils.aser_to_glucose import generate_aser_to_glucose_dict

# Load the Glucose matching result and unusable dicts
glucose_matching = np.load('../process_dataset/Final_Version/glucose_final_matching.npy', allow_pickle=True).item()
fail_index = np.load('unusable_index.npy', allow_pickle=True).item()

total_head, total_tail = [], []
both_head, both_tail = [], []
for i in trange(1, 11):
    for ind in glucose_matching[i].keys():
        if ind in fail_index['head'][i] or ind in fail_index['tail'][i]:
            continue
        else:
            total_head.extend(glucose_matching[i][ind]['total_head'])
            total_tail.extend(glucose_matching[i][ind]['total_tail'])
            both_head.extend([h[0] for h in glucose_matching[i][ind]['both']])
            both_tail.extend([h[1] for h in glucose_matching[i][ind]['both']])
total_tail = list(np.unique(total_tail))
total_head = list(np.unique(total_head))
both_tail = list(np.unique(both_tail))
both_head = list(np.unique(both_head))
print(
    "There are total {} unique heads and {} unique tails.\nAmong which, {} heads and {} tails contribute are connected by edges.".format(
        len(total_head), len(total_tail), len(both_head), len(both_tail)))

# Add the edges!
G_Glucose = nx.DiGraph()
for i in trange(1, 11):
    for ind in glucose_matching[i].keys():
        if ind in fail_index['head'][i] or ind in fail_index['tail'][i]:
            continue
        else:
            for h, t in glucose_matching[i][ind]['both']:
                _, re_h, re_t, _ = generate_aser_to_glucose_dict(h, t, True)
                if (re_h, re_t) not in G_Glucose:
                    G_Glucose.add_edge(re_h, re_t, dataset='GLUCOSE', relation='Cause', list_id=i,
                                       hid=glucose_matching[i][ind]['head_id'], tid=glucose_matching[i][ind]['tail_id'])
            for head in glucose_matching[i][ind]['total_head']:
                for tail in glucose_matching[i][ind]['total_tail']:
                    _, re_head, re_tail, _ = generate_aser_to_glucose_dict(head, tail)
                    G_Glucose.add_node(re_head, dataset='GLUCOSE', list_id=i, hid=glucose_matching[i][ind]['head_id'])
                    G_Glucose.add_node(re_tail, dataset='GLUCOSE', list_id=i, tid=glucose_matching[i][ind]['tail_id'])
                    if head.split(' ')[0] == tail.split(' ')[0]:
                        if (re_head, re_tail) not in G_Glucose:
                            G_Glucose.add_edge(re_head, re_tail, dataset='GLUCOSE', relation='Cause', list_id=i,
                                               hid=glucose_matching[i][ind]['head_id'],
                                               tid=glucose_matching[i][ind]['tail_id'])

# Sample some edges and nodes to check
edge = list(G_Glucose.edges.data())
node = list(G_Glucose.nodes.data())
for i in range(10, 30):
    print(edge[i])
    print(node[i])

nx.write_gpickle(G_Glucose, './G_Glucose.pickle')
nx.write_gpickle(G_Glucose, '../../dataset/G_Glucose.pickle')

print("Total Edges in Glucose: {}".format(sum([1 for _, _, feat_dict in G_Glucose.edges.data()])))
print("Total Nodes in Glucose: {}".format(sum([1 for _, feat_dict in G_Glucose.nodes.data()])))
