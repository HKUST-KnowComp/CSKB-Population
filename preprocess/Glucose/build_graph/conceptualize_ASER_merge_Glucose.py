import sys
from random import sample

import networkx as nx
import numpy as np
from tqdm import tqdm, trange

sys.path.append('../../')
from Glucose.utils.aser_to_glucose import generate_aser_to_glucose_dict
from Glucose.utils.glucose_utils import glucose_subject_list

# Load the filtered ASER graph
aser = nx.read_gpickle('../../dataset/G_aser_core.pickle')
node2id_dict = np.load("../../dataset/ASER_core_node2id.npy", allow_pickle=True).item()
id2node_dict = dict([(node2id_dict[node], node) for node in node2id_dict])
glucose_matching = np.load('../process_dataset/Final_Version/glucose_final_matching.npy', allow_pickle=True).item()

# We test the coverage in the norm ASER
print_str = "\n\nStatistics in ASER_Norm:\n\n"
total_count, total_head, total_tail, total_both = 0, 0, 0, 0
for i in trange(1, 11):
    list_count, list_head, list_tail, list_both = len(glucose_matching[i]), 0, 0, 0
    for ind in range(len(glucose_matching[i])):
        for h in glucose_matching[i][ind]['total_head']:
            if h in node2id_dict.keys():
                list_head += 1
                break
        for t in glucose_matching[i][ind]['total_tail']:
            if t in node2id_dict.keys():
                list_tail += 1
                break
        for h, t in glucose_matching[i][ind]['both']:
            if h in node2id_dict.keys() and t in node2id_dict.keys():
                list_both += 1
                break
    print_str += (
        "In list {}, Total Head: {}\tMatched Head: {} ({}%)\tMatched Tail: {} ({}%)\tMatched Both: {} ({}%)\n"
            .format(i, list_count, list_head, round(list_head / list_count, 3) * 100,
                    list_tail, round(list_tail / list_count, 3) * 100, list_both,
                    round(list_both / list_count, 3) * 100))
    total_count += list_count
    total_head += list_head
    total_tail += list_tail
    total_both += list_both
print_str += (
    "\n\nIn total: Total Head: {}\tMatched Head: {} ({}%)\tMatched Tail: {} ({}%)\tMatched Both: {} ({}%)".format(
        total_count, total_head, 100 * round(total_head / total_count, 3),
        total_tail, 100 * round(total_tail / total_count, 3), total_both, 100 * round(total_both / total_count, 3)))
print(print_str)

# Do some ASER edge type statistics
all_edge_types = {}
for head, tail, feat_dict in aser.edges.data():
    for r in feat_dict["edge_type"]:
        if r in all_edge_types.keys():
            all_edge_types[r] += 1
        else:
            all_edge_types[r] = 1
print("Edge types in ASER:")
print(all_edge_types)


def reverse_px_py(original: str):
    """
    This function replace PersonX PersonY in a string.
    :param original: The original string to be processed.
    :return: The string with PersonX and PersonY reversed.
    """
    return original.replace("PersonX", "[PX]").replace("PersonY", "[PY]").replace("[PX]", "PersonY").replace(
        "[PY]", "PersonX")


def get_conceptualized_graph(G: nx.DiGraph):
    """
    This function get the conceptualized version of a graph
    :param G: A directed graph from networkx
    :return: The conceptualized version of the input graph. Conceptualized means changing the subjects like 'i', 'you'
    to 'PersonX' 'PersonY'
    """
    G_conceptualized = nx.DiGraph()
    for head, tail, feat_dict in tqdm(G.edges.data()):
        head = id2node_dict[head]
        tail = id2node_dict[tail]
        head_split = head.split()
        tail_split = tail.split()
        head_subj = head_split[0]
        tail_subj = tail_split[0]
        relations = feat_dict["edge_type"]
        for r in relations:
            if head_subj == tail_subj and head_subj in glucose_subject_list:
                new_rel = r + "_agent"
            elif head_subj != tail_subj and head_subj in glucose_subject_list and tail_subj in glucose_subject_list:
                new_rel = r + "_theme"
            else:
                new_rel = r + "_general"
            _, re_head, re_tail, _ = generate_aser_to_glucose_dict(head, tail, True)
            re_head_reverse, re_tail_reverse = reverse_px_py(re_head), reverse_px_py(re_tail)
            if len(re_head) > 0 and len(re_tail) > 0:
                if G_conceptualized.has_edge(re_head, re_tail):
                    G_conceptualized.add_edge(re_head, re_tail, relation=list(
                        set(G_conceptualized[re_head][re_tail]["relation"] + [new_rel])))
                else:
                    G_conceptualized.add_edge(re_head, re_tail, relation=[new_rel])
            if len(re_head_reverse) > 0 and len(re_tail_reverse) > 0:
                if G_conceptualized.has_edge(re_head_reverse, re_tail_reverse):
                    G_conceptualized.add_edge(re_head_reverse, re_tail_reverse, relation=list(
                        set(G_conceptualized[re_head_reverse][re_tail_reverse]["relation"] + [new_rel])))
                else:
                    G_conceptualized.add_edge(re_head_reverse, re_tail_reverse, relation=[new_rel])
    return G_conceptualized


aser_conceptualized = get_conceptualized_graph(aser)
print("Before Conceptualization:\nNumber of Edges: {}\tNumber of Nodes: {}\n".format(len(aser.edges), len(aser.nodes)))
print("After Conceptualization:\nNumber of Edges: {}\tNumber of Nodes: {}\n".format(len(aser_conceptualized.edges),
                                                                                    len(aser_conceptualized.nodes)))

nx.write_gpickle(aser_conceptualized, '../../dataset/G_aser_concept.pickle')

# Let's sample some ASER conceptualization to check whether it's correct
for i in sample(list(aser_conceptualized.edges.data()), 30) + ['\n'] + sample(list(aser_conceptualized.nodes.data()),
                                                                              10):
    print(i)


# Now let's calculate the shortest path
def get_shortest_path(G, head, tail):
    try:
        p = nx.shortest_path_length(G, source=head, target=tail)
    except nx.NodeNotFound:
        return -1
    except nx.NetworkXNoPath:
        return -1
    return p


full_path, norm_path = [], []
for i in range(1, 11):
    for ind in trange(len(glucose_matching[i])):
        norm_temp, full_temp = [], []
        for h, t in glucose_matching[i][ind]['both']:
            _, re_h, re_t, _ = generate_aser_to_glucose_dict(h, t, True)
            if re_h in aser_conceptualized and re_t in aser_conceptualized:
                norm_temp.append(get_shortest_path(aser_conceptualized, re_h, re_t))
        if norm_temp:
            try:
                norm_path.append(min([i for i in norm_temp if i > 0]))
            except ValueError:
                norm_path.append(0)
        else:
            norm_path.append(0)
        for h, t in glucose_matching[i][ind]['both']:
            try:
                hid = node2id_dict[h]
                tid = node2id_dict[t]
            except KeyError:
                continue
            if hid in aser and tid in aser:
                full_temp.append(get_shortest_path(aser, hid, tid))
        if full_temp:
            try:
                full_path.append(min([i for i in full_temp if i > 0]))
            except ValueError:
                full_path.append(0)
        else:
            full_path.append(0)
print("Average Shortest Path in Full ASER is: {}".format(np.mean([i for i in full_path if i > 0])))
print("Average Shortest Path in Norm ASER is: {}".format(np.mean([i for i in norm_path if i > 0])))

# Calculate the average path in a simple graph:
G_simple = nx.Graph()
G_simple.add_nodes_from(aser_conceptualized)
G_simple.add_edges_from(aser_conceptualized.edges.data())
G_simple_full = nx.Graph()
G_simple_full.add_nodes_from(aser)
G_simple_full.add_edges_from(aser.edges.data())

full_path, norm_path = [], []
for i in range(1, 11):
    for ind in trange(len(glucose_matching[i])):
        norm_temp, full_temp = [], []
        for h, t in glucose_matching[i][ind]['both']:
            _, re_h, re_t, _ = generate_aser_to_glucose_dict(h, t, True)
            if re_h in G_simple and re_t in G_simple:
                norm_temp.append(get_shortest_path(G_simple, re_h, re_t))
        if norm_temp:
            try:
                norm_path.append(min([i for i in norm_temp if i > 0]))
            except ValueError:
                norm_path.append(0)
        else:
            norm_path.append(0)
        for h, t in glucose_matching[i][ind]['both']:
            try:
                hid = node2id_dict[h]
                tid = node2id_dict[t]
            except KeyError:
                continue
            if hid in G_simple_full and tid in G_simple_full:
                full_temp.append(get_shortest_path(G_simple_full, hid, tid))
        if full_temp:
            try:
                full_path.append(min([i for i in full_temp if i > 0]))
            except ValueError:
                full_path.append(0)
        else:
            full_path.append(0)
print("In No Direction Scenario:")
print("Average Shortest Path in Full ASER is: {}".format(np.mean([i for i in full_path if i > 0])))
print("Average Shortest Path in Norm ASER is: {}".format(np.mean([i for i in norm_path if i > 0])))

# Now let's start merging with Glucose
G_Glucose = nx.read_gpickle('../../dataset/G_Glucose.pickle')
print("Node Coverage for Glucose Graph is: {}%\nEdge Coverage for Glucose Graph is: {}%".format(
    100 * round(sum([node in aser_conceptualized for node in G_Glucose.nodes()]) / len(G_Glucose.nodes()), 4),
    100 * round(sum([edge in aser_conceptualized.edges for edge in G_Glucose.edges()]) / len(G_Glucose.edges()), 4)))

print("Before Merging:\nEdges in ASER: {}\t\t\t\tNodes in ASER: {}\n".format(len(aser_conceptualized.edges()),
                                                                             len(aser_conceptualized.nodes())))
aser_conceptualized.add_nodes_from(list(G_Glucose.nodes.data()))
aser_conceptualized.add_edges_from(list(G_Glucose.edges.data()))
print("\nAfter Merging:\nEdges in ASER+Glucose: {}\t\t\tNodes in ASER+Glucose: {}".format(
    len(aser_conceptualized.edges()),
    len(aser_conceptualized.nodes())))
print("New Edges: {}\tNew Nodes: {}".format(len(aser_conceptualized.edges()) - 41336290,
                                            len(aser_conceptualized.nodes()) - 11872745))
nx.write_gpickle(aser_conceptualized, '../../dataset/G_aser_glucose.pickle')
