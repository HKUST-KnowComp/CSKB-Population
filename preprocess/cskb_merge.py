import pandas as pd
from random import sample
import networkx as nx
import numpy as np
from tqdm import tqdm, trange
import networkx as nx
import sys
sys.path.append("../")
sys.path.append("../Glucose")
from utils.utils import subject_list, object_list, group_list
from Glucose.utils.aser_to_glucose import generate_aser_to_glucose_dict

def reverse_px_py(original: str):
    return original.replace("PersonX", "[PX]").replace("PersonY", "[PY]").replace("[PX]", "PersonY").replace(
        "[PY]", "PersonX")
  
def merge_rel_dict(d1: dict, d2: dict):
  d_merge = {}
  for key in set(d1.keys()) |  set(d2.keys()):
    d_merge[key] = d1.get(key, 0) + d2.get(key, 0)
  return d_merge

def normalize_head_tail(head, tail):
    head_split = head.split()
    tail_split = tail.split()
    head_subj = head_split[0]
    tail_subj = tail_split[0]

    _, re_head, re_tail, _ = generate_aser_to_glucose_dict(head, tail, True)
    re_head_reverse, re_tail_reverse = reverse_px_py(re_head), reverse_px_py(re_tail)
    return re_head, re_tail, re_head_reverse, re_tail_reverse


relations = ['oEffect', 'oReact', 'oWant', 'xAttr', 
             'xIntent', 'xNeed', 'xReact', 'xWant', 'xEffect']
ATOMIC_tuples = dict([(r, 
                    np.load('/home/data/tfangaa/CKGP/data/new_matching/ASER-format-words/ATOMIC_{}.npy'.format(r), 
                    allow_pickle=True)) for r in relations])
clause_idx = np.load('../../ASER-core/Matching-atomic/clause_idx.npy', allow_pickle=True)
wc_idx = np.load('../../ASER-core/Matching-atomic/wildcard_idx.npy', allow_pickle=True)
node2id_dict = np.load("/home/data/tfangaa/CKGP/data/ASER_raw_data/aser_raw_node_dict.npy", allow_pickle=True)[()]

atomic_raw = pd.read_csv("/home/tfangaa/Downloads/ATOMIC/v4_atomic_all_agg.csv")
split_dict = dict((i,spl) for i,spl in enumerate(atomic_raw['split']))

############################################################
# 1. ATOMIC
############################################################

def get_atomic_graph(ATOMIC_tuples, relations):
  G_atomic = nx.DiGraph()
  for r in relations:
    for hid, tuple_list in tqdm(enumerate(ATOMIC_tuples[r])):
      if hid in clause_idx or hid in wc_idx:
        continue
      for tid, tuples in enumerate(tuple_list):
        head_norm_list = []
        tail_norm_list = []
        for head, tail in tuples:
          if len(head) == 0 or len(tail) == 0:
            continue
          re_head, re_tail, _, _ = normalize_head_tail(head, tail)
          head_norm_list.append(re_head)
          tail_norm_list.append(re_tail)
        head_norm_list, tail_norm_list = list(set(head_norm_list)), list(set(tail_norm_list))
        head_agg, tail_agg = "\t".join(head_norm_list), "\t".join(tail_norm_list)

        if (head_agg, tail_agg) in G_atomic.edges:
          G_atomic[head_agg][tail_agg]["relation"] = \
            list(set(G_atomic[head_agg][tail_agg]["relation"])|set([r]))
          G_atomic[head_agg][tail_agg]["hid"] = \
            list(set(G_atomic[head_agg][tail_agg]["hid"])|set([hid]))
          G_atomic[head_agg][tail_agg]["tid"] = \
            list(set(G_atomic[head_agg][tail_agg]["tid"])|set([tid]))
        else:
          G_atomic.add_edge(head_agg, tail_agg, relation=[r], hid=[hid], tid=[tid], split=split_dict[hid])
  return G_atomic
G_atomic = get_atomic_graph(ATOMIC_tuples, relations)
if not os.path.exists("../data/final_graph_file"):
    os.mkdir("../data/final_graph_file")
    os.mkdir("../data/final_graph_file/CSKG")
nx.write_gpickle(G_atomic, 
  "../data/final_graph_file/CSKG/G_atomic_agg_node.pickle")

############################################################
# 2. ConceptNet
############################################################
from utils.atomic_utils import SUBJ2POSS, PP_SINGLE

omcs_tuples = np.load("../data/omcs_tuples.npy", allow_pickle=True)[()]
parsed_omcs_dict = {
  "trn":np.load("../data/new_matching/ASER-format-words/omcs_trn.npy", allow_pickle=True),
  "tst":np.load("../data/new_matching/ASER-format-words/omcs_tst.npy", allow_pickle=True),
  "dev":np.load("../data/new_matching/ASER-format-words/omcs_dev.npy", allow_pickle=True),  
}

def get_cn_graph(parsed_omcs_dict, omcs_tuples):
  G_cn = nx.DiGraph()
  # check coverage

  for spl in parsed_omcs_dict:
    for hid, (head, rel, tail) in tqdm(enumerate(omcs_tuples[spl])):
      head_norm_list = []
      tail_norm_list = []
      for pp in PP_SINGLE:
        if pp + " " + head in node2id_dict and pp + " " + tail.lower() in node2id_dict:
          re_head, re_tail, _, _ = normalize_head_tail(head, tail)
          head_norm_list.append(re_head)
          tail_norm_list.append(re_tail)
      collapsed_list, r = parsed_omcs_dict[spl][hid]
      assert r == rel
      for head, tail in collapsed_list:
        if not (len(head) > 0 and len(tail) > 0):
          continue
        re_head, re_tail, _, _ = normalize_head_tail(head, tail)
        head_norm_list.append(re_head)
        tail_norm_list.append(re_tail)
      head_norm_list, tail_norm_list = list(set(head_norm_list)), list(set(tail_norm_list))
      head_agg, tail_agg = "\t".join(head_norm_list), "\t".join(tail_norm_list)

      if (head_agg, tail_agg) in G_cn.edges:
        G_cn[head_agg][tail_agg]["relation"] = \
          list(set(G_cn[head_agg][tail_agg]["relation"])|set([rel]))
        G_cn[head_agg][tail_agg]["hid"] = \
          list(set(G_cn[head_agg][tail_agg]["hid"])|set([hid]))
      else:
        G_cn.add_edge(head_agg, tail_agg, relation=[rel], hid=[hid], split=spl)
  return G_cn
G_cn = get_cn_graph(parsed_omcs_dict, omcs_tuples)

nx.write_gpickle(G_cn, 
                 "../data/final_graph_file/CSKG/G_cn_agg_node.pickle")

############################################################
# 3. ATOMIC2020
############################################################
parsed_atomic2020_dict = {
  "trn":np.load("../data/new_matching/ASER-format-words/atomic2020_trn.npy", allow_pickle=True),
  "tst":np.load("../data/new_matching/ASER-format-words/atomic2020_tst.npy", allow_pickle=True),
  "dev":np.load("../data/new_matching/ASER-format-words/atomic2020_dev.npy", allow_pickle=True),}

from tqdm import tqdm

def get_atomic2020_graph(parsed_atomic2020_dict):
  G_atomic2020 = nx.DiGraph()

  for spl in parsed_atomic2020_dict:
    for hid, item in tqdm(enumerate(parsed_atomic2020_dict[spl])):
      if len(item) == 0:
        continue
      collapsed_list, rel = item
      head_norm_list, tail_norm_list = [], []
      for head, tail in collapsed_list:
        if not (len(head) > 0 and len(tail) > 0):
          continue
        re_head, re_tail, _, _ = normalize_head_tail(head, tail)
        head_norm_list.append(re_head)
        tail_norm_list.append(re_tail)
      head_norm_list, tail_norm_list = list(set(head_norm_list)), list(set(tail_norm_list))
      head_agg, tail_agg = "\t".join(head_norm_list), "\t".join(tail_norm_list)

      if (head_agg, tail_agg) in G_atomic2020.edges:
        G_atomic2020[head_agg][tail_agg]["relation"] = \
          list(set(G_atomic2020[head_agg][tail_agg]["relation"])|set([rel]))
        G_atomic2020[head_agg][tail_agg]["hid"] = \
          list(set(G_atomic2020[head_agg][tail_agg]["hid"])|set([hid]))
      else:
        G_atomic2020.add_edge(head_agg, tail_agg, relation=[rel], hid=[hid], split=spl)
  return G_atomic2020
G_atomic2020 = get_atomic2020_graph(parsed_atomic2020_dict)

nx.write_gpickle(G_atomic2020, "../data/final_graph_file/CSKG/G_atomic2020_agg_node.pickle")
