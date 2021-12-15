import os
import sys
import time
sys.path.append('../')
import argparse
import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool
from itertools import chain
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from aser.database.kg_connection import ASERKGConnection

st = time.time()

kg_conn = ASERKGConnection('KG.db', 
                           mode='memory', grain="words", load_types=["merged_eventuality", "words", "eventuality"])

print('time:', time.time()-st)


G_aser = nx.DiGraph()
for node in tqdm(kg_conn.merged_eventuality_cache):
    G_aser.add_node(node, 
                    patterns=kg_conn.get_event_patterns(node),
                    freq=kg_conn.get_event_frequency(node))



# traverse all the nodes, and get it's all neighbors
gather_relations = lambda key, successor_dict: \
  list(set(chain(*[list(item[1].keys()) for item in successor_dict[key]])))

gather_weights = lambda key, successor_dict, rels:\
  dict([(r, sum([item[1].get(r, 0) for item in successor_dict[key]])) for r in rels])
  

for node in tqdm(kg_conn.merged_eventuality_cache):
    successor_dict = kg_conn.merged_eventuality_relation_cache["head_words"].get(node, {})
    selected_tails = [(key, gather_weights(key, successor_dict, gather_relations(key, successor_dict)))\
                      for key, relations in successor_dict.items()]
    for key, rels in selected_tails:
      G_aser.add_edge(node, key, 
                      relations=rels, )

nx.write_gpickle(G_aser, "data/G_aser_core_nofilter_di.pickle")