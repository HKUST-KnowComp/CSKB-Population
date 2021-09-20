import pandas as pd
from tqdm import tqdm
import numpy as np

omcs_relations = ['Causes', 'HasSubEvent', 'xReason',]
atomic_new_relations = ['isAfter', 'isBefore',  'HinderedBy']
event_relations = omcs_relations + atomic_new_relations

atomic_2020 = {
  "trn": pd.read_csv("../data/atomic2020_data-feb2021/train.tsv", sep="\t"),
  "dev": pd.read_csv("../data/atomic2020_data-feb2021/dev.tsv", sep="\t"),  
  "tst": pd.read_csv("../data/atomic2020_data-feb2021/test.tsv", sep="\t"),
}

atomic_2020_events = {
  "trn":[(atomic_2020["trn"].loc[i][0], atomic_2020["trn"].loc[i][1], atomic_2020["trn"].loc[i][2])\
         for i in range(len(atomic_2020["trn"])) \
         if atomic_2020["trn"].loc[i][1] in event_relations],
  "dev":[(atomic_2020["dev"].loc[i][0], atomic_2020["dev"].loc[i][1], atomic_2020["dev"].loc[i][2])\
         for i in range(len(atomic_2020["dev"])) \
         if atomic_2020["dev"].loc[i][1] in event_relations],
  "tst":[(atomic_2020["tst"].loc[i][0], atomic_2020["tst"].loc[i][1], atomic_2020["tst"].loc[i][2])\
         for i in range(len(atomic_2020["tst"])) \
         if atomic_2020["tst"].loc[i][1] in event_relations],
}

new_atomic_tuples = {"trn":[], "tst":[], "dev":[]}
for r in ["trn", "dev", "tst"]:
    for head, rel, tail in tqdm(atomic_2020_events[r]):
        if rel in atomic_new_relations:
            new_atomic_tuples[r].append((head, rel, tail))
