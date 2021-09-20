import os
import sys
import json
import argparse
sys.path.append('../')
import numpy as np
import pandas as pd
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from tqdm import tqdm
from utils.atomic_utils import SUBJ2POSS, PP_SINGLE
from itertools import permutations
from itertools import chain
from utils.utils import *
from multiprocessing import Pool

def prepare_atomic2020():
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
    return new_atomic_tuples

def simple_parse(sent):
    """
      Deal with possessive cases only.
    """
    strs = sent.split()
    new_tokens = []
    for token in strs:
      if token.endswith("'s") and token != "'s":
        new_tokens.append(token[:-2])
        new_tokens.append("'s")
      else:
        new_tokens.append(token)
    return new_tokens


def instantiate_ppn(head, tail):
    """
      Input: Head and Tail in the original CSKG.
        e.g., head = "PersonX takes PersonY's phone",
              tail = "PersonY is sad"
    
      Output: list(tuple())
        A list of tuples with concrete personal pronouns.
        [(he takes her phone, she is sad), (he takes my phone, i am sad), ...]
    """
    # 1. Manually process possessive cases, the most common parsing issue in the dataset.
    head_strs = simple_parse(head)
    tail_strs = simple_parse(tail)
    if not (len(head_strs) > 0 and len(tail_strs) > 0):
        return []        
    
    # 2. dictionary that stores the indecies of "PersonX", "PersonY", and "PersonZ"
    # Personal pronoun dict
    pp_dict = {"head":{}, "tail":{}} 
    for key, strs in {"head":head_strs, "tail":tail_strs}.items():
        for i, word in enumerate(strs):
            if word in ["PersonX", "PersonY", "PersonZ"]:
                pp_dict[key][word] = pp_dict[key].get(word, []) + [i]
    
    # 3. get the PersonX -> he/she permutation
    all_cs_pp = set(pp_dict["head"].keys()) | set(pp_dict["tail"].keys()) 
    # commonsense PPs, PersonX/Y
    num_special_pp = len(all_cs_pp)
    # number of personal pronoun placeholders.
    perm_pp = list(permutations(PP_SINGLE, num_special_pp))
#     print(perm_pp)
    # 4. Replace
    result_tuples = []           
    for perm in perm_pp:
        if len(set(perm)) < num_special_pp :
            # Make sure the permutation contains distinct PPs
            continue
        to_pp_single_dict = {cs_pp:perm[i] for i, cs_pp in enumerate(all_cs_pp)}
        result_tuple = {"head":"", "tail":""}
        for key, strs in {"head":head_strs, "tail":tail_strs}.items():
            replacements = [to_pp_single_dict.get(token, token)\
                                              for i, token in enumerate(strs)]
            # 4.1 Possessive case processing
            if "'s" in strs:
                # Convert I 's -> my, he 's -> his
                for cs_pp in pp_dict[key]: #PersonX/Y/Z
                    for idx in pp_dict[key][cs_pp]:
                        # if this pp need to be substitute
                        if replacements[idx] in SUBJ2POSS and \
                        replacements[min(idx+1, len(replacements)-1)] == "'s":
                            replacements[idx] = SUBJ2POSS[replacements[idx]]
                            replacements[idx+1] = "\REMOVE"
                while "\REMOVE" in replacements:
                    replacements.remove("\REMOVE")            
            # 4.2 dealing with singular formats 
            if "is" in replacements:
                for i in range(len(replacements)-1):
                    if replacements[i] == "i" and replacements[i+1] == "is":
                        replacements[i+1] = "am"
                    if replacements[i] == "you" and replacements[i+1] == "is":
                        replacements[i+1] = "are"

            # if no need to deal with possessive case
            # simply replace PersonX/Y/Z with he/she.
            result_tuple[key] = " ".join(replacements)
        result_tuples.append((result_tuple["head"], result_tuple["tail"]))
    return result_tuples



def fill_sentence(sent, r, has_subject):
    if r in ['xEffect']:
        # + subject
        if has_subject:
            return [sent]
        else:
            return [' '.join(["PersonX", sent])]
    elif r in ['oEffect']:
        if has_subject:
            return [sent]
        else:
            return [' '.join(["PersonY", sent])]
    elif r in ['oReact', 'xReact']:
        # + subject / + subject is
        if has_subject:
            return [sent]
        else:
            if r == "oReact":
                return [' '.join(["PersonY", sent])] + \
                    [' '.join(["PersonY", 'is', sent])]
            else:
                return [' '.join(["PersonX", sent])] + \
                    [' '.join(["PersonX", 'is', sent])]
    elif r in ['xAttr']:
        # + subject is 
        if has_subject:
            return [sent]
        else:
            return [' '.join(["PersonX", 'is', sent])]
    elif r in ['oWant', 'xWant']:
        # + subject want / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if r == "oWant":
                if sent.lower().split()[0] == 'to':
                    return [' '.join(["PersonY", 'want', sent])] \
                                 + [' '.join(["PersonY", " ".join(sent.lower().split()[1:]) ]) ]
                else:
                    return [' '.join(["PersonY", 'want to', sent])] \
                                 + [' '.join(["PersonY", sent]) ]
            else:
                if sent.lower().split()[0] == 'to':
                    return [' '.join(["PersonX", 'want', sent])] \
                                 + [' '.join(["PersonX", " ".join(sent.lower().split()[1:]) ]) ]
                else:
                    return [' '.join(["PersonX", 'want to', sent])] \
                                 + [' '.join(["PersonX", sent]) ]
    elif r in ['xIntent']:
        # + subject intent / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if sent.lower().split()[0] == 'to':
                return [' '.join(["PersonX", 'intent', sent]) ] \
                             + [' '.join(["PersonX", " ".join(sent.lower().split()[1:]) ]) ]
            else:
                return [' '.join(["PersonX", 'intent to', sent]) ]\
                             + [' '.join(["PersonX", sent]) ]
    elif r in ['xNeed']:
        # + subject need / + subject
        if has_subject:
            return [sent]
        else:
            # if start with 'to'
            if sent.lower().split()[0] == 'to':
                return [' '.join(["PersonX", 'need', sent]) ]\
                             + [' '.join(["PersonX", " ".join(sent.lower().split()[1:]) ]) ]
            else:
                return [' '.join(["PersonX", 'need to', sent]) ]\
                             + [' '.join(["PersonX", sent]) ]

def unfold_parse_results(e):
    # return the words of the extractor results
    if len(e) == 0:
        return ""
    if len(e[0]) == 0:
        return ""
    return " ".join(e[0][0].words)
def contain_subject(dependencies):
    return any(dep in [item[1] for item in dependencies] for dep in ['nsubj', 'nsubjpass'])
def process_pp(sent):
    """
        Deal with the situation of "person x", "person y", "personx", "persony"
    """
    fill_words = {"person x":"PersonX", "person y":"PersonY", 
                  "personx":"PersonX", "persony":"PersonY",}
    single_word_filter = {"x":"PersonX", "y": "PersonY"}
    for strs in PP_filter_list:
        if strs in sent:
            sent = sent.replace(strs, fill_words[strs])
    sent_split = sent.split()

    if "x" in sent_split or "y" in sent_split:
      sent = " ".join([single_word_filter.get(item, item) for item in sent_split])
    return sent
  
def parse(atomic_data, r, idx):
    extracted_event_list = [[] for i in range(len(atomic_data))]
    parse_cache = {}
    for i in tqdm(range(idx, len(atomic_data), num_thread)): 
        if i in wc_idx or i in clause_idx:
            continue
        tmp_node = []
        head = atomic_data["event"][i]
        for tail_raw in json.loads(atomic_data[r][i]):
            if tail_raw == 'none':
                continue
            # filter the text
            tail_raw = tail_raw.lower()
            tail_raw = process_pp(tail_raw)            
            parsed_result = e_extractor.parse_text(tail_raw)[0]
            filled_sentences = fill_sentence(tail_raw, r, 
              tail_raw.startswith("PersonX") or tail_raw.startswith("PersonY"))
#                 or contain_subject(parsed_result['dependencies']))
            
            candi_tuples = list(chain(*[instantiate_ppn(head, tail) for tail in filled_sentences]))
            head_dict = {h:"" for h, _ in candi_tuples}
            tail_dict = {t:"" for _, t in candi_tuples}
            for h in head_dict:
                if not h in parse_cache:
                    parse_cache[h] = unfold_parse_results(e_extractor.extract_from_text(h))
                head_dict[h] = parse_cache[h]
            for t in tail_dict:
                if not t in parse_cache:
                    parse_cache[t] = unfold_parse_results(e_extractor.extract_from_text(t))
                tail_dict[t] = parse_cache[t]
            tmp_node.append([
              (
                head_dict[h],
                tail_dict[t],
              ) for h, t in candi_tuples])
        extracted_event_list[i] = tmp_node
        
    return extracted_event_list

def parse_cn(tuples, idx):
    extracted_event_list = [[] for i in range(len(tuples))]
    parse_cache = {}
    for i in tqdm(range(idx, len(tuples), num_thread)):
        head, rel, tail = tuples[i]
        if not isinstance(head, str) or not isinstance(tail, str):
            continue
        if dataset == "conceptnet":
            head = "PersonX " + head.lower()
            tail = "PersonX " + tail.lower()
        
        # filter the text
        tail_raw = process_pp(tail)

        candi_tuples = instantiate_ppn(head, tail)
        head_dict = {h:"" for h, _ in candi_tuples}
        tail_dict = {t:"" for _, t in candi_tuples}
        for h in head_dict:
            if not h in parse_cache:
                parse_cache[h] = unfold_parse_results(e_extractor.extract_from_text(h))
            head_dict[h] = parse_cache[h]
        for t in tail_dict:
            if not t in parse_cache:
                parse_cache[t] = unfold_parse_results(e_extractor.extract_from_text(t))
            tail_dict[t] = parse_cache[t]
        tmp_node= [
          (
            head_dict[h],
            tail_dict[t],
          ) for h, t in candi_tuples]
        extracted_event_list[i] = [tmp_node, rel]
        
    return extracted_event_list

parser = argparse.ArgumentParser()

parser.add_argument("--relation", default='xWant', type=str, required=False,
                    choices=['oEffect', 'oReact', 'oWant', 'xAttr', 
                             'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant'],
                    help="choose which relation to process")
parser.add_argument("--dataset", default='atomic', type=str, required=False,
                    choices=['atomic', 'atomic2020', 'conceptnet'],
                    help="dataset")
parser.add_argument("--port", default=14000, type=int, required=False,
                    help="port of stanford parser")
args = parser.parse_args()

e_extractor = SeedRuleEventualityExtractor(
            corenlp_path = "/data/aser/stanford-corenlp-full-2018-02-27",
            corenlp_port= args.port)
PP_filter_list = ["person x", "person y", "personx", "persony"]  

dataset = args.dataset

if dataset == "atomic":
    atomic_data = pd.read_csv('../data/v4_atomic_all_agg.csv')

    clause_idx = np.load('../data/clause_idx.npy', allow_pickle=True)
    wc_idx = np.load('../data/wildcard_idx.npy', allow_pickle=True)
    num_thread = 5
    relation = args.relation

    # all_results = parse(atomic_data, args.relation, 0)

    num_thread = 5
    workers = Pool(num_thread)
    all_results = []
    for i in range(num_thread):
        tmp_result = workers.apply_async(
            parse, 
            args=(atomic_data, relation, i))
        all_results.append(tmp_result)

    workers.close()
    workers.join()

    all_results = [tmp_result.get() for tmp_result in all_results]
    all_results = [list(chain(*item)) for item in zip(*all_results)]
    if not os.path.exists('../data/new_matching'):
        os.mkdir('../data/new_matching')
    if not os.path.exists('../data/new_matching/ASER-format-words'):
        os.mkdir('../data/new_matching/ASER-format-words')
    np.save('../data/new_matching/ASER-format-words/ATOMIC_'+relation, all_results) 
elif dataset == "conceptnet":
    omcs_tuples = np.load("../data/omcs_tuples.npy", allow_pickle=True)[()]
    for spl in ["trn", "dev", "tst"]:
        num_thread = 5
        # # the maximum number of a thread that the parser supports is 5
        workers = Pool(num_thread)
        all_results = []
        for i in range(num_thread):
            tmp_result = workers.apply_async(
              parse_cn, 
              args=(omcs_tuples[spl], i))
            all_results.append(tmp_result)

        workers.close()
        workers.join()

        all_results = [tmp_result.get() for tmp_result in all_results]
        all_results = [list(chain(*item)) for item in zip(*all_results)]
        if not os.path.exists('../data/new_matching'):
            os.mkdir('../data/new_matching')
        if not os.path.exists('../data/new_matching/ASER-format-words'):
            os.mkdir('../data/new_matching/ASER-format-words')
        np.save("../data/new_matching/ASER-format-words/omcs_{}".format(spl), all_results)
elif dataset == "atomic2020":
#     atomic2020_tuples = np.load("../data/ATOMIC_data/atomic2020_new.npy", allow_pickle=True)[()]
    atomic2020_tuples = prepare_atomic2020()
    for spl in ["dev", "tst", "trn"]:
        num_thread = 5
        # # the maximum number of a thread that the parser supports is 5
        workers = Pool(num_thread)
        all_results = []
        for i in range(num_thread):
            tmp_result = workers.apply_async(
              parse_cn, 
              args=(atomic2020_tuples[spl], i))
            all_results.append(tmp_result)

        workers.close()
        workers.join()

        all_results = [tmp_result.get() for tmp_result in all_results]
        all_results = [list(chain(*item)) for item in zip(*all_results)]
        if not os.path.exists('../data/new_matching'):
            os.mkdir('../data/new_matching')
        if not os.path.exists('../data/new_matching/ASER-format-words'):
            os.mkdir('../data/new_matching/ASER-format-words')
        np.save("../data/new_matching/ASER-format-words/atomic2020_{}".format(spl), all_results)