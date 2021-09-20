import os
import sys

import numpy as np
import pandas as pd
from tqdm import trange

sys.path.append('../../')
from Glucose.utils.utils import trim

save_path = "parsing_result"
replace_it_by_something = True

assert os.path.exists(os.path.join('./', save_path)), "No Matching Folder detected, please do the parsed matching first"

structure = np.load('../../dataset/Glucose_parsed_stru_dict.npy', allow_pickle=True).item()
glucose = pd.read_csv('../../dataset/GLUCOSE.csv', index_col=None)
glucose_index2uniqueID = np.load('../../dataset/glucose_index2uniqueID.npy', allow_pickle=True).item()
glucose_uniqueID2index = {glucose_index2uniqueID[k]: k for k in glucose_index2uniqueID.keys()}

total_dict = {}
report = open(os.path.join('./', save_path, 'Final_Report.txt'), 'w')


def replace_it_in_list(original: list):
    """
    This function replace all the 'it' to 'something' because 'it' is to general in knowledge.
    :param original: A list of strings to be replaced.
    :return: A list of strings after the replacement.
    """
    result = []
    for i in original:
        word = i.split(' ')
        result.append(" ".join(['something' if (w == 'it' or w == 'It') else w for w in word]))
    return result


total_statistic = {'total': [], 'head': [], 'tail': [], 'both': []}
for i in trange(1, 11):
    total_dict[i] = {}
    try:
        sp = np.load('./{}/spacy/glucose_aser_1_{}_{}.npy'.format(save_path, i, i), allow_pickle=True).item()
        nonsp = np.load('./{}/nonspacy/glucose_aser_0_{}_{}.npy'.format(save_path, i, i), allow_pickle=True).item()
    except FileNotFoundError:
        print("\nWARNING: File not found for List {}\n".format(i))
        continue
    for ind in sp[i].keys():
        total_dict[i][ind] = {'head': [], 'tail': [], 'both': [], 'total_head': [], 'total_tail': [],
                              'head_id': structure['head_id'][i][ind], 'tail_id': structure['tail_id'][i][ind],
                              'original_head': trim(glucose.loc[glucose_uniqueID2index[
                                                                    structure['head_id'][i][
                                                                        ind]], '{}_generalNL'.format(i)].split('>')[0]),
                              'original_tail': trim(glucose.loc[glucose_uniqueID2index[
                                                                    structure['tail_id'][i][
                                                                        ind]], '{}_generalNL'.format(i)].split('>')[
                                                        -1])}
        for index in range(len(sp[i][ind]['head'])):
            matched_head = list(np.unique(sp[i][ind]['head'][index] + nonsp[i][ind]['head'][index]))
            matched_tail = list(np.unique(sp[i][ind]['tail'][index] + nonsp[i][ind]['tail'][index]))
            if replace_it_by_something:
                if any([something in total_dict[i][ind]['original_head'] for something in
                        ['Something_A', 'Something_B', 'Something_C', 'Something_D']]):
                    matched_head = replace_it_in_list(
                        list(np.unique(sp[i][ind]['head'][index] + nonsp[i][ind]['head'][index])))
                if any([something in total_dict[i][ind]['original_tail'] for something in
                        ['Something_A', 'Something_B', 'Something_C', 'Something_D']]):
                    matched_tail = replace_it_in_list(
                        list(np.unique(sp[i][ind]['tail'][index] + nonsp[i][ind]['tail'][index])))
            if matched_head:
                total_dict[i][ind]['head'].append(matched_head)
            if matched_tail:
                total_dict[i][ind]['tail'].append(matched_tail)
            if matched_tail and matched_head:
                for h in matched_head:
                    for t in matched_tail:
                        total_dict[i][ind]['both'].append((h, t))
        new_heads, new_tails = [], []
        for h in total_dict[i][ind]['head']:
            new_heads.extend(h)
        for t in total_dict[i][ind]['tail']:
            new_tails.extend(t)
        total_dict[i][ind]['total_head'] = list(np.unique(new_heads))
        total_dict[i][ind]['total_tail'] = list(np.unique(new_tails))

    list_length = len(total_dict[i])
    list_head = sum([1 for ind in total_dict[i].keys() if total_dict[i][ind]['total_head'] != []])
    list_head_node = sum(len(total_dict[i][ind]['total_head']) for ind in total_dict[i].keys())
    list_tail = sum([1 for ind in total_dict[i].keys() if total_dict[i][ind]['total_tail'] != []])
    list_tail_node = sum(len(total_dict[i][ind]['total_tail']) for ind in total_dict[i].keys())
    list_both = sum([1 for ind in total_dict[i].keys() if total_dict[i][ind]['both'] != []])
    list_both_tuple = sum([len(total_dict[i][ind]['both']) for ind in total_dict[i].keys()])

    report.write(
        "For List {}:\nTotal Heads & Tails: {}\nMatched Heads: {}  Matched Head Nodes: {}  Matched Percentage: {}%  Average Matched Node: {}\nMatched Tail: {}  Matched Tail Nodes: {}  Matched Percentage: {}%  Averaged Matched Node: {}\nBoth Matched: {}  Both Matched Percentage: {}%\n\n".format(
            i, list_length, list_head, list_head_node, 100 * round(list_head / list_length, 4),
            round(list_head_node / list_head, 4), list_tail, list_tail_node, 100 * round(list_tail / list_length, 4),
            round(list_tail_node / list_tail, 4), list_both_tuple, 100 * round(list_both / list_length, 4)))
    total_statistic['total'].append(list_length)
    total_statistic['head'].append((list_head, list_head_node))
    total_statistic['tail'].append((list_tail, list_tail_node))
    total_statistic['both'].append((list_both, list_both_tuple))

report.write(
    "\nTotal Statistics:\nTotal Heads & Tails: {}\nMatched Heads: {}  Matched Head Nodes: {}  Matched Percentage: {}%  Averaged Matched Node: {}\nMatched Tail: {}  Matched Tail Nodes: {}  Matched Percentage: {}%  Average Matched Node: {}\nBoth Matched Nodes: {}  Both Matched Percentage: {}%\n\n".format(
        sum(total_statistic['total']), sum([i[0] for i in total_statistic['head']]),
        sum([i[1] for i in total_statistic['head']]),
        100 * round(sum([i[0] for i in total_statistic['head']]) / sum(total_statistic['total']), 4),
        round(sum([i[1] for i in total_statistic['head']]) / sum([i[0] for i in total_statistic['head']]), 4),
        sum([i[0] for i in total_statistic['tail']]), sum(i[1] for i in total_statistic['tail']),
        100 * round(sum([i[0] for i in total_statistic['tail']]) / sum(total_statistic['total']), 4),
        round(sum(i[1] for i in total_statistic['tail']) / sum([i[0] for i in total_statistic['tail']]), 4),
        sum([k[1] for k in total_statistic['both']]),
        100 * round(sum([k[0] for k in total_statistic['both']]) / sum(total_statistic['total']), 4)))

np.save(os.path.join('./', save_path, 'glucose_final_matching.npy'), total_dict)
np.save('../../dataset/glucose_final_matching.npy', total_dict)
report.close()
