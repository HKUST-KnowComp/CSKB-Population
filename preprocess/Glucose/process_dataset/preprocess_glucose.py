import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../../')
from Glucose.utils.utils import *

if not os.path.exists('../../dataset/'):
    os.mkdir('../../dataset')

assert os.path.isfile(
    '../../dataset/GLUCOSE.csv'), 'Please download the Glucose Dataset and Rename it to ' \
                                  'GLUCOSE.csv, and then move it to the ../dataset/ folder.'

glucose = pd.read_csv('../../dataset/GLUCOSE.csv')
glucose_index2sentence = {i: glucose.loc[i, 'selected_sentence'] for i in glucose.index}
glucose_unique_sentences = glucose['selected_sentence'].unique()
glucose_sentence2uniqueID = {sentence: index for index, sentence in enumerate(glucose_unique_sentences)}
glucose_index2uniqueID = {i: glucose_sentence2uniqueID[glucose_index2sentence[i]] for i in glucose.index}
np.save('../../dataset/glucose_index2sentence.npy', glucose_index2sentence)
np.save('../../dataset/glucose_sentence2uniqueID.npy', glucose_sentence2uniqueID)
np.save('../../dataset/glucose_index2uniqueID.npy', glucose_index2uniqueID)

parsed_stru_dict = {'head': {}, 'tail': {}, 'head_id': {}, 'tail_id': {}}
for j in range(1, 11):
    for k in parsed_stru_dict.keys():
        parsed_stru_dict[k][j] = []

for i in tqdm(glucose.index):
    for j in range(1, 11):
        if glucose.loc[i, '{}_generalStructured'.format(j)] != 'escaped':
            parts = glucose.loc[i, '{}_generalStructured'.format(j)].split('>')
            parsed_stru_dict['head'][j].append(trim(parts[0]))
            parsed_stru_dict['tail'][j].append(trim(parts[2]))
            parsed_stru_dict['head_id'][j].append(glucose_sentence2uniqueID[glucose.loc[i, 'selected_sentence']])
            parsed_stru_dict['tail_id'][j].append(glucose_sentence2uniqueID[glucose.loc[i, 'selected_sentence']])

np.save('../../dataset/Glucose_parsed_stru_dict', parsed_stru_dict)

print("Glucose is ready to use at ../../dataset/")
for i in range(1, 11):
    print("List [{}]: Total Knowledge: {}\tUnique Knowledge: {}".format(i, len(parsed_stru_dict['head'][i]),
                                                                        len(np.unique(parsed_stru_dict['head_id'][i]))))
