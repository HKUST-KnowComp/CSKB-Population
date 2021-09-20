import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from aser.extract.aser_extractor import SeedRuleASERExtractor
from tqdm import trange

sys.path.append('../../')
from Glucose.utils.utils import *


def extract_a_structure(stru: str):
    structure = {}
    for part in trim(stru).split(']'):
        if part.split('[')[-1] == '':
            continue
        elif '||' in part:
            structure[part.split('[')[-1]] = trim(part.split('||')[0].replace('{', ''))
        else:
            structure[part.split('[')[-1]] = trim(part.split('[')[0].replace('{', '').replace('}_', '').split('||')[0])
    return structure


def report_human_percentage(subject: list):
    human, total = 0, 0
    for i in subject:
        total += i[1]
        if 'Someone_A\'s' in i[0] or 'Someone_B\'s' in i[0]:
            continue
        elif 'Someone' in i[0]:
            human += i[1]
    return human, total - human


def report_pattern_distribution(glucose_dataset: dict):
    e_extractor = SeedRuleASERExtractor(corenlp_port=args.port, corenlp_path=args.CorenlpPath)
    pattern_dict = {}
    for i in trange(1, 11):
        for ind in glucose_dataset[i].keys():
            if glucose_dataset[i][ind]['total_head']:
                h_ext = e_extractor.extract_from_text(glucose_dataset[i][ind]['total_head'][0])[0][0]
                for event in h_ext:
                    if event.pattern not in pattern_dict.keys():
                        pattern_dict[event.pattern] = 1
                    else:
                        pattern_dict[event.pattern] += 1
            if glucose_dataset[i][ind]['total_tail']:
                t_ext = e_extractor.extract_from_text(glucose_dataset[i][ind]['total_tail'][0])[0][0]
                for event in t_ext:
                    if event.pattern not in pattern_dict.keys():
                        pattern_dict[event.pattern] = 1
                    else:
                        pattern_dict[event.pattern] += 1
    print("The pattern dict is:", pattern_dict)
    return pattern_dict


def plot_subject(h: int, non_h: int, title: str):
    plt.rc('font', family='Times New Roman', size=11)
    size = [h, non_h]
    labels = ['human', 'non-human']
    explode = (0.02, 0.02)
    plt.pie(x=size, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True)
    plt.axis('equal')
    plt.legend()
    plt.title("{}s Subject Distribution".format(title))
    plt.savefig("./{}_subject_distribution.pdf".format(title), bbox_inches='tight')
    plt.clf()
    return


def plot_pattern(pattern: list, glucose_pattern: dict):
    plt.rc('font', family='Times New Roman')
    aser_total = 468.1
    aser = {'s-v': 109, 's-v-o': 129, 's-v-a': 5.2, 's-v-o-o': 3.5, 's-be-a': 89.9, 's-v-be-a': 1.2, 's-v-be-o': 1.2,
            's-v-v-o': 12.4, 's-v-v': 8.7, 's-be-a-p-o': 13.2, 's-v-p-o': 39, 's-v-o-p-o': 27.2, 'spass-v': 15.1,
            'spass-v-p-o': 13.5}
    pattern_inASER = [(i, glucose_pattern[i]) for i in pattern if i in aser.keys()]
    glucose_total = sum([i[1] for i in pattern_inASER])
    aser_pat = [aser[k[0]] / aser_total for k in pattern_inASER]
    glucose_pat = [k[1] / glucose_total for k in pattern_inASER]
    size = len(pattern_inASER)
    x = np.arange(size)
    total_width = 0.8
    n = 2
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(x, glucose_pat, width=width, label='Glucose pattern', tick_label=[''] * len(pattern_inASER))
    plt.bar(x + width, aser_pat, width=width, label='ASER pattern', tick_label=[''] * len(pattern_inASER))
    plt.legend()
    plt.xticks(ticks=x + width / 2, labels=[i[0] for i in pattern_inASER])
    plt.xticks(fontproperties='Times New Roman', size=11, rotation=45)
    plt.yscale("log")
    plt.savefig('./Glucose_Pattern.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9499, help="The Port number stanford NLP will use")
    parser.add_argument("--GlucosePath", type=str, default='../../dataset/Glucose_parsed_stru_dict.npy',
                        help="The path to the glucose dataset in parsed format")
    parser.add_argument("--MatchPath", type=str, default='./Final_Version/glucose_final_matching.npy',
                        help="The path to the matched dict of Glucose-aser")
    parser.add_argument("--CorenlpPath", type=str,
                        default="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
                        help="Path to the Stanford CoreNLP path")
    args = parser.parse_args()
    glucose = np.load(args.GlucosePath, allow_pickle=True).item()
    match = np.load(args.MatchPath, allow_pickle=True).item()
    subject_dict = {'head': {}, 'tail': {}, 'total': {}}
    for i in ['head', 'tail']:
        for j in trange(1, 11):
            for k in trange(len(glucose[i][j])):
                temp_dict = extract_a_structure(glucose[i][j][k])
                if 'subject' in temp_dict.keys():
                    if temp_dict['subject'] in subject_dict[i].keys():
                        subject_dict[i][temp_dict['subject']] += 1
                        subject_dict['total'][temp_dict['subject']] += 1
                    else:
                        subject_dict[i][temp_dict['subject']] = 1
                        subject_dict['total'][temp_dict['subject']] = 1

    total_sort = sorted(subject_dict['total'].items(), reverse=True, key=lambda x: x[1])

    h, noh = report_human_percentage(total_sort)
    plot_subject(h, noh, "Total")
    glucose_dict = report_pattern_distribution(match)
    plot_pattern(sorted(glucose_dict, reverse=True, key=lambda x: glucose_dict[x]), glucose_dict)
