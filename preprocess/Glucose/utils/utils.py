import re
import sys
from itertools import permutations

sys.path.append('../../')
from Glucose.utils.glucose_utils import *


def trim(s):
    """
    This function get rid of the empty space at the beginning and end of the string
    :param s: a string to be trimmed
    :return: the string after trimming.
    """
    if s.startswith(' ') or s.endswith(' '):
        return re.sub(r"^(\s+)|(\s+)$", "", s)
    return s


def trim_list(l: list):
    """
    This function trims a list of strings
    :param l: The list of strings to be trimmed
    :return: The list of strings after trimmed
    """
    if not l:
        return None
    else:
        for stri in range(len(l)):
            l[stri] = trim(l[stri])
        return l


def generate_subject_rule(subject: list):
    """
    This function generates a subject replacement words for 4 subject list, Someone A,B,C,D.
    :param subject: A list containing all the subjects we'll need. This is used by taking the length of this list and
    see how many numbers of subject we'll need.
    :return: A list of all possible situations for the subject replacement, for example, ['I', 'you', 'he', 'she']
    """
    lis = []
    for i in permutations(glucose_subject_list):
        if list(i[:len(subject)]) not in lis:
            lis.append(list(i[:len(subject)]))
    return lis


def generate_rule():
    """
    This function generate a list of replacement dictionaries.
    :return: A list of replacement dicts. The key for each dictionary is the words to be replaced,
    and its corresponding value is what we are trying to find in the string and replace it by the value.
    For example, PART of the dict will be {'they': ['Some people_A']}
    """
    rules = []
    subject_list = [someone_a_list, someone_b_list, someone_c_list, someone_d_list]
    object_list = [single_thing_a_list, single_thing_b_list, single_thing_c_d_list]
    replacing_sub_rules = generate_subject_rule(subject_list)
    replacing_obj_rules = list(permutations(glucose_object_list))
    for obj_rule in replacing_obj_rules:
        for sub_rule in replacing_sub_rules:
            rule_based_dict = {'they': single_groups_list + single_group_list,
                               'there': single_places_list + single_place_list}
            for index, obj_rep in enumerate(obj_rule):
                rule_based_dict[obj_rep] = object_list[index]
            for ind, i in enumerate(sub_rule):
                rule_based_dict[i] = subject_list[ind]
            rules.append(rule_based_dict)
    return rules


rule = generate_rule()
