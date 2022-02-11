# Acquired Data Format

networkx DiGraph

hid: The re-indexed id by the "selected_sentence" column. From 0 to 22720. Split train/dev/test based on this id.

list_id: the index of the dimension. From 1 to 10.

# GLUCOSE Preprocessing Instruction

**This is the entire preprocessing instruction for the dataset [GLUCOSE](https://arxiv.org/pdf/2009.07758.pdf) in our
CKGP task.**

You can use this [download link](https://tinyurl.com/yyeo92pt) in the original paper to get the dataset.

Use the following command to install all the required packages in this project. The ASER library can be installed by
following [this instruction](https://github.com/HKUST-KnowComp/ASER).

```commandline
pip install -r requirements.txt
```

We assume that there is a folder `dataset` under the top main directory (at the same level with `Glucose` folder).
Before you start, you should download the dataset, rename it to GLUCOSE.csv, and put it into the `dataset` folder.

The code directory is shown as below:

```
Glucose
│   LICENSE
│   README.md
│
├───build_graph
│       build_glucose_aser_subgraph.py
│       build_glucose_graph.ipynb
│       build_glucose_graph.py
│       build_glucose_graph_tq.ipynb
│       conceptualize_ASER_merge_Glucose.ipynb
│       conceptualize_ASER_merge_Glucose.py
│
├───process_dataset
│       glucose_parsed_matching.py
│       glucose_statistics.py
│       merge_match_result.py
│       preprocess_glucose.py
│
├───update_graph
│       postprocess_glucose_graph.ipynb
│
└───utils
        aser_to_glucose.py
        glucose_utils.py
        utils.py
```

## 1. Extract the knowledge in GLUCOSE

Use the following command to preprocess the original csv dataset.

```commandline
cd ./process_dataset/
python preprocess_glucose.py
```

This will generate a dict containing the head, tail, head_id, tail_id seperated GLUCOSE dataset. For this task, we use
the general structured language as our parsing target.

## 2. Parsed Matching

Before parsing, you'll have to prepare the Stanford CoreNLP parser
from [this webpage](https://stanfordnlp.github.io/CoreNLP/) and the spacy parser from [here](https://spacy.io/usage).

You'll also need the parser and merged eventuality cache from [ASER](https://hkust-knowcomp.github.io/ASER/). Move the
merged event to the `dataset` folder and name it as `merged_event.npy`, the file can be downloaded
at [here](https://hkustconnect-my.sharepoint.com/:u:/g/personal/wwangbw_connect_ust_hk/EQE6rYNuzTRImQ4O4NuemM0BPdVKd9Un5_C1lhnJsKUT-A?e=WbxQNW).

To reduce the parsing time, we separated the parsing process for each GLUCOSE list. The thorough parsing process will
take less than 20 days without separating, separating it into 10 procedures can reduce the time to 1 day.

For example, for list 1 and 2, you should:

```commandline
python glucose_parsed_matching.py --CorenlpPath [Your Path to CoreNLP library] --start 1 --end 1 --spacy 1
python glucose_parsed_matching.py --CorenlpPath [Your Path to CoreNLP library] --start 1 --end 1 --spacy 0

python glucose_parsed_matching.py --CorenlpPath [Your Path to CoreNLP library] --start 2 --end 2 --spacy 1
python glucose_parsed_matching.py --CorenlpPath [Your Path to CoreNLP library] --start 2 --end 2 --spacy 0
```

you should do this for all the remaining lists 3 - 10.

## 3. Merging the parsed result

After parsing all 10 lists, you should get 20 `npy` files for 10 lists, separated in `spacy` and `nonspacy` folders.

Run this command to generate the final matching result of GLUCOSE:

```commandline
python merge_match_result.py
```

## 4. Build GLUCOSE Graph

We've done the parsing preprocessing part for GLUCOSE, next, we'll build a directional graph.

Run the following command:

```commandline
cd ../build_graph/
python build_glucose_graph.py
```

This will generate a file `G_Glucose.pickle` in the `dataset` folder and `build_graph` folder.

Each edge has several attributes:

- `head`: The head of the knowledge tuple
- `tail`: The tail of the knowledge tuple
- `dataset`: The dataset this knowledge tuple belongs to
- `relation`: The relation this knowledge tuple belongs to
- `list_id`: The relation list index this knowledge belongs to in Glucose
- `hid`: The unique index of the selected sentence as the origin of this knowledge tuple
- `tid`: Equal as `hid`

*Note that the unique index for selected sentence can be retrieved by*

```python
{ind: i for ind, i in enumerate(list(glucose['selected_sentence'].unique()))}
```

## 5. Conceptualize ASER and merge with GLUCOSE

You need to prepare the networkx based ASER graph by adding all the nodes and edges from original ASER into a networkx
graph called `G_aser_core.pickle` under the `dataset` folder.

Then create a unique node->id matching dictionary for all the nodes in ASER graph, and save it
as `ASER_core_node2id.npy` under the `dataset` folder.

Finally, run the command to general all the required graph.

```commandline
python conceptualize_ASER_merge_Glucose.py
```

## 6. Matching the relation in GLUCOSE to ATOMIC

As indicated in the GLUCOSE paper, the relations defined in GLUCOSE can be matched to the ATOMIC relations.
In `update_graph` folder, we provide a `postprocess_glucose_graph.ipynb` which demostrates the process of converting the
GLUCOSE relation based graph to ATOMIC relations based graph.

You can read the jupyter notebook yourself and process it to the ATOMIC form. In our experiment, we use the ATOMIC
relations based GLUCOSE graph as our final approach.
