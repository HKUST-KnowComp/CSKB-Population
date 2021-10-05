# Reproduce DISCOS knowledge graph using current data

In [DISCOS](https://arxiv.org/abs/2101.00154) ([github repo](https://github.com/HKUST-KnowComp/DISCOS-commonsense)), we train a discriminator using ground-truth commonsense triples in ATOMIC and randomly sampled negative triples, and use the model to infer the plausibility of ASER edges to form a new commonsense knowledge base. 

1. Prepare candidates from ASER\_norm. `prepare_candidates.ipynb`
2. Select best models using `eval_automatic.py`
3. Infer using `infer.py`

You could download the results [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EjSTzTPtXsxHsSE6_uYeNRgBEldSCBq569TYF1TcLEIOtQ?e=Fs13Tm). The results is stored in .csv file with 3 columns, `head`, `tail`, and `score`. The `score` column is the scores given by KG-Bert trained on ATOMIC2020. The higher the scores the more plausible a triple is according to the classifier. You can filter the results by setting a threshold for the scores. 