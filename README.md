# CSKB-Population
Codes for the EMNLP2021 paper: [Benchmarking Commonsense Knowledge Base Population with an Effective Evaluation Dataset](https://arxiv.org/abs/2109.07679)

For data preprocessing and the codes released upon EMNLP 2021, checkout the branch `emnlp2021` (https://github.com/HKUST-KnowComp/CSKB-Population/tree/emnlp2021).

## Environment

We tested our codes on `python 3.8.5`. Here are some packages their corresponding versions.

```
networkx                 2.5
numpy                    1.19.2
pandas                   1.2.2
scikit-learn             0.24.1
scipy                    1.6.3
sklearn                  0.0
spacy                    3.2.0
stanfordnlp              0.2.0
torch                    1.7.1
torch-geometric          1.7.2
torch-scatter            2.0.7
torch-sparse             0.6.9
torchsummary             1.5.1
torchtext                0.8.1
tqdm                     4.56.0
transformers             3.4.0
```

## Dataset Preprocess

See the branch [`emnlp2021`](https://github.com/HKUST-KnowComp/CSKB-Population/tree/emnlp2021)

## Model Training

### Download the data

```
kaggle datasets download -d tianqingfang/ckbp-emnlp2021
```

Or download at https://www.kaggle.com/datasets/tianqingfang/ckbp-emnlp2021.

Put the `train.csv` under `data/ckbp_csv/emnlp2021/`.

The annotated evaluation set is at `data/evaluation_set.csv`.

### Train w/ KG-BERT (BERT-base)

KG-BERT baseline is provided here.

```
CUDA_VISIBLE_DEVICES=0 python models/train_kgbert_baseline.py \
    --ptlm bert-base-uncased \
    --lr 1e-5 \
    --epochs 1 \
    --output_dir results \
    --train_csv_path data/ckbp_csv/emnlp2021/train.csv \
    --relation_as_special_token \
    --save_best_model \
    --seed 100 --batch_size 64 --test_batch_size 128 --save_best_model --experiment_name ""
```

### Access the whole graph data

Check out `Model Training` section on the branch [`emnlp2021`](https://github.com/HKUST-KnowComp/CSKB-Population/tree/emnlp2021#model-training).

## Results

(Note that the evaluation metric is a bit different from that in the paper. In the paper we use a grouped AUC where AUC scores within different groups are calculated and averaged. Here we report the overall AUC across all relations.)

|Model|All|Ori Test Set|CSKB head|ASER edge|
|:--:|:--:|:--:|:--:|:--:|
|KG-BERT (BERT-base) | 62.5 | 74.2 | 51.9 | 54.7|
|KG-BERT (RoBERTa-large) | 70.9 | 78.0 | 63.4 | 64.6|

## Reproduce DISCOS using current version of ASER\_norm

Checkout the [`DISCOS-reproduce`](https://github.com/HKUST-KnowComp/CSKB-Population/tree/emnlp2021/DISCOS-reproduce) folder under the branch [`emnlp2021`](https://github.com/HKUST-KnowComp/CSKB-Population/tree/emnlp2021).








