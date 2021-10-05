# CSKB-Population
Codes for the EMNLP2021 paper: [Benchmarking Commonsense Knowledge Base Population with an Effective Evaluation Dataset](https://arxiv.org/abs/2109.07679)

## Dataset Preprocess

### Prerequisite

1. Parsing and aligning of CSKBs with ASER: 

Download ATOMIC data first

`sh data/download_atomic.sh`

2. Then run the preprocessing script for processing ATOMIC, ATOMIC2020, and ConceptNet.

```
cs preprocess
sh prepare_atomic_cn.sh
```

3. Preprocess GLUCOSE.

Checkout the README in `preprocess/Glucose` folder.

4. Merge CSKBs with ASER, get the graph structures of CSKB, stored in `networkx`. Get training graph with negative examples.

```
python cskb_merge.py
python merge_cskb_aser.py
```


## Model Training

Download or finished prepare a networkx graph file for training. E.g., Download [G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EtBpzkm37nhHn0yMXbP89UYBYHNZqIhH5aJn1Iaauf0GoQ?e=tYGA4e) and place it at the `data` folder.

```
data_root="data"

if [ ! -d $data_root/model ] 
then
    mkdir $data_root/model
fi
if [ ! -d $data_root/runs ] 
then
    mkdir $data_root/runs
fi
if [ ! -d $data_root/graph_cache ] 
then
    mkdir $data_root/graph_cache
fi

gpu_id=0

CUDA_VISIBLE_DEVICES=$gpu_id python BertSAGE/train.py --gpu $gpu_id \
--model kgbertsage_va --optimizer ADAM --lr 5e-5 --lrdecay 1.0 \
--graph_cache_path $data_root/graph_cache \
--file_path $data_root/G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle \
--model_dir $data_root/models \
--num_neighbor_samples 4 --epochs 1 \
--tensorboard_dir $data_root/runs \
--target_dataset all --batch_size 32 --save_every_checkpoint --eval_on none 
```

Other scripts are shown in the `scripts` folder.

## Evaluation

Download the annotated [evaluation_set.csv](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/EtBpzkm37nhHn0yMXbP89UYBYHNZqIhH5aJn1Iaauf0GoQ?e=tYGA4e) and put it to the `data` folder. Run the evaluation codes `evaluate.py`.

Some model checkpoints in the paper is uploaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/tfangaa_connect_ust_hk/ElBOmqQ8bm5NjkuaUnrAaj0B1mU0cyFWr3LLlc0tEKvZVg?e=URjyua).

## Reproduce DISCOS using current version of ASER\_norm

Checkout the `DISCOS-reproduce` folder

## Other downloading methods

Data from Baidu Net disk: https://pan.baidu.com/s/1Bu_TlJpk4RFS1k2ezvbt5Q  password: v4ib








