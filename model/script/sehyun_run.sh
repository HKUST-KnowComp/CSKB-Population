# sh run_exp.sh 1 all 2 32

data_root="/data/tfangaa/sehyun"

CUDA_VISIBLE_DEVICES=$1 python BertSAGE/train.py --gpu $1 \
--model kgbertsage_va --optimizer ADAM --lr 3e-5 --lrdecay 1.0 \
--test_every 500 --epochs $3 --metric acc \
--graph_cache_path $data_root/data/graph_cache \
--file_path ~/sehyun/repos/CKGP/atomic_preproc/data/merged_data/G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle \
--model_dir $data_root/models \
--tensorboard_dir $data_root/runs \
--target_dataset $2 --batch_size $4

/data/tfangaa/sehyun/data/graph_cache/neg_prepared_neg-1.0_ASER_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1_all-all-relations_rel_in_edge+neighbor.pickle
~/sehyun/repos/CKGP/atomic_preproc/data/merged_data/G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle

/data/tfangaa/sehyun/data/graph_cache/id2nodestoken_G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle


