data_root="/home/data/tfangaa/CKGP/model_data"
# data_root="/data/tfangaa/sehyun"
gpu_id=3

CUDA_VISIBLE_DEVICES=$gpu_id python BertSAGE/train.py --gpu $gpu_id \
--model simple_relational --optimizer ADAM --lr 5e-5 --lrdecay 1.0 \
--test_every 250 --epochs 1 --metric acc \
--graph_cache_path $data_root/data/graph_cache \
--file_path $data_root/data/raw_graph/G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle \
--model_dir $data_root/models \
--num_neighbor_samples 4 \
--tensorboard_dir $data_root/runs \
--target_dataset all --batch_size 32 --save_every_checkpoint --eval_on none 

# --use_nl_relation --highest_aser_rel \
# --rel_in_neighbor \
