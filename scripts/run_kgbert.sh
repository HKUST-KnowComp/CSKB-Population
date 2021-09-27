data_root="data"

if [ ! -d $data_root/models ] 
then
    mkdir $data_root/models
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

CUDA_VISIBLE_DEVICES=$gpu_id python model/BertSAGE/train.py --gpu $gpu_id \
--model kgbert_va --optimizer ADAM --lr 5e-5 --lrdecay 1.0 \
--graph_cache_path $data_root/graph_cache \
--file_path $data_root/G_nodefilter_aser_all_inv_10_shuffle_10_other10_negprop_1.pickle \
--model_dir $data_root/models \
--num_neighbor_samples 4 --epochs 1 \
--tensorboard_dir $data_root/runs \
--target_dataset all --batch_size 32 --save_every_checkpoint --eval_on none 