CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_oEffect_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xAttr_1hop_thresh_100_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xNeed_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_oReact_1hop_thresh_100_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xEffect_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xReact_1hop_thresh_100_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_oWant_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xIntent_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
CUDA_VISIBLE_DEVICES=3 python -u BertSAGE/train.py --model simple --load_edge_types ASER --neg_prop 1 --graph_cach_path data/graph_cache/neg_{}_{}_{}.pickle --negative_sample prepared_neg --file_path data/graph_raw_data/G_aser_xWant_1hop_thresh_20_neg_other_20_inv_10.pickle --encoding_style single_cls_raw --seed 401 --log_name 0308
