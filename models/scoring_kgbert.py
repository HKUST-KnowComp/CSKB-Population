import os
import sys
import torch
import time
import random
import argparse
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
sys.path.append(os.getcwd())

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

from models.model import KGBERTClassifier
from models.model_utils import score_triples
from models.dataloader import CKBPDataset
from transformers import AutoTokenizer

from utils.ckbp_utils import special_token_list

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()


    # model related
    group_model = parser.add_argument_group("model configs")
    group_model.add_argument("--ptlm", default='bert-base-uncased', type=str, 
                        required=False, help="path to the trained model.")
    group_model.add_argument("--model_path", default='', type=str, 
                        required=False, help="path to the model")
    group_model.add_argument("--tokenizer", default='', type=str, 
                        required=False, help="path to the tokenizer")


    # training-related args
    group_trainer = parser.add_argument_group("training configs")

    group_trainer.add_argument("--device", default='cuda', type=str, required=False,
                    help="device")
    group_trainer.add_argument("--test_batch_size", default=32, type=int, required=False,
                        help="test batch size")
    group_trainer.add_argument("--max_length", default=30, type=int, required=False,
                        help="max_seq_length of h+r+t")
    

    # IO-related

    group_data = parser.add_argument_group("IO related configs")
    group_data.add_argument("--output_dir", default="results",
                        type=str, required=False,
                        help="where to output.")
    group_data.add_argument("--evaluation_file_path", default="data/evaluation_set.csv", 
                            type=str, required=False)
    group_data.add_argument("--log_dir", default='logs', type=str, required=False,
                        help="Where to save logs.") 
    group_data.add_argument("--scorer_name", default='', type=str, required=False,
                        help="Name of the scorer.") 
    
    group_data.add_argument("--seed", default=401, type=int, required=False,
                    help="random seed")

    args = parser.parse_args()

    return args

def main():


    # get all arguments
    args = parse_args()

    save_dir = os.path.join(args.output_dir, args.scorer_name )
    os.makedirs(save_dir, exist_ok=True)

    logger = logging.getLogger("kg-bert scorer")
    handler = logging.FileHandler(os.path.join(save_dir, f"log.txt"))
    logger.addHandler(handler)

    # set random seeds
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # load model
    model = KGBERTClassifier(args.ptlm).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model.model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(args.model_path))

    sep_token = tokenizer.sep_token

    # load data

    infer_file = pd.read_csv(args.evaluation_file_path)

    val_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': 5
    }


    eval_dataset = CKBPDataset(infer_file, tokenizer, args.max_length, sep_token=sep_token) 

    eval_dataloader = DataLoader(eval_dataset, **val_params, drop_last=False)

    # model training
    
    criterion = torch.nn.CrossEntropyLoss()

    predicted_scores = score_triples(tokenizer, model, args.device, eval_dataloader, model_type="kgbert")

    infer_file["score"] = predicted_scores

    pd.DataFrame(infer_file["score"]).to_csv(os.path.join(save_dir, os.path.basename(args.evaluation_file_path)), index=False)

if __name__ == "__main__":
    main()
