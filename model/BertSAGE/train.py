import warnings
warnings.filterwarnings("ignore")
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from dataloader import *
from model import *
from utils import get_logger, get_gpu_usage
import argparse
import pickle
import gc


parser = argparse.ArgumentParser()

# model related
parser.add_argument("--model", default='simple', type=str, required=False,
                    choices=["graphsage", "graphsage_relational",
                    "simple", "simple_relational",
                    "kgbert_va", "kgbert_vb",
                    "kgbertsage_va", "kgbertsage_vb"],
                    help="choose model")
parser.add_argument("--encoder", default='bert', type=str, required=False,
                    choices=["bert", "roberta"],
                    help="choose encoder")
parser.add_argument("--num_layers", default=1, type=int, required=False,
                    help="number of graphsage layers")
parser.add_argument("--num_neighbor_samples", default=4, type=int, required=False,
                    help="num neighbor samples in GraphSAGE")
parser.add_argument("--encoding_style", default='single_cls_trans', type=str, required=False,
                    choices=["pair_cls_trans", "pair_cls_raw", "single_cls_trans", "single_cls_raw", "single_mean"],
                    help="the encoding style of classifier (pair_xxx are not available for graphsage)")
parser.add_argument("--agg_func", default='MEAN', type=str, required=False,
                    choices=["MEAN", "MAX", "ATTENTION"],
                    help="the encoding style of classifier (pair_xxx are not available for graphsage)")
# train related
parser.add_argument("--gpu", default='0', type=str, required=False,
                    help="choose which gpu to use")
parser.add_argument("--optimizer", default='SGD', type=str, required=False,
                    choices=["SGD", "ADAM"],
                    help="optimizer to be used")
parser.add_argument("--lr", default=0.01, type=float, required=False,
                    help="learning rate")
parser.add_argument("--lrdecay", default=0.8, type=float, required=False,
                    help="learning rate decay every 2000 steps")
parser.add_argument("--decay_every", default=500, type=int, required=False,
                    help="show test result every x steps")
parser.add_argument("--test_every", default=250, type=int, required=False,
                    help="show test result every x steps")
parser.add_argument("--batch_size", default=32, type=int, required=False,
                    help="batch size")
parser.add_argument("--epochs", default=3, type=int, required=False,
                    help="batch size")
parser.add_argument("--metric", default='f1', type=str, required=False,
                    choices=["f1", "acc"],
                    help="evaluation metric, either f1 or acc")
parser.add_argument("--save_every_checkpoint", action="store_true",
                        help="whether to save every steps.")
parser.add_argument("--eval_on", default='link_prediction', type=str, required=False,
                    choices=["link_prediction", "human_annotation", "none"],
                    help="evaluate on link prediction or the human annotated tuples")
parser.add_argument("--use_nl_relation", action="store_true",
                        help="whether to use natural language to encode relation")

# data related
parser.add_argument("--highest_aser_rel", action="store_true",
                        help="whether to use select the aser relation with the highest weight")
parser.add_argument("--target_relation", default='all', type=str, required=False,
                    help="target relation. all or one of the commonsense relations.")
parser.add_argument("--target_dataset", default='all', type=str, required=False,
                    choices=["all", "atomic", "glucose", "cn"],
                    help="target dataset. all or one of the commonsense datasets.")
parser.add_argument("--eval_dataset", default='all', type=str, required=False,
                    choices=["all", "atomic", "glucose", "cn"],
                    help="evaluation dataset. Only has effect when target_dataset is"
                    "set to all.")
parser.add_argument("--load_edge_types", default='ASER', type=str, required=False,
                    choices=["CS", "ASER", "CS+ASER"],
                    help="load what edges to data_loader.adj_lists")
parser.add_argument("--negative_sample", default='prepared_neg', type=str, required=False,
                    choices=["prepared_neg", "from_all", "fix_head"],
                    help="nagative sample methods")
parser.add_argument("--neg_prop", default=1.0, type=float, required=False,
                    help="whether to include relation in adj matrix")
parser.add_argument("--save_tokenized", action="store_true", 
                    help="whether to tokenize all nodes first and save them.")

# save paths
parser.add_argument("--graph_cache_path", default="graph_cache",
                    type=str, required=False,
                    help="path of graph cache")
parser.add_argument("--file_path", default='', type=str, required=True,
                    help="load training graph pickle")
parser.add_argument("--model_dir", default='models', type=str, required=False,
                    help="Where to save models.")
parser.add_argument("--tensorboard_dir", default='runs', type=str, required=False,
                    help="the directory to store tensorboard files.")
parser.add_argument("--log_name", default='unnamed', type=str, required=False,
                    help="special names of log files")
# etc
parser.add_argument("--seed", default=401, type=int, required=False,
                    help="random seed")

args = parser.parse_args()
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
lr = args.lr
show_step = args.test_every
batch_size= args.batch_size
num_epochs = args.epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_batch_size = 64
neg_prop = args.neg_prop

file_path = args.file_path

# dataloader path

# whether it's a triple classification task (h, r, t)
rel_in_edge = "rel_in_edge" if ("va" in args.model or "relational" in args.model) else ""
highest_aser_rel = "_highest_aser_rel" if args.highest_aser_rel else ""
use_nl_rel = "_use_nl_rel" if args.use_nl_relation else ""
save_tokenized = "_save_token" if args.save_tokenized else ""

graph_cache = os.path.join(args.graph_cache_path, "neg_{}_{}_{}_{}_{}.pickle")
relation_string = f"{args.target_relation}-{args.target_dataset}-relations"
if args.target_dataset == "all" and args.eval_dataset != "all":
    relation_string += f"-{args.eval_dataset}-eval"
graph_cache = graph_cache.format(f"{args.negative_sample}-{args.neg_prop}",
                                 args.load_edge_types,
                                 os.path.basename(file_path).rsplit(".", 1)[0],
                                 relation_string,
                                 rel_in_edge + highest_aser_rel + use_nl_rel + save_tokenized)
# cache_with_neighbor = graph_cache.rsplit(".", 1)[0] + "+neighbor.pickle"
# if os.path.exists(cache_with_neighbor):
#    graph_cache = cache_with_neighbor

if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
model_dir = os.path.join(
    args.model_dir,
    f"{os.path.basename(graph_cache).rsplit('.', 1)[0]}")
    #f"_{args.target_relation}_negprop{args.neg_prop}")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# Model

if "simple" in args.model:
    model_save_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_{}_seed{}.pth'\
    .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer,
        args.lr, args.lrdecay, args.decay_every, args.metric, args.seed))
    log_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_{}_{}_seed{}.log' \
                            .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer,
                                    args.lr, args.lrdecay, args.decay_every, args.metric, args.log_name, args.seed))
elif "sage" in args.model:  # graphsage, kgbertsage
    model_save_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_layer{}_neighnum_{}_graph_{}_{}_aggfunc{}_seed{}.pth'\
                        .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer, args.lr,
                            args.lrdecay, args.decay_every, args.num_layers, 
                            args.num_neighbor_samples, args.load_edge_types, args.metric, args.agg_func, args.seed))
    log_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_layer{}_neighnum_{}_graph_{}_{}_aggfunc{}_{}_seed{}.log'\
                        .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer, args.lr,
                            args.lrdecay, args.decay_every, args.num_layers,
                            args.num_neighbor_samples, args.load_edge_types, args.metric, args.agg_func, args.log_name, args.seed))
elif "kgbert_" in args.model:
    model_save_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_{}_seed{}.pth'\
    .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer,
        args.lr, args.lrdecay, args.decay_every, args.metric, args.seed))
    log_path = os.path.join(model_dir, '{}_{}_best_{}_bs{}_opt_{}_lr{}_decay{}_{}_{}_{}_seed{}.log' \
                            .format(args.model, args.encoding_style, args.encoder, batch_size, args.optimizer,
                                    args.lr, args.lrdecay, args.decay_every, args.metric, args.log_name, args.seed))

tensorboard_dir = os.path.join(args.tensorboard_dir,
                               os.path.basename(model_dir),
                               os.path.basename(model_save_path).rsplit(".", 1)[0])
os.makedirs(tensorboard_dir, exist_ok=True)

logging = get_logger(log_path)

seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
logging("set random seed = %d for random, PYTHONHASHSEED, numpy, torch" % seed)

logging(graph_cache)
if not os.path.exists(graph_cache):
    id2nodestoken_path = os.path.join(args.graph_cache_path,
                                  "id2nodestoken_" + os.path.basename(file_path))
    s = time.time()
    data_loader = MultiGraphDataset(file_path, device, args.encoder,
        node_token_path=id2nodestoken_path,
        target_relation=args.target_relation, target_dataset=args.target_dataset,
        eval_dataset=args.eval_dataset,
        edge_include_rel=("va" in args.model or "relational" in args.model),
        negative_sample=args.negative_sample, load_edge_types=args.load_edge_types,
        neg_prop=neg_prop,
        highest_aser_rel=args.highest_aser_rel,
        use_nl_rel=args.use_nl_relation,
        save_tokenized=args.save_tokenized)
    with open(graph_cache, "wb") as writer:
        pickle.dump(data_loader,writer,pickle.HIGHEST_PROTOCOL)  
    e = time.time()
    logging(f"after dumping graph cache to {graph_cache}"
            f"\ntime taken: {e - s}")
else:
    s = time.time()
    with open(graph_cache, "rb") as reader:
        data_loader = pickle.load(reader)
    e = time.time()
    logging(f"after loading graph cache from {graph_cache}"
            f"\ntime taken: {e - s}")

if "simple" in args.model:
    model = SimpleClassifier(encoder=args.encoder,
                             adj_lists=data_loader.get_adj_list(),
                             nodes_tokenized=data_loader.get_nodes_tokenized(),
                             nodes_text=data_loader.get_nid2text(),
                             device=device,
                             enc_style=args.encoding_style,
                             num_class=2,
                             include_rel="relational" in args.model,
                             relation_tokenized=data_loader.get_relations_tokenized(),
                             )
elif 'graphsage' in args.model:
    model = LinkPrediction(encoder=args.encoder,
                           adj_lists=data_loader.get_adj_list(),
                           nodes_tokenized=data_loader.get_nodes_tokenized(),
                           device=device,
                           id2node=data_loader.get_nid2text(),
                           num_layers=args.num_layers,
                           num_neighbor_samples=args.num_neighbor_samples,
                           enc_style=args.encoding_style,
                           agg_func=args.agg_func,
                           num_class=2,
                           include_rel="relational" in args.model,
                           relation_tokenized=data_loader.get_relations_tokenized(),
                           )
elif 'kgbert' in args.model:
    model = KGBertClassifier(encoder=args.encoder,
                        adj_lists=data_loader.get_adj_list() if "sage" in args.model else None,
                        nodes_tokenized=data_loader.get_nodes_tokenized(),
                        relation_tokenized=data_loader.get_relations_tokenized(),
                        id2node=data_loader.get_nid2text(),
                        enc_style=args.encoding_style,
                        agg_func=args.agg_func,
                        num_neighbor_samples=args.num_neighbor_samples,
                        device=device,
                        version=args.model,
                        )

# print(model)
criterion = torch.nn.CrossEntropyLoss()
if args.optimizer == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
elif args.optimizer == "ADAM":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()

step = 0

best_valid_acc = 0
best_test_acc = 0   
best_valid_f1 = 0
best_test_f1 = 0 

my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lrdecay)

logging("metric: {}".format(args.metric))
writer = SummaryWriter(log_dir=tensorboard_dir)


for epoch in range(num_epochs):
    for batch in data_loader.get_batch(batch_size=batch_size, mode="train"):
        step += 1
        if step % args.decay_every == 0:
            my_lr_scheduler.step()
        # batch list((node_id1, node_id2))
        edges, labels = batch
        # allocate to right device
        edges = edges.to(device)
        labels = labels.to(device)

        logits = model(edges, edges.shape[0])
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        gpu_usage = get_gpu_usage(int(args.gpu))
        if gpu_usage > 0.9:
            del batch, edges, labels, logits, loss
            gc.collect()
            torch.cuda.empty_cache()
            print(f"The gpu usage was {gpu_usage} > 0.9, so it was released."
                  f"\nAfter release: {get_gpu_usage(int(args.gpu))}")
        # evaluate
        if step % show_step == 0:
            if args.save_every_checkpoint:
                torch.save(model.state_dict(), model_save_path + ".step." + str(step))
            if args.eval_on == "link_prediction":
                val_acc, val_f1 = eval(data_loader, model, test_batch_size, device, "valid")
                test_acc, test_f1 = eval(data_loader, model, test_batch_size, device, "test")
                
                if args.metric == "acc" and val_acc > best_valid_acc:
                        best_valid_acc = val_acc
                        best_test_acc = test_acc
                        best_valid_f1 = val_f1
                        best_test_f1 = test_f1
                        torch.save(model.state_dict(), model_save_path)
                elif args.metric == "f1" and val_f1 > best_valid_f1:
                        best_valid_acc = val_acc
                        best_test_acc = test_acc
                        best_valid_f1 = val_f1
                        best_test_f1 = test_f1
                        torch.save(model.state_dict(), model_save_path)
                logging("epoch {}, step {}, current valid acc: {}, current test acc: {}, current valid f1:{}, current test f1: {}, ".format(epoch, step, val_acc, test_acc, val_f1, test_f1))
                logging("current best val acc: {}, test acc: {} current best f1: {}, test f1: {}".format(best_valid_acc, best_test_acc, best_valid_f1, best_test_f1))
                writer.add_scalars("acc", {"val": val_acc, "test": test_acc}, step)
                writer.add_scalars("f1", {"val": val_f1, "test": test_f1}, step)
            elif args.eval_on == "human_annotation":
                pass
            elif args.eval_on == "none":
                print("epoch {}, step {}, no evaluation".format(epoch, step))
