import random
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, RobertaModel, BertTokenizer

MAX_SEQ_LENGTH = 30
PAD_VAL = 0

def eval(data_loader,
         model,
         test_batch_size,
         device,
         mode="test",
         is_train=True,
         get_accuracy=False):
    pred_y = []
    gt_y = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader.get_batch(batch_size=test_batch_size, mode=mode):
            edges, labels = batch
            # allocate to right device
            edges = edges.to(device)
            labels = labels.to(device)

            logits = model(edges, edges.shape[0])  # (batch_size, 2)
            predicted = torch.max(logits, dim=1)[1]
            if edges.shape[1] == 3:
                # check
                predicted = torch.where(predicted == 1, edges[:, 2], 0)
                labels = torch.where(labels == 1, edges[:, 2], 0)

            pred_y.extend(predicted.cpu().tolist())
            gt_y.extend(labels.cpu().tolist())
    if is_train:
        model.train()
        get_accuracy = True
    if get_accuracy:
        report = classification_report(gt_y, pred_y, output_dict=True)
        return report["accuracy"], report["macro avg"]["f1-score"]
    else:
        included_rels = sorted(set(pred_y + gt_y))
        label_names = ["neg"] + [data_loader.id2rel[r] for r in included_rels if r != 0]
        return classification_report(gt_y, pred_y, target_names=label_names, digits=4)

def pad_to_max(seq, val=PAD_VAL):
    return pad_sequence(seq, padding_value=val, batch_first=True)[:, :MAX_SEQ_LENGTH]

class LinkPrediction(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, device,
                 num_layers=1, num_neighbor_samples=10, enc_style='single_cls_raw',
                 agg_func='MEAN', num_class=2, include_rel=False, relation_tokenized=None):
        super(LinkPrediction, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        if include_rel:
            assert relation_tokenized is not None
        self.graph_model = GraphSage(
            encoder=encoder,
            num_layers=num_layers,
            input_size=768,
            output_size=768,
            adj_lists=adj_lists,
            nodes_tokenized=nodes_tokenized,
            device=device,
            agg_func=agg_func,
            num_neighbor_samples=num_neighbor_samples,
            enc_style=enc_style,
            relation_tokenized=relation_tokenized)

        emb_dim = 768 * (3 if include_rel else 2)
        self.link_classifier = Classification(emb_dim, num_class, device)

    def forward(self, edges, b_s):
        if isinstance(edges, torch.Tensor):
            # previous version
            if edges.shape[1] == 3:
                nodes = edges[:, :2]
                relations = edges[:, 2]
                all_nodes = nodes.reshape([-1])
                data = [all_nodes, relations]
            else:
                data = edges.reshape([-1])
        elif isinstance(edges, list):
            # current version. Include nodes not in node2id
            data = []
            all_nodes = []
            relations = []
            for head, tail, relation in edges:
                all_nodes.append(head)
                all_nodes.append(tail)
                relations.append(relation)
            data = [all_nodes, relations]

        embs = self.graph_model(data)  # (2*batch_size, emb_size)
        if isinstance(embs, list):  # edge included relation
            node_embs, rel_embs = embs
            node_embs = node_embs.reshape([b_s, -1])
            embs = torch.cat([node_embs, rel_embs], 1)
        else:
            embs = embs.reshape([b_s, -1])
        logits = self.link_classifier(embs)  # (batch_size, 2*emb_size)

        return logits


class KGBertClassifier(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, relation_tokenized,
                 device, enc_style="single_cls_trans", agg_func="mean", version="kgbert_va",
                 num_neighbor_samples=4):
        super().__init__()
        self.nodes_tokenized = nodes_tokenized
        self.relation_tokenized = relation_tokenized
        self.device = device
        self.enc_style = enc_style
        self.agg_func = agg_func
        self.version = version
        self.num_neighbor_samples = num_neighbor_samples
        self.adj_lists = adj_lists
        self.emb_size = 768  # bert's embedding size
        if encoder == "bert":
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
        if "kgbertsage" in version:
            # custom sage layer
            self.sage_layer = nn.Linear(self.emb_size * 3, self.emb_size).to(device)
        if version[-1] == "a":
            self.link_classifier = Classification(self.emb_size, 2, device)
        else:
            self.link_classifier = Classification(self.emb_size, 10, device)

        # used for aggregate function
        self.fill_tensor = torch.nn.Parameter(torch.zeros(1, 768)).to(self.device)
        self.bilinear = torch.nn.Parameter(torch.rand(768, 768)).to(self.device)

    def get_roberta_embs(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.roberta_model(tokens['input_ids'],
                                     attention_mask=tokens['attention_mask'],
                                     token_type_ids=tokens['token_type_ids'])

        if 'cls_raw' in self.enc_style:
            return outputs[0][:, 0, :]
        if 'cls_trans' in self.enc_style:  # with one more linear layer
            return outputs[1]
        if 'mean' in self.enc_style:
            mask = tokens['attention_mask']
            sum_mask = torch.sum(mask, dim=1).unsqueeze(-1).expand(mask.shape)
            norm_mask = mask / sum_mask
            norm_mask = norm_mask.unsqueeze(-1).expand(outputs[0].shape)
            return torch.sum(outputs[0] * norm_mask, dim=1)

    def forward(self, edges, batch_size):
        """
            edges: tensor (batch_size, 3)  # (head, tail, relation)
        """
        tokens = {}
        sentences = []  # (batch_size, max_length)
        if self.version[-1] == "a":
            for (head, tail, relation) in edges:
                if isinstance(head, int) or isinstance(head, torch.Tensor):
                    head = self.nodes_tokenized[int(head)][:-1].to(self.device)  # remove last [SEP] token
                else:
                    head = self.tokenizer(head)[:-1].to(self.device) # remove last [SEP] token
                rel = self.relation_tokenized[int(relation)].to(self.device)
                if isinstance(tail, int) or isinstance(tail, torch.Tensor):
                    tail = self.nodes_tokenized[int(tail)][1:].to(self.device)  # remove first [CLS] token
                else:
                    tail = self.tokenizer(tail)[1:].to(self.device)
                sentences.append(torch.cat([head, rel, tail]))
        else:  # version b
            for (head, tail) in edges:
                head = self.nodes_tokenized[int(head)].to(self.device)
                tail = self.nodes_tokenized[int(tail)][1:].to(self.device)  # remove first [CLS] token
                sentences.append(torch.cat([head, tail]))

        attention_mask = [torch.ones_like(input_id) for input_id in sentences]
        tokens['input_ids'] = pad_to_max(sentences).to(self.device)
        tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
        tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])
        embs = self.get_roberta_embs(tokens)  # (batch_size, emb_size)

        if "kgbertsage" in self.version:
            aggregate_feats = self._get_neighbor_feats(edges, batch_size)

            all_feats = torch.cat([embs, aggregate_feats], dim=1)  # [batch_size, emb_size * 3]
            embs = F.relu(self.sage_layer(all_feats))

        logits = self.link_classifier(embs)  # (batch_size, emb_size)

        return logits

    def _get_neighbor_feats(self, edges, batch_size):
        sampled_neighbors = self._sample_neighbors(edges, num_sample=self.num_neighbor_samples)

        all_neighbors = list(dict.fromkeys(neigh for neigh in chain(*sampled_neighbors)))
        neighbor2id = {neighbor: idx for idx, neighbor in enumerate(all_neighbors)}

        neigh_tokens = {}
        neigh_input_ids = []
        if self.version[-1] == "a":
            for node_id, (neigh_id, rel_id) in all_neighbors:
                node = self.nodes_tokenized[int(node_id)][:-1].to(self.device)
                rel = self.relation_tokenized[int(rel_id)].to(self.device)
                neighbor_node = self.nodes_tokenized[int(neigh_id)][1:].to(self.device)
                neigh_input_ids.append(torch.cat([node, rel, neighbor_node]))
        else:
            for node_id, neigh_id in all_neighbors:
                node = self.nodes_tokenized[int(node_id)].to(self.device)
                neighbor_node = self.nodes_tokenized[int(neigh_id)][1:].to(self.device)
                neigh_input_ids.append(torch.cat([node, neighbor_node]))

        neigh_attention_mask = [torch.ones_like(input_id) for input_id in neigh_input_ids]
        neigh_tokens['input_ids'] = pad_to_max(neigh_input_ids).to(self.device)
        neigh_tokens['attention_mask'] = pad_to_max(neigh_attention_mask).to(self.device)
        neigh_tokens['token_type_ids'] = torch.zeros_like(neigh_tokens['input_ids'])

        neigh_embs = self.get_roberta_embs(neigh_tokens)  # [len(all_neighbors), embed size]
        aggregate_feats = self.aggregate(sampled_neighbors, neigh_embs, neighbor2id)  # [batch_size * 2, emb_size]
        aggregate_feats = aggregate_feats.reshape([batch_size, -1])  # [batch_size, emb_size * 2]

        return aggregate_feats

    def _sample_neighbors(self, edges, num_sample=10):
        """
        if neighbor include rel,
            samp_neighs: list of list of (node, (neighbor, rel)) tuples.
        else:
            samp_neighs: list of list of (node, neighbor) tuples.
        """
        neighbors_list = []
        for edge in edges:
            if isinstance(edge[0], int) or isinstance(edge[0], torch.Tensor):
                neighbors_list.append(self.adj_lists[int(edge[0])])  # head
            else:
                neighbors_list.append([])  # head
            if isinstance(edge[1], int) or isinstance(edge[1], torch.Tensor):
                neighbors_list.append(self.adj_lists[int(edge[1])])  # tail
            else:
                neighbors_list.append([])  # tail
        if num_sample is not None:
            samp_neighs = [random.choices(neighbors, k=num_sample) if len(neighbors) > 0 else []
                           for neighbors in neighbors_list]
        else:
            samp_neighs = neighbors_list

        assert len(samp_neighs) == 2 * len(edges)
        sampled_neighbors = []
        for idx, neighbors in enumerate(samp_neighs):
            if idx % 2:  # odd idx means it is the neighbor of head
                # sampled_neighbors.append([(edges[idx // 2][0].item(), n) for n in neighbors])  # head
                h = edges[idx // 2][0]
                if isinstance(h, int) or isinstance(h, torch.Tensor):
                    sampled_neighbors.append([(int(h), n) for n in neighbors])  # head
                else:
                    # Node doesn't exist in the graph. no neighbor
                    sampled_neighbors.append([])
            else:
                # sampled_neighbors.append([(edges[idx // 2][1].item(), n) for n in neighbors])  # tail
                t = edges[idx // 2][1]
                if isinstance(t, int) or isinstance(t, torch.Tensor):
                    sampled_neighbors.append([(int(t), n) for n in neighbors])  # tail
                else:
                    sampled_neighbors.append([])

        return sampled_neighbors

    def aggregate(self, neighbors_list, neigh_embs, neighbor2id, self_feats=None):
        """
        This is the same as graphsage.aggregate except for the arg names and
        the neighbors_list is list of list of (head, neighbor) tuple.
        """
        if self.agg_func == 'MEAN':
            agg_list = [torch.mean(neigh_embs[[int(neighbor2id[n]) for n in neighbors]], dim=0).unsqueeze(0) \
                            if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'MAX':
            agg_list = [
                torch.max(neigh_embs[[int(neighbor2id[n]) for n in neighbors]], dim=0).values.unsqueeze(0) \
                    if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'ATTENTION':
            # print("here!")
            # assert self_feats != None, "attentive aggregation requires self features"
            agg_list = []
            # self_feats: 128*768
            for neighbors, self_feat in zip(neighbors_list, self_feats):
                if len(neighbors) == 0:
                    agg_list.append(self.fill_tensor)
                    continue
                neighbor_emb = neigh_embs[[int(neighbor2id[n]) for n in neighbors]]  # n_neighbor, embeddings
                # neighbor_emb: 4*768
                # p rint(self_feat.shape)
                attn = torch.mm(torch.mm(neighbor_emb, self.bilinear), self_feat.unsqueeze(-1)).squeeze(-1)
                attn_s = torch.softmax(attn, 0)
                # print(attn_s.shape, neighbor_emb.shape)
                agg_res = torch.sum((neighbor_emb.T * attn_s), dim=-1)
                agg_list.append(agg_res.unsqueeze(0))

            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)


class BaselineClassifier(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, relation_tokenized,
                 device, enc_style="single_cls_raw", agg_func="mean", version="transE",
                 num_neighbor_samples=4):
        super().__init__()
        self.nodes_tokenized = nodes_tokenized
        self.relation_tokenized = relation_tokenized
        self.device = device
        self.enc_style = enc_style
        self.agg_func = agg_func
        self.version = version
        self.num_neighbor_samples = num_neighbor_samples
        self.adj_lists = adj_lists
        self.emb_size = 768  # bert's embedding size
        if encoder == "bert":
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
        if "kgbertsage" in version:
            # custom sage layer
            self.sage_layer = nn.Linear(self.emb_size * 3, self.emb_size).to(device)
        if version[-1] == "a":
            self.link_classifier = Classification(self.emb_size, 2, device)
        else:
            self.link_classifier = Classification(self.emb_size, 10, device)

        # used for aggregate function
        self.fill_tensor = torch.nn.Parameter(torch.rand(1, 768)).to(self.device)
        self.bilinear = torch.nn.Parameter(torch.rand(768, 768)).to(self.device)

    def get_roberta_embs(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.roberta_model(tokens['input_ids'],
                                     attention_mask=tokens['attention_mask'],
                                     token_type_ids=tokens['token_type_ids'])

        if 'cls_raw' in self.enc_style:
            return outputs[0][:, 0, :]
        if 'cls_trans' in self.enc_style:  # with one more linear layer
            return outputs[1]
        if 'mean' in self.enc_style:
            mask = tokens['attention_mask']
            sum_mask = torch.sum(mask, dim=1).unsqueeze(-1).expand(mask.shape)
            norm_mask = mask / sum_mask
            norm_mask = norm_mask.unsqueeze(-1).expand(outputs[0].shape)
            return torch.sum(outputs[0] * norm_mask, dim=1)

    def forward(self, edges, batch_size):
        """
            edges: tensor (batch_size, 3)  # (head, tail, relation)
        """
        tokens = {}
        sentences = []  # (batch_size, max_length)
        if self.version[-1] == "a":
            for (head, tail, relation) in edges:
                head = self.nodes_tokenized[int(head)][:-1].to(self.device)  # remove last [SEP] token
                rel = self.relation_tokenized[int(relation)].to(self.device)
                tail = self.nodes_tokenized[int(tail)][1:].to(self.device)  # remove first [CLS] token
                sentences.append(torch.cat([head, rel, tail]))
        else:  # version b
            for (head, tail) in edges:
                head = self.nodes_tokenized[int(head)].to(self.device)
                tail = self.nodes_tokenized[int(tail)][1:].to(self.device)  # remove first [CLS] token
                sentences.append(torch.cat([head, tail]))

        attention_mask = [torch.ones_like(input_id) for input_id in sentences]
        tokens['input_ids'] = pad_to_max(sentences).to(self.device)
        tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
        tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])

        # TODO: Modify the input to BERT and sentence definition above
        embs = self.get_roberta_embs(tokens)  # (batch_size, emb_size)

        if "kgbertsage" in self.version:
            aggregate_feats = self._get_neighbor_feats(edges, batch_size)

            all_feats = torch.cat([embs, aggregate_feats], dim=1)  # [batch_size, emb_size * 3]
            embs = F.relu(self.sage_layer(all_feats))

        # TODO: modify to TransE Loss
        logits = self.link_classifier(embs)  # (batch_size, emb_size)

        return logits

    def _get_neighbor_feats(self, edges, batch_size):
        sampled_neighbors = self._sample_neighbors(edges, num_sample=self.num_neighbor_samples)

        all_neighbors = list(dict.fromkeys(neigh for neigh in chain(*sampled_neighbors)))
        neighbor2id = {neighbor: idx for idx, neighbor in enumerate(all_neighbors)}

        neigh_tokens = {}
        neigh_input_ids = []
        if self.version[-1] == "a":
            for node_id, (neigh_id, rel_id) in all_neighbors:
                node = self.nodes_tokenized[int(node_id)][:-1].to(self.device)
                rel = self.relation_tokenized[int(rel_id)].to(self.device)
                neighbor_node = self.nodes_tokenized[int(neigh_id)][1:].to(self.device)
                neigh_input_ids.append(torch.cat([node, rel, neighbor_node]))
        else:
            for node_id, neigh_id in all_neighbors:
                node = self.nodes_tokenized[int(node_id)].to(self.device)
                neighbor_node = self.nodes_tokenized[int(neigh_id)][1:].to(self.device)
                neigh_input_ids.append(torch.cat([node, neighbor_node]))

        neigh_attention_mask = [torch.ones_like(input_id) for input_id in neigh_input_ids]
        neigh_tokens['input_ids'] = pad_to_max(neigh_input_ids).to(self.device)
        neigh_tokens['attention_mask'] = pad_to_max(neigh_attention_mask).to(self.device)
        neigh_tokens['token_type_ids'] = torch.zeros_like(neigh_tokens['input_ids'])

        neigh_embs = self.get_roberta_embs(neigh_tokens)  # [len(all_neighbors), embed size]
        aggregate_feats = self.aggregate(sampled_neighbors, neigh_embs, neighbor2id)  # [batch_size * 2, emb_size]
        aggregate_feats = aggregate_feats.reshape([batch_size, -1])  # [batch_size, emb_size * 2]

        return aggregate_feats

    def _sample_neighbors(self, edges, num_sample=10):
        """
        if neighbor include rel,
            samp_neighs: list of list of (node, (neighbor, rel)) tuples.
        else:
            samp_neighs: list of list of (node, neighbor) tuples.
        """
        neighbors_list = []
        for edge in edges:
            neighbors_list.append(self.adj_lists[int(edge[0])])  # head
            neighbors_list.append(self.adj_lists[int(edge[1])])  # tail
        if num_sample is not None:
            samp_neighs = [random.choices(neighbors, k=num_sample) if len(neighbors) > 0 else []
                           for neighbors in neighbors_list]
        else:
            samp_neighs = neighbors_list

        assert len(samp_neighs) == 2 * len(edges)
        sampled_neighbors = []
        for idx, neighbors in enumerate(samp_neighs):
            if idx % 2:  # odd idx means it is the neighbor of head
                sampled_neighbors.append([(edges[idx // 2][0].item(), n) for n in neighbors])  # head
            else:
                sampled_neighbors.append([(edges[idx // 2][1].item(), n) for n in neighbors])  # tail

        return sampled_neighbors

    def aggregate(self, neighbors_list, neigh_embs, neighbor2id, self_feats=None):
        """
        This is the same as graphsage.aggregate except for the arg names and
        the neighbors_list is list of list of (head, neighbor) tuple.
        """
        if self.agg_func == 'MEAN':
            agg_list = [torch.mean(neigh_embs[[int(neighbor2id[n]) for n in neighbors]], dim=0).unsqueeze(0) \
                            if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'MAX':
            agg_list = [
                torch.max(neigh_embs[[int(neighbor2id[n]) for n in neighbors]], dim=0).values.unsqueeze(0) \
                    if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'ATTENTION':
            # print("here!")
            # assert self_feats != None, "attentive aggregation requires self features"
            agg_list = []
            # self_feats: 128*768
            for neighbors, self_feat in zip(neighbors_list, self_feats):
                if len(neighbors) == 0:
                    agg_list.append(self.fill_tensor)
                    continue
                neighbor_emb = neigh_embs[[int(neighbor2id[n]) for n in neighbors]]  # n_neighbor, embeddings
                # neighbor_emb: 4*768
                # p rint(self_feat.shape)
                attn = torch.mm(torch.mm(neighbor_emb, self.bilinear), self_feat.unsqueeze(-1)).squeeze(-1)
                attn_s = torch.softmax(attn, 0)
                # print(attn_s.shape, neighbor_emb.shape)
                agg_res = torch.sum((neighbor_emb.T * attn_s), dim=-1)
                agg_list.append(agg_res.unsqueeze(0))

            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, neigh_embs.shape[1]).fill_(0).to(self.device)


class SimpleClassifier(nn.Module):
    def __init__(self, encoder, adj_lists, nodes_tokenized, nodes_text, device,
                 enc_style, num_class=2, include_rel=False, relation_tokenized=None):
        del adj_lists
        super(SimpleClassifier, self).__init__()
        self.nodes_tokenized = nodes_tokenized
        self.nodes_text = nodes_text  # to check the correctness
        self.device = device
        if encoder == "bert":
            # self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
            # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.enc_style = enc_style
        if include_rel:
            assert relation_tokenized is not None
        self.relation_tokenized = relation_tokenized

        if "pair" in enc_style:
            encode_dim = 768
        elif include_rel:
            encode_dim = 768 * 3
        else:
            encode_dim = 768 * 2
        self.link_classifier = Classification(encode_dim, num_class, device)

    def get_roberta_embs(self, tokens, padded_sequences=None):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        '''
        print("checking...")
        assert tokens['input_ids'].equal(torch.tensor(padded_sequences['input_ids']).to(self.device)), (tokens['input_ids'][0], torch.tensor(padded_sequences['input_ids']).to(self.device)[0])
        assert tokens['attention_mask'].equal(torch.tensor(padded_sequences['attention_mask']).to(self.device))
        assert tokens['token_type_ids'].equal(torch.tensor(padded_sequences['token_type_ids']).to(self.device)), (tokens['token_type_ids'], torch.tensor(padded_sequences['token_type_ids']).to(self.device))
        outputs_ = self.roberta_model(torch.tensor(padded_sequences['input_ids']).to(self.device),
                                     attention_mask=torch.tensor(padded_sequences['attention_mask']).to(self.device),
                                     token_type_ids=torch.tensor(padded_sequences['token_type_ids']).to(self.device)
                                     )
        '''
        outputs = self.roberta_model(tokens['input_ids'],
                                     attention_mask=tokens['attention_mask'],
                                     token_type_ids=tokens['token_type_ids'])

        # assert outputs_[0].equal(outputs[0])
        # assert outputs_[1].equal(outputs[1])

        if 'cls_raw' in self.enc_style:
            return outputs[0][:, 0, :]
        if 'cls_trans' in self.enc_style:  # with one more linear layer
            return outputs[1]
        if 'mean' in self.enc_style:
            mask = tokens['attention_mask']
            sum_mask = torch.sum(mask, dim=1).unsqueeze(-1).expand(mask.shape)
            norm_mask = mask / sum_mask
            norm_mask = norm_mask.unsqueeze(-1).expand(outputs[0].shape)
            return torch.sum(outputs[0] * norm_mask, dim=1)

    def forward(self, edges, b_s):
        '''
        embs = self.get_roberta_embs(
            pad_sequence([self.nodes_tokenized[int(node)] for node in all_nodes],
                         padding_value=0,
                         batch_first=1)[:, :MAX_SEQ_LENGTH].to(self.device)
        )
        '''
        if isinstance(edges, torch.Tensor):
            e_s = edges.shape[1]
            all_nodes = edges.reshape([-1])  # bs_, 2 -> bs_ * 2
        else:
            e_s = len(edges[0])

        tokens = {}
        if 'single' in self.enc_style:
            if e_s != 3:
                input_ids = [self.nodes_tokenized[int(node)].to(self.device) for node in all_nodes]
            else:
                input_ids = []  # (batch_size, max_length)
                for (head, tail, relation) in edges:
                    if isinstance(head, torch.Tensor) or isinstance(head, int):
                        head = self.nodes_tokenized[int(head)].to(self.device)
                    elif isinstance(head, str):
                        head = self.tokenizer(head).to(self.device) # remove last [SEP] token
                    else:
                        # others not supported
                        print(type(head))
                        assert False
                    if isinstance(tail, torch.Tensor) or isinstance(tail, int):
                        tail = self.nodes_tokenized[int(tail)].to(self.device)
                    elif isinstance(tail, str):
                        tail = self.tokenizer(tail).to(self.device) # remove last [SEP] token
                    else:
                        print(type(tail))
                        assert False
                    rel = self.relation_tokenized[int(relation)].to(self.device)
                    # rel is encoded without [CLS] or [SEP]. Add them here
                    rel = torch.cat([head[0:1], rel, head[-1:]])
                    input_ids.extend([head, tail, rel])
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
            tokens['input_ids'] = pad_to_max(input_ids).to(self.device)
            tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
            tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])
            # compare = self.tokenizer([self.nodes_text[int(node)] for node in all_nodes], padding=True)
        if 'pair' in self.enc_style:
            # compare = self.tokenizer([self.nodes_text[int(edge[0])] for edge in edges], [self.nodes_text[int(edge[1])] for edge in edges], padding=True)
            pair_encodings = [
                (self.nodes_tokenized[int(edge[0])].to(self.device), self.nodes_tokenized[int(edge[1])].to(self.device))
                for edge in edges]
            input_ids = [torch.cat((pair[0], pair[1][1:]), dim=0) for pair in pair_encodings]
            attention_mask = [torch.ones(pair[0].shape[0] + pair[1].shape[0] - 1) for pair in pair_encodings]
            token_type_ids = [
                torch.cat((torch.zeros_like(pair[0]).long(), torch.ones(pair[1].shape[0] - 1).long().to(self.device)),
                          dim=0) for pair in pair_encodings]
            tokens['input_ids'] = pad_to_max(input_ids).to(self.device)
            tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
            tokens['token_type_ids'] = pad_to_max(token_type_ids).to(self.device)

        embs = self.get_roberta_embs(
            tokens,
            # compare
        )
        logits = self.link_classifier(embs.reshape([b_s, -1]))  # (batch_size, 2*emb_size)

        return logits


class Classification(nn.Module):

    def __init__(self, emb_size, num_classes, device):
        super(Classification, self).__init__()

        # self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
        self.linear = nn.Linear(emb_size, num_classes).to(device)

    def forward(self, embs):
        logists = self.linear(embs)
        return logists


class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.linear = nn.Linear(self.input_size * 2, self.out_size)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes    -- list of nodes
        """
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        # [b_s, emb_size * 2]
        combined = F.relu(self.linear(combined))  # [b_s, emb_size]
        return combined


class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, encoder, num_layers, input_size, output_size,
                 adj_lists, nodes_tokenized, device, agg_func='MEAN', num_neighbor_samples=10,
                 enc_style='single_cls_raw', relation_tokenized=None,
                 with_rel=False):
        """
            with_rel: whether to include relation in BertSAGE
        """
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.agg_func = agg_func
        self.num_neighbor_samples = num_neighbor_samples
        self.enc_style = enc_style
        if encoder == "bert":
            self.roberta_model = BertModel.from_pretrained("bert-base-uncased").to(device)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif encoder == "roberta":
            self.roberta_model = RobertaModel.from_pretrained('roberta-base').to(device)
        print("using agg func {}".format(self.agg_func))
        self.adj_lists = adj_lists
        self.nodes_tokenized = nodes_tokenized
        self.relation_tokenized = relation_tokenized

        for index in range(1, num_layers + 1):
            layer_size = self.out_size if index != 1 else self.input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, self.out_size).to(device))
        # self.fill_tensor = torch.FloatTensor(1, 768).fill_(0).to(self.device)
        self.fill_tensor = torch.nn.Parameter(torch.rand(1, 768)).to(self.device)
        self.bilinear = torch.nn.Parameter(torch.rand(768, 768)).to(self.device)

    def get_roberta_embs(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)

            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.roberta_model(tokens['input_ids'],
                                     attention_mask=tokens['attention_mask'],
                                     token_type_ids=tokens['token_type_ids'])

        if 'cls_raw' in self.enc_style:
            return outputs[0][:, 0, :]
        if 'cls_trans' in self.enc_style:  # with one more linear layer
            return outputs[1]
        if 'mean' in self.enc_style:
            mask = tokens['attention_mask']
            sum_mask = torch.sum(mask, dim=1).unsqueeze(-1).expand(mask.shape)
            norm_mask = mask / sum_mask
            norm_mask = norm_mask.unsqueeze(-1).expand(outputs[0].shape)
            return torch.sum(outputs[0] * norm_mask, dim=1)

        # outputs = self.roberta_model(input_ids)
        # return torch.mean(outputs[0], dim=1) # aggregate embs

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch -- 1. torch.tensor of all head and tail nodes
                       2. list of [all nodes, all relations]
        """
        relations_batch = None  # initially set to none
        if isinstance(nodes_batch, list):
            nodes_batch, relations_batch = nodes_batch
        lower_layer_nodes = list(nodes_batch)  # node idx, may contains nodes not in node2id, which are str s.

        nodes_batch_layers = [(lower_layer_nodes,)]

        for i in range(self.num_layers):
            lower_layer_neighs, lower_layer_nodes = self._get_unique_neighs_list(lower_layer_nodes,
                                                                                 num_sample=self.num_neighbor_samples)
            # lower_layer_neighs: list(list())
            # lower_layer_nodes: list(nodes of next layer)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_layer_neighs))

        all_nodes = np.unique([int(n) for n in list(chain(*[layer[0] for layer in nodes_batch_layers]))])
        all_nodes_idx = dict([(node, idx) for idx, node in enumerate(all_nodes)])

        tokens = {}
        if 'single' in self.enc_style:
            input_ids = [self.nodes_tokenized[int(node)].to(self.device) if not isinstance(node, str) else self.tokenizer(node).to(self.device)  for node in all_nodes]
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
            tokens['input_ids'] = pad_to_max(input_ids).to(self.device)
            tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
            tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])
        # all_neigh_nodes = pad_sequence([self.nodes_tokenized[node ] for node in all_nodes], padding_value=1).transpose(0, 1)[:, :MAX_SEQ_LENGTH].to(self.device)
        else:
            assert False, "graphsage must take single encoding style"
        hidden_embs = self.get_roberta_embs(
            tokens
        )
        # return pre_hidden_embs[[all_nodes_idx[int(n)] for n in nodes_batch]]
        # (num_all_node, emb_size)
        '''
        for layer_idx in range(1, self.num_layers+1):
            this_layer_nodes = nodes_batch_layers[layer_idx][0] # all nodes in this layer
            # 0 indicates nodes, 1 indicates neighbors
            neigh_nodes, neighbors_list = nodes_batch_layers[layer_idx-1] # previous layer
            # list(), list(list())

            aggregate_feats = self.aggregate(neighbors_list, pre_hidden_embs, all_nodes_idx)
            # (this_layer_nodes_num, emb_size)

            sage_layer = getattr(self, 'sage_layer'+str(layer_idx))
            
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]], #pre_hidden_embs[layer_nodes],
                                        aggregate_feats=aggregate_feats)

            # cur_hidden_embs = torch.cat([pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]].unsqueeze(1), 
                                    # aggregate_feats.unsqueeze(1)], dim=1) # (b_s, 2, emb_size)
            # cur_hidden_embs = torch.mean(cur_hidden_embs, dim=1)

            pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]] = cur_hidden_embs

        # (input_batch_node_size, emb_size)
        # output the embeddings of the input nodes
        return pre_hidden_embs[[all_nodes_idx[int(n)] for n in nodes_batch]]
        '''
        pre_hidden_embs = [hidden_embs.clone() for layer_idx in range(self.num_layers + 1)]
        cur_hidden_embs = [None for layer_idx in range(self.num_layers)]
        # (num_all_node, emb_size)

        for layer_idx in range(1, self.num_layers + 1):
            this_layer_nodes = nodes_batch_layers[layer_idx][0]  # all nodes in this layer
            neigh_nodes, neighbors_list = nodes_batch_layers[layer_idx - 1]  # previous layer
            # list(), list(list())
            self_feats = pre_hidden_embs[layer_idx - 1][[all_nodes_idx[int(n)] for n in this_layer_nodes]]
            aggregate_feats = self.aggregate(neighbors_list, pre_hidden_embs[layer_idx - 1], all_nodes_idx, self_feats)
            # (this_layer_nodes_num, emb_size)

            sage_layer = getattr(self, 'sage_layer' + str(layer_idx))

            cur_hidden_embs[layer_idx - 1] = sage_layer(self_feats=self_feats,  # pre_hidden_embs[layer_nodes],
                                                        aggregate_feats=aggregate_feats)

            # cur_hidden_embs = torch.cat([pre_hidden_embs[[all_nodes_idx[int(n)] for n in this_layer_nodes]].unsqueeze(1), 
            # aggregate_feats.unsqueeze(1)], dim=1) # (b_s, 2, emb_size)
            # cur_hidden_embs = torch.mean(cur_hidden_embs, dim=1)
            # It's there where changed
            pre_hidden_embs[layer_idx][[all_nodes_idx[int(n)] for n in this_layer_nodes]] = cur_hidden_embs[
                layer_idx - 1]

        # (input_batch_node_size, emb_size)
        # output the embeddings of the input nodes
        saged_nodes = pre_hidden_embs[self.num_layers][[all_nodes_idx[int(n)] for n in nodes_batch]]
        # if edges included relations
        if relations_batch is not None:
            tokens = {}
            input_ids = [self.relation_tokenized[int(rel)].to(self.device) for rel in relations_batch]
            attention_mask = [torch.ones_like(input_id) for input_id in input_ids]
            tokens['input_ids'] = pad_to_max(input_ids).to(self.device)
            tokens['attention_mask'] = pad_to_max(attention_mask).to(self.device)
            tokens['token_type_ids'] = torch.zeros_like(tokens['input_ids'])
            relation_embs = self.get_roberta_embs(
                tokens
            )
            return [saged_nodes, relation_embs]

        return saged_nodes

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        try:
            # this block works if the adj_list contains relations also:
            # [(node1, rel1), (node2, rel2), ...] format.
            neighbors_list = [[neigh[0] for neigh in self.adj_lists[int(node)]] if not isinstance(node, str) else [] for node in nodes]
            # may contain nodes that are str (not in node2id)
        except TypeError:
            # this means that the adj_list only contains neighbor nodes:
            # [node1, node2, node3, ...] format.
            neighbors_list = [self.adj_lists[int(node)] if not isinstance(node, str) else [] for node in nodes]
        if not num_sample is None:
            samp_neighs = [np.random.choice(neighbors, num_sample) if len(neighbors) > 0 else [] for neighbors in
                           neighbors_list]
        else:
            samp_neighs = neighbors_list
        _unique_nodes_list = np.unique(list(chain(*samp_neighs)))
        return samp_neighs, _unique_nodes_list

    def aggregate(self, neighbors_list, pre_hidden_embs, all_nodes_idx, self_feats=None):
        if self.agg_func == 'MEAN':
            agg_list = [torch.mean(pre_hidden_embs[[int(all_nodes_idx[n]) for n in neighbors]], dim=0).unsqueeze(0) \
                            if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, pre_hidden_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'MAX':
            agg_list = [
                torch.max(pre_hidden_embs[[int(all_nodes_idx[n]) for n in neighbors]], dim=0).values.unsqueeze(0) \
                    if len(neighbors) > 0 else self.fill_tensor for neighbors in neighbors_list]
            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, pre_hidden_embs.shape[1]).fill_(0).to(self.device)

        if self.agg_func == 'ATTENTION':
            # print("here!")
            # assert self_feats != None, "attentive aggregation requires self features"
            agg_list = []
            # self_feats: 128*768
            for neighbors, self_feat in zip(neighbors_list, self_feats):
                if len(neighbors) == 0:
                    agg_list.append(self.fill_tensor)
                    continue
                neighbor_emb = pre_hidden_embs[[int(all_nodes_idx[n]) for n in neighbors]]  # n_neighbor, embeddings
                # neighbor_emb: 4*768
                # p rint(self_feat.shape)
                attn = torch.mm(torch.mm(neighbor_emb, self.bilinear), self_feat.unsqueeze(-1)).squeeze(-1)
                attn_s = torch.softmax(attn, 0)
                # print(attn_s.shape, neighbor_emb.shape)
                agg_res = torch.sum((neighbor_emb.T * attn_s), dim=-1)
                agg_list.append(agg_res.unsqueeze(0))

            if len(agg_list) > 0:
                return torch.cat(agg_list, dim=0)
            else:
                return torch.FloatTensor(0, pre_hidden_embs.shape[1]).fill_(0).to(self.device)
