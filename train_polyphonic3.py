import argparse
import os
import json
import codecs
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from transformers import BertModel, BertTokenizer
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel


from model import Poly_Phoneme_Classifier
from hparams import create_hparams
import parse_nk

SPLIT_TOKEN = "▁"


os.environ['CUDA_LAUNCH_BLOCKING'] = "0"


def masked_augmax(logits, mask, dim, min_val=-1e7):
    logits = logits.exp().mul(mask)
    # one_minus_mask = (1.0 - mask).byte()
    # replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    # max_value, _ = replaced_vector.max(dim=dim)
    max_value = torch.argmax(logits, dim=1)
    return max_value


class Mask_Softmax_Eval(nn.Module):
    def __init__(self, plus=1.0):
        super(Mask_Softmax_Eval, self).__init__()
        self.plus = plus
    def forward(self, logits, output_mask):
        # logits = logits + (output_mask + 1e-45).log()
        return torch.nn.functional.log_softmax(logits)


        # logits_exp = logits.exp()
        # logits_exp = logits_exp.mul(output_mask)
        # partition = logits_exp.sum(dim=1, keepdim=True) + self.plus
        # logits_softmax = logits_exp / partition
        # # print('CHECK logits_softmax: ', logits_softmax[0])
        # # output_mask_false = 1.0 - output_mask
        # # multi_logits = output_mask_false + logits_softmax
        # multi_logits = logits_softmax + 5e-5
        # # multi_partition = multi_logits.sum(dim=-1, keepdim=True) + self.plus
        # logits_logsoftmax = multi_logits.log()
        # # print('CHECK logits_softmax:', logits_softmax)
        # # logits_logsoftmax = logits_softmax.log()
        # # print('CHECK logits_logsoftmax:', logits_logsoftmax)
        # # print('CHECK logits:', logits[0])
        # # logits = logits.mul(output_mask)
        # # print('CHECK logits:', logits[0])
        # return logits_softmax

class Mask_Softmax(nn.Module):
    def __init__(self, plus=1.0):
        super(Mask_Softmax, self).__init__()
        self.plus = plus
    def forward(self, logits, output_mask):
        # logits = logits + (output_mask + 1e-45).log()
        return torch.nn.functional.log_softmax(logits)

        # logits_exp = logits.exp()
        # logits_exp = logits_exp.mul(output_mask)
        # partition = logits_exp.sum(dim=1, keepdim=True) + self.plus
        # logits_softmax = logits_exp / partition
        # print('CHECK logits_softmax:', logits_softmax[0, :])
        # # output_mask_false = 1.0 - output_mask
        # # multi_logits = output_mask_false + logits_softmax
        # multi_logits = logits_softmax + 5e-5
        # # multi_partition = multi_logits.sum(dim=-1, keepdim=True) + self.plus
        # logits_logsoftmax = multi_logits.log()
        # # logits_logsoftmax = logits_softmax.log()
        # return logits_logsoftmax


class G2PBert(nn.Module):
    def __init__(self, num_classes):
        super(G2PBert, self).__init__()
        # self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.bert = BertModel.from_pretrained('./bert/bert-base-chinese')
        self.classifier = nn.Linear(768, num_classes)


    def forward(self, input_ids, attention_mask, poly_ids):
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask}
        outputs = self.bert(**inputs)
        hidden = outputs[0][0]
        batch_size = input_ids.size(0)
        # print('CHECK hidden: ', len(hidden))
        # print('CHECK hidden[0]: ', hidden[0].size())

        # print('CHECK torch.arange(batch_size): ', torch.arange(batch_size))
        # print('CHECK poly_ids: ', poly_ids)
        poly_hidden = hidden[torch.arange(batch_size), poly_ids]
        logits = self.classifier(poly_hidden)

        return logits


class G2PTransformer(nn.Module):
    def __init__(self, num_classes):
        super(G2PTransformer, self).__init__()
        self.bert = BertModel.from_pretrained('./bert/bert-base-chinese')
        self.poly_phoneme_classifier = Poly_Phoneme_Classifier(hparams)

        self.linear = nn.Linear(1024, num_classes)

        self.bert_embedding_features_dim = 768
        self.transformer_embedding_features_dim = 1324
        self.embedding_features_dim = 1024
        self.select_model_hidden_dim = 512

        self.linear_pre = nn.Sequential(
            nn.Linear(self.bert_embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.select_model_hidden_dim, self.transformer_embedding_features_dim),
            )

        self.linear_aft = nn.Sequential(
            nn.Linear(self.embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.select_model_hidden_dim, num_classes),
            )

    def forward(self, input_ids, attention_mask, poly_ids):
        batch_size = input_ids.size(0)
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask}
        outputs = self.bert(**inputs)
        hidden = outputs[0][0]
        hidden = self.linear_pre(hidden)
        transformer_output = self.poly_phoneme_classifier.forward_train_polyphonic(hidden)
        poly_hidden = transformer_output[torch.arange(batch_size), poly_ids]
        logits = self.linear_aft(poly_hidden)
        return logits

class G2PTransformerMask(nn.Module):
    def __init__(self, num_classes):
        super(G2PTransformerMask, self).__init__()
        self.g2ptransformer = G2PTransformer(num_classes)
        self.mask_criterion = Mask_Softmax()

    def forward(self, input_ids, attention_mask, poly_ids, output_mask):
        logits = self.g2ptransformer(input_ids, attention_mask, poly_ids)

        # output_mask = torch.FloatTensor(1, self.g2pdataset.num_classes)
        # # mask_padded.zero_()
        # output_mask.fill_(-float('inf')) 
        # for k in range(self.g2pdataset.num_classes):
            # # index = torch.LongTensor([i, j, mask_character[k]])
            # index = torch.LongTensor([[i, j, mask_character[k]]])
            # # value = torch.ones(index.shape[0])
            # value = torch.zeros(index.shape[0])
            # # print('CHECK MASK index:', index)
            # # mask_padded.index_put_((index,), torch.Tensor([1,1]))
            # mask_padded.index_put_(tuple(index.t()), value)
        # loss = criterion(logits, labels)

        # print('CHECK logits SHAPE:', logits.size())
        # print('CHECK output_mask SHAPE:', output_mask.size())
        # logits = logits + output_mask
        # logits = logits.mul(output_mask)
        
        # logits_with_mask = self.mask_criterion(logits, output_mask)
        # return logits_with_mask
        return logits

class G2PDataset(Dataset):
    def __init__(self, sent_file, label_file, class2idx_file, max_length=512):
        super(G2PDataset, self).__init__()
        self.max_length = max_length
        self.sents = open(sent_file).readlines()
        self.labels = open(label_file).readlines()

        assert len(self.sents) == len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        # self.tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese')
        # self.tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese', do_lower_case=bert_do_lower_case)

        with open(class2idx_file, "rb") as f:
            self.class2idx = pickle.load(f)
        self.num_classes = len(self.class2idx)
        self.total_size = len(self.labels)

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"
        sent = self.sents[index].strip()
        label = self.labels[index].strip()

        sent = sent.replace(SPLIT_TOKEN, cls_tok)
        toks = self.tokenizer.tokenize(sent)

        poly_idx = toks.index(cls_tok) + 1

        toks = list(filter(lambda x: x != cls_tok, toks))
        toks.insert(0, cls_tok)
        toks.append(sep_tok)

        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_id = self.class2idx[label]

        return input_ids, poly_idx, label_id


class G2PDatasetMask(Dataset):
    def __init__(self, sent_file, label_file, class2idx_file, merge_cedict_file, max_length=512):
        super(G2PDatasetMask, self).__init__()
        self.g2pdataset = G2PDataset(sent_file, label_file, class2idx_file, max_length)
        with codecs.open(merge_cedict_file, 'r', 'utf-8') as usernames:
            self.merge_cedict = json.load(usernames)
        # with open(merge_cedict_file, "rb") as f:
        #     self.merge_cedict = pickle.load(f)
        self.merge_cedict['[UNK]'] = []
        # print('CHECK self.merge_cedict:', self.merge_cedict)

    def __len__(self):
        return self.g2pdataset.total_size

    def __getitem__(self, index):
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"
        sent = self.g2pdataset.sents[index].strip()
        label = self.g2pdataset.labels[index].strip()

        # print('CHECK sent', sent)

        sent = sent.replace(SPLIT_TOKEN, cls_tok)
        toks = self.g2pdataset.tokenizer.tokenize(sent)

        poly_idx = toks.index(cls_tok) + 1
        poly_character = toks[poly_idx]

        toks = list(filter(lambda x: x != cls_tok, toks))
        toks.insert(0, cls_tok)
        toks.append(sep_tok)

        input_ids = self.g2pdataset.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_id = self.g2pdataset.class2idx[label]

        # print('CHECK poly_character', poly_character)

        output_mask = []
        output_mask_toks = self.merge_cedict[poly_character]
        for output_mask_item in output_mask_toks:
            output_mask.append(self.g2pdataset.class2idx[output_mask_item])

        return input_ids, poly_idx, label_id, output_mask, self.g2pdataset.num_classes


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs
    all_input_ids, poly_ids, label_ids = zip(*data)

    all_input_ids = merge(all_input_ids)
    poly_ids = torch.tensor(poly_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)

    return all_input_ids, poly_ids, label_ids


def collate_fn_mask(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs
    def mask(sequences, output_mask, num_classes):
        lengths = [len(seq) for seq in sequences]
        # mask_output = torch.zeros((len(sequences), max(lengths), num_classes)).long()
        # print('CHECK num_classes:', num_classes)
        # print('CHECK num_classes:', num_classes.type())
        mask_output = torch.FloatTensor(len(sequences), num_classes[0])
        # mask_output.fill_(-float('inf')) 
        mask_output.fill_(0.0) 
        for i in range(len(output_mask)):
            mask_sequence = output_mask[i]
            # print('CHECK mask_sequence:', mask_sequence)
            for j in range(len(mask_sequence)):
                mask_character = mask_sequence[j]
                index = torch.LongTensor([[i, mask_character]])
                value = torch.ones(index.shape[0])
                mask_output.index_put_(tuple(index.t()), value)
        return mask_output
    all_input_ids, poly_ids, label_ids, output_mask, num_classes = zip(*data)

    all_input_ids = merge(all_input_ids)
    poly_ids = torch.tensor(poly_ids, dtype=torch.long)
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    output_mask = mask(all_input_ids, output_mask, num_classes)

    return all_input_ids, poly_ids, label_ids, output_mask


def get_dataloader(use_output_mask, sent_file, label_file, class2idx, merge_cedict_file,
                   batch_size, max_length, shuffle=False):

    if use_output_mask:
        dataset = G2PDatasetMask(sent_file, label_file, class2idx, merge_cedict_file, max_length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn_mask,
                                num_workers=4)
    else:
        dataset = G2PDataset(sent_file, label_file, class2idx, max_length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn,
                                num_workers=4)

    return dataloader


def trunc_length(input_ids):
    length = torch.sum(torch.sign(input_ids), 1)
    max_length = torch.max(length)

    input_ids = input_ids[:, :max_length]

    return input_ids


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    train_dataloader = get_dataloader(args.use_output_mask, args.train_file, args.train_label,
                                      args.class2idx, args.merge_cedict, args.batch_size,
                                      args.max_length, shuffle=True)

    val_dataloader = get_dataloader(args.use_output_mask, args.val_file, args.val_label,
                                    args.class2idx, args.merge_cedict, args.batch_size,
                                    args.max_length, shuffle=True)

    test_dataloader = get_dataloader(args.use_output_mask, args.test_file, args.test_label,
                                     args.class2idx, args.merge_cedict, args.batch_size,
                                     args.max_length, shuffle=True)

    with open(args.class2idx, "rb") as f:
        class2idx = pickle.load(f)
    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    # model = G2PBert(num_classes)
    # model = G2PTransformer(num_classes)
    model = G2PTransformerMask(num_classes)
    device = torch.cuda.current_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    mask_criterion = Mask_Softmax()
    mask_criterion_eval = Mask_Softmax_Eval()
    model_dir = "./save/bert_transformer"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_acc = 0
    for epoch in range(args.train_polyphonic_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader, start=1):
            # print('CEHCK batch:', batch)
            # if idx > 200:
            #     break
            batch = tuple(t.to(device) for t in batch)
            if args.use_output_mask:
                input_ids, poly_ids, labels, output_mask = batch
                mask = torch.sign(input_ids)
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask,
                          "output_mask": output_mask}
            else:
                input_ids, poly_ids, labels = batch
                mask = torch.sign(input_ids)
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask}

            # inputs = {"input_ids": input_ids,
                      # "poly_ids": poly_ids,
                      # "attention_mask": mask}
            logits = model(**inputs)
            logits = mask_criterion(logits, output_mask)
            loss = criterion(logits, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

            if idx % 100 == 0:
                print("loss : {:.4f}".format(loss.item()))
        all_preds = []
        all_mask_preds = []
        all_labels = []
        model.eval()
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            # input_ids, poly_ids,  labels = batch
            # mask = torch.sign(input_ids)

            # inputs = {"input_ids": input_ids,
                      # "poly_ids": poly_ids,
                      # "attention_mask": mask}
            if args.use_output_mask:
                input_ids, poly_ids, labels, output_mask = batch
                mask = torch.sign(input_ids)
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask,
                          "output_mask": output_mask}
            else:
                input_ids, poly_ids, labels = batch
                mask = torch.sign(input_ids)
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask}
            with torch.no_grad():
                logits = model(**inputs)

            # logits = logits.exp()
            # output_mask_false = 1.0 - output_mask
            # logits = logits - output_mask_false
            logits = mask_criterion_eval(logits, output_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            mask_preds = masked_augmax(logits, output_mask, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_mask_preds.append(mask_preds)
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        mask_preds = np.concatenate(all_mask_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        # print('CHECK preds:', preds)
        # print('CHECK mask_preds:', mask_preds)
        # print('CHECK labels:', labels)

        val_acc = accuracy_score(labels, preds)
        mask_val_acc = accuracy_score(labels, mask_preds)
        pred_diff_acc = accuracy_score(preds, mask_preds)
        print("epoch :{}, acc: {:.2f}, mask acc: {:.2f}, pred_diff_acc: {:.2f}".format(epoch, val_acc*100, mask_val_acc*100, pred_diff_acc*100))
        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = model.state_dict()
            save_file = os.path.join(
                model_dir, "{:.2f}_model.pt".format(val_acc*100))
            torch.save(state_dict, save_file)

    model.eval()
    all_preds = []
    all_mask_preds = []
    all_labels = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        # input_ids, poly_ids, labels = batch
        # mask = torch.sign(input_ids)
        # inputs = {"input_ids": input_ids,
                  # "poly_ids": poly_ids,
                  # "attention_mask": mask}
        if args.use_output_mask:
            input_ids, poly_ids, labels, output_mask = batch
            mask = torch.sign(input_ids)
            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask,
                      "output_mask": output_mask}
        else:
            input_ids, poly_ids, labels = batch
            mask = torch.sign(input_ids)
            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask}

        with torch.no_grad():
            logits = model(**inputs)
        # logits = logits.exp()
        # output_mask_false = 1.0 - output_mask
        # logits = logits - output_mask_false
        logits = mask_criterion_eval(logits, output_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        mask_preds = masked_augmax(logits, output_mask, dim=1).cpu().numpy()

        all_preds.append(preds)
        all_mask_preds.append(mask_preds)
        all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        mask_preds = np.concatenate(all_mask_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

    test_acc = accuracy_score(labels, preds)
    mask_test_acc = accuracy_score(labels, mask_preds)
    pred_diff_acc = accuracy_score(preds, mask_preds)
    print("Final acc: {:.2f}, mask acc: {:.2f}, pred_diff_acc: {:.2f}".format(test_acc*100, mask_test_acc*100, pred_diff_acc*100))
    # print("Final acc: {:.2f}".format(test_acc*100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hparams = create_hparams()
    hparams.populate_arguments(parser)
    parser.add_argument("--train_file", type=str, default="./filelists/train_polyphonic.sent")
    parser.add_argument("--train_label", type=str, default="./filelists/train_polyphonic.lb")
    parser.add_argument("--class2idx", type=str, default="./filelists/class2idx.pkl")
    parser.add_argument("--merge_cedict", type=str, default="./filelists/universal_cedict.json")
    # parser.add_argument("--merge_cedict", type=str, default="./filelists/digest_cedict.pkl")

    parser.add_argument("--val_file", type=str, default="./filelists/dev_polyphonic.sent")
    parser.add_argument("--val_label", type=str, default="./filelists/dev_polyphonic.lb")

    parser.add_argument("--test_file", type=str, default="./filelists/test_polyphonic.sent")
    parser.add_argument("--test_label", type=str, default="./filelists/test_polyphonic.lb")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_polyphonic_epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_output_mask", type=bool, default=True)
    args = parser.parse_args()

    main(args)