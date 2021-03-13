import os
import json
import codecs
import pickle
import argparse

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

import parse_nk
from model import Poly_Phoneme_Classifier
from hparams import create_hparams
from train_polyphonic import G2PTransformerMask
from train_polyphonic4 import Mask_Softmax_Eval, get_dataloader

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
BOS_TOKEN = "시"
EOS_TOKEN = "끝"
SPLIT_TOKEN = "▁"
cls_tok = "[CLS]"
sep_tok = "[SEP]"

os.environ['CUDA_LAUNCH_BLOCKING'] = "2"


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
    mask_output = torch.FloatTensor(len(sequences), num_classes)
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

def masked_augmax(logits, mask, dim, min_val=-1e7):
    logits = logits.exp()
    logits = logits.mul(mask)
    # one_minus_mask = (1.0 - mask).byte()
    # replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    # max_value, _ = replaced_vector.max(dim=dim)
    max_value = torch.argmax(logits, dim=1)
    return max_value


def main(args):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    with codecs.open(args.class2idx, 'r', 'utf-8') as usernames:
        class2idx = json.load(usernames)
    idx2class = {v: k for k, v in class2idx.items()}
    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    with codecs.open(args.merge_cedict, 'r', 'utf-8') as usernames:
        merge_cedict = json.load(usernames)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    test_dataloader = get_dataloader(args.use_output_mask, args.test_file, args.test_label,
                                     args.class2idx, args.merge_cedict, args.batch_size,
                                     args.max_length, shuffle=False)
    model = G2PTransformerMask(num_classes, hparams)
    model.load_state_dict(torch.load(args.saved_model_path))
    device = torch.cuda.current_device()
    model = model.to(device)
    model.eval()
    mask_criterion_eval = Mask_Softmax_Eval()
    all_preds = []
    all_mask_preds = []
    all_labels = []
    file_out = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        if batch == None:
            continue
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
        with torch.no_grad():
            logits = model(**inputs)
        # logits = logits.exp()
        # output_mask_false = 1.0 - output_mask
        # logits = logits - output_mask_false
        logits = mask_criterion_eval(logits, output_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        for i in range(len(preds)):
            if labels[i] != preds[i]:
                wrong_sen = tokenizer.convert_ids_to_tokens(input_ids[i])
                wrong_sen = wrong_sen[:wrong_sen.index('[SEP]')]
                file_out.append(
                    ''.join(wrong_sen) + '|' + wrong_sen[poly_ids[i]] + '|' + idx2class[int(labels[i])] + '|' +
                    idx2class[int(preds[i])])
                print(file_out[-1])

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
    print("Final acc: {:.2f}, mask acc: {:.2f}, pred_diff_acc: {:.2f}".format(test_acc * 100, mask_test_acc * 100,
                                                                              pred_diff_acc * 100))
    # print("Final acc: {:.2f}".format(test_acc*100))
    with open('./filelists/wrongcase_nomask_g2pM.txt', 'w', encoding='utf8') as fout:
        fout.write('\n'.join(file_out))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hparams = create_hparams()
    hparams.populate_arguments(parser)
    parser.add_argument("--train_file", type=str, default="./filelists/train_polyphonic.sent")
    parser.add_argument("--train_label", type=str, default="./filelists/train_polyphonic.lb")
    # parser.add_argument("--class2idx", type=str, default="./filelists/uni_class2idx.json")
    # parser.add_argument("--class2idx", type=str, default="./filelists/class2idx.pkl")
    # parser.add_argument("--merge_cedict", type=str, default="./filelists/universal_cedict.json")
    # parser.add_argument("--merge_cedict", type=str, default="./filelists/digest_cedict.pkl")

    parser.add_argument("--saved_model_path", type=str, default="./save/no_mask/97.77_model.pt")

    parser.add_argument("--emo_tts_test_file", type=str, default="./filelists/emo_sen2phone.txt")
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