import os
import json
import codecs
import pickle
import argparse

import numpy as np
import pandas as pd
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
from train_polyphonic import G2PTransformerMask, Mask_Softmax_Eval

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
    cedict = pickle.load(open('./filelists/digest_cedict.pkl', 'rb'))

    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    with codecs.open(args.merge_cedict, 'r', 'utf-8') as usernames:
        merge_cedict = json.load(usernames)

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

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
    with open(args.emo_tts_test_file, encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f):
            sents = line.strip()
            toks = tokenizer.tokenize(sents)
            # print('CHECK toks: ', toks)
            line2 = f.readline()
            if not line2:
                break
            pinyins = line2.strip().split(' ')
            pinyins = ['er5' if i == 'r5' else i for i in pinyins]
            # print('CHECK pinyins: ', pinyins)
            assert len(toks) == len(pinyins)

            

            input_ids = tokenizer.convert_tokens_to_ids(toks)
            input_ids = torch.tensor(input_ids, dtype=torch.long)

            label_idxs = []
            poly_idxs = []
            output_masks = []
            all_input_ids = []


            for idx, char in enumerate(sents):
                prons = merge_cedict[char]
                if len(prons) > 1:
                    poly_idxs.append(idx)
                    label_idxs.append(class2idx[pinyins[idx]])
                    all_input_ids.append(input_ids)
                    output_mask = []
                    for output_mask_item in prons:
                        output_mask.append(class2idx[output_mask_item])
                    output_masks.append(output_mask)

            # print('CHECK label_idxs: ', label_idxs)
            # print('CHECK poly_idxs: ', poly_idxs)
            # print('CHECK output_masks: ', output_masks)

            if label_idxs != []:
                all_input_ids = merge(all_input_ids).to(device)
                attention_mask = torch.sign(all_input_ids).to(device)
                poly_idxs = torch.tensor(poly_idxs, dtype=torch.long).to(device)
                label_idxs = torch.tensor(label_idxs, dtype=torch.long).to(device)
                output_masks = mask(all_input_ids, output_masks, num_classes).to(device)
                inputs = {"input_ids": all_input_ids,
                          "poly_ids": poly_idxs,
                          "attention_mask": attention_mask,
                          "output_mask": output_masks}
                with torch.no_grad():
                    logits = model(**inputs)

                logits = mask_criterion_eval(logits, output_masks)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                mask_preds = masked_augmax(logits, output_masks, dim=1).cpu().numpy()
                # if not (preds == mask_preds).all():
                #     print('CHECK preds:', preds)
                #     print('CHECK mask_preds:', mask_preds)
                #     print('CHECK label_idxs:', label_idxs)
                #     print('CHECK output_mask:', np.where(output_masks.cpu().numpy()==1.0))
                for i in range(len(label_idxs)):
                    if label_idxs[i] != preds[i]:
                        wrong_sen = tokenizer.convert_ids_to_tokens(all_input_ids[i])
                        if wrong_sen[poly_idxs[i]] not in cedict:
                            flag = 'NC'
                        elif idx2class[int(label_idxs[i])] not in cedict[wrong_sen[poly_idxs[i]]]:
                            flag = 'NP'
                            if len(cedict[wrong_sen[poly_idxs[i]]]) == 1:
                                flag = 'NPO'
                        else:
                            flag = 'P'

                        # file_out.append(
                        #     ''.join(wrong_sen) + '|' +str(int(poly_idxs[i])) + '|' + wrong_sen[poly_idxs[i]] + '|' + idx2class[int(label_idxs[i])] + '|' +
                        #     idx2class[int(preds[i])] + '|' + flag)
                        file_out.append(
                            [''.join(wrong_sen), str(int(poly_idxs[i])), wrong_sen[poly_idxs[i]], idx2class[int(label_idxs[i])], idx2class[int(preds[i])], flag])
                        # print(file_out[-1])



                all_preds.append(preds)
                all_mask_preds.append(mask_preds)
                all_labels.append(label_idxs.cpu().numpy())
                preds = np.concatenate(all_preds, axis=0)
                mask_preds = np.concatenate(all_mask_preds, axis=0)
                labels = np.concatenate(all_labels, axis=0)
            else:
                continue

        test_acc = accuracy_score(labels, preds)
        mask_test_acc = accuracy_score(labels, mask_preds)
        pred_diff_acc = accuracy_score(preds, mask_preds)
        print("Final acc: {:.2f}, mask acc: {:.2f}, pred_diff_acc: {:.2f}".format(test_acc*100, mask_test_acc*100, pred_diff_acc*100))

        # with open('./filelists/wrongcase_mask_emoTTS.txt', 'w', encoding='utf8') as fout:
        #     fout.write('\n'.join(file_out))

        column = ['句子内容', '字的位置', '字', '标注读音', '预测读音', '是否在g2pM集']
        test = pd.DataFrame(columns=column, data=file_out)
        test.to_csv('./filelists/wrongcase_mask_emoTTS.csv', encoding='utf_8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    hparams = create_hparams()
    hparams.populate_arguments(parser)
    parser.add_argument("--train_file", type=str, default="./filelists/train_polyphonic.sent")
    parser.add_argument("--train_label", type=str, default="./filelists/train_polyphonic.lb")
    #parser.add_argument("--class2idx", type=str, default="./filelists/uni_class2idx.json")
    # parser.add_argument("--class2idx", type=str, default="./filelists/class2idx.pkl")
    #parser.add_argument("--merge_cedict", type=str, default="./filelists/universal_cedict.json")
    # parser.add_argument("--merge_cedict", type=str, default="./filelists/digest_cedict.pkl")

    parser.add_argument("--saved_model_path", type=str, default="./save/mask/97.76_model.pt")

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