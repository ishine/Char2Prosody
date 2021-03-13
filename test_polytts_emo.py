import os
import time
import math
import json
import codecs
import argparse
import numpy as np
from tqdm import tqdm
from numpy import finfo

from sklearn.metrics import accuracy_score

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from distributed import apply_gradient_allreduce

import parse_nk
from models import G2PTransformerMask, poly_tonesandhi
from data_utils import TextMelLoader, TextMelCollate, G2PDatasetMask, get_dataloader, polyTTS_get_dataloader
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from train_tts import Gumbel_Softmax

class pack_polytone_sanhdi(nn.Module):
    def __init__(self, hparams):
        super(Cascaded_Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.structure_feature_dim = hparams.structure_feature_dim
        self.num_classes = hparams.num_classes
        self.tts_use_structure = hparams.tts_use_structure and hparams.poly_use_structure

        self.pinyin_to_yinsu_dict = to_gpu(torch.from_numpy(np.array(pinyin_to_yinsu(hparams.class2idx)))).float()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.poly_phoneme_classifier = poly_tonesandhi(hparams.num_classes, hparams)

        if hparams.poly_use_structure:
            self.poly_phoneme_classifier.load_state_dict(torch.load(hparams.saved_model_path_sandhi_structure))
        else:
            self.poly_phoneme_classifier.load_state_dict(torch.load(hparams.saved_model_path_sandhi))

        self.mask_criterion = Gumbel_Softmax()
        self.tacotron2 = Tacotron2(hparams)


def test():
    hparams = create_hparams()
    with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
        class2idx = json.load(usernames)
    print("num classes: {}".format(len(class2idx)))

    num_classes = len(class2idx)
    # model = G2PTransformerMask(num_classes, hparams)
    # model.load_state_dict(torch.load('./save/poly_only/97.98_model.pt'))
    # model = G2PTransformerMask(num_classes, hparams)
    # model.load_state_dict(torch.load('./save/poly_only_syntax_frozen/97.57_model.pt'))
    model = poly_tonesandhi(num_classes, hparams)
    model.load_state_dict(torch.load('./save/poly_tts_CNN_syntax_frozen/95.49_model.pt'))
    # model.load_state_dict(torch.load('./save/poly_tts_CNN/96.84_model.pt'))

    device = torch.cuda.current_device()
    model = model.to(device)
    model.eval()

    mask_criterion = Gumbel_Softmax()

    all_preds = []
    all_labels = []
    model.eval()

    test_dataloader = polyTTS_get_dataloader(hparams.use_output_mask, './filelists/bznsyp_character_audio_text_train_filelist.txt',
                                             hparams, hparams.poly_batch_size,
                                             hparams.poly_max_length, shuffle=True)
    # test_dataloader = polyTTS_get_dataloader(hparams.use_output_mask, './filelists/bznsyp_character_audio_text_test_filelist.txt',
    #                                          hparams, hparams.poly_batch_size,
    #                                          hparams.poly_max_length, shuffle=True)
    # test_dataloader = polyTTS_get_dataloader(hparams.use_output_mask,
    #                                          './filelists/emo_sen2phone_test.txt',
    #                                          hparams, hparams.poly_batch_size,
    #                                          hparams.poly_max_length, shuffle=True)
    # test_dataloader = get_dataloader(hparams.use_output_mask, hparams.val_file, hparams.val_label,
    #                                   hparams, hparams.poly_batch_size,
    #                                   hparams.poly_max_length, shuffle=True)

    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        if hparams.use_output_mask:
            input_ids, poly_ids, labels, output_mask = batch
            mask = torch.sign(input_ids)
            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask}

        else:
            input_ids, poly_ids, labels = batch
            mask = torch.sign(input_ids)
            inputs = {"input_ids": input_ids,
                      "poly_ids": poly_ids,
                      "attention_mask": mask}
        # print('CHECK poly_ids:', poly_ids)
        with torch.no_grad():
            logits, _ = model(**inputs)
        # print('Check logits:',logits)]
        # batch_size = logits.size(0)
        # logits = logits[torch.arange(batch_size), poly_ids]
        labels, logits, output_mask = model.select_poly(labels, logits, output_mask, poly_ids)
        logits = mask_criterion(logits, output_mask, True)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        # print("CHECK pred", preds)
        # print("CHECK label", labels)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    test_acc = accuracy_score(labels, preds)
    print("test for emotion TTS : acc: {:.2f}".format(test_acc * 100))


if __name__ == '__main__':
    test()
