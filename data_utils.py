import json
import json
import random
import codecs
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, poly_yinsu_to_sequence, poly_yinsu_to_mask

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel


SPLIT_TOKEN = "▁"


def _is_erhua(pinyin):
    """
    Decide whether pinyin (without tone number) is retroflex (Erhua)
    """
    if len(pinyin) <= 1 or pinyin[:-1] == 'er':
        return False
    elif pinyin[-2] == 'r':
        return True
    else:
        return False

def text_and_pinyin_norm(texts, pinyins):
    pinyins = pinyins.split(' ')
    assert len(texts) == len(pinyins)
    texts_norm = []
    pinyins_nrom = []
    for (text, pinyin) in zip(texts, pinyins):
        # print('CEHCK (text, pinyin):', text, pinyin)
        # print('CEHCK _is_erhua(pinyin):', _is_erhua(pinyin))
        if text != '儿' and _is_erhua(pinyin):  # erhuayin
            # print('CEHCK HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            texts_norm.append(text)
            texts_norm.append('儿')
            pinyin_norm  = pinyin[:-2] + pinyin[-1]
            pinyins_nrom.append(pinyin_norm)
            pinyins_nrom.append('er5')
        elif text == '儿' and pinyin[:-1] == 'rr':
            # print('CEHCK HERE TOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            texts_norm.append(text)
            pinyins_nrom.append('er5')
        else:
            texts_norm.append(text)
            pinyins_nrom.append(pinyin)
    assert len(texts_norm) == len(pinyins_nrom)
    return ''.join(texts_norm), ' '.join(pinyins_nrom)


class G2PDatasetMask(torch.utils.data.Dataset):
    def __init__(self, sent_file, label_file, hparams, max_length=512):
        super(G2PDatasetMask, self).__init__()

        self.max_length = max_length
        self.sents = open(sent_file).readlines()
        self.labels = open(label_file).readlines()

        assert len(self.sents) == len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)

        self.num_classes = len(self.class2idx)
        self.total_size = len(self.labels)

        with codecs.open(hparams.merge_cedict, 'r', 'utf-8') as usernames:
            self.merge_cedict = json.load(usernames)
        self.merge_cedict['[UNK]'] = []

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"
        sent = self.sents[index].strip()
        label = self.labels[index].strip()

        # print('CHECK sent', sent)

        sent = sent.replace(SPLIT_TOKEN, cls_tok)
        toks = self.tokenizer.tokenize(sent)

        poly_idx = toks.index(cls_tok) + 1
        poly_character = toks[poly_idx]

        toks = list(filter(lambda x: x != cls_tok, toks))
        toks.insert(0, cls_tok)
        toks.append(sep_tok)

        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label = label.replace('lu:', 'lv')
        label = label.replace('nu:', 'nv')
        if label == 'r5':
            label = 'er5'
        label_id = self.class2idx[label]


        output_mask = []
        output_mask_toks = self.merge_cedict[poly_character]

        if len(output_mask_toks) >= 1:
            for output_mask_item in output_mask_toks:
                output_mask.append(self.class2idx[output_mask_item])
            return input_ids, poly_idx, label_id, output_mask, self.num_classes
        # else:
        #     print('CHECK output_mask_toks 0:', sent)


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

    data = filter (lambda x:x is not None, data)
    # data = filter (lambda x:len(x) == 5, data)
    # print('CHECK zip(*data):', zip(*data))
    # print('CHECK input:', zip(*data))
    data = [*data]
    # print('CHECK data length:', len(data))
    data_to_check_length = len(data)

    if data_to_check_length != 0:
        all_input_ids, poly_ids, label_ids, output_mask, num_classes = zip(*data)


        all_input_ids = merge(all_input_ids)
        poly_ids = torch.tensor(poly_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        output_mask = mask(all_input_ids, output_mask, num_classes)
        return all_input_ids, poly_ids, label_ids, output_mask


def get_dataloader(use_output_mask, sent_file, label_file, hparams,
                   batch_size, max_length, shuffle=False):

    if use_output_mask:
        dataset = G2PDatasetMask(sent_file, label_file, hparams, max_length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=collate_fn_mask,
                                num_workers=4)
    return dataloader


class polyTTS_G2PDatasetMask(torch.utils.data.Dataset):
    def __init__(self, audiopaths_and_text, hparams, max_length=512):
        super(polyTTS_G2PDatasetMask, self).__init__()

        self.max_length = max_length
        self.sents_and_lables = load_filepaths_and_text(audiopaths_and_text)
        # self.sents = open(sent_file).readlines()
        # self.labels = open(label_file).readlines()
        # assert len(self.sents) == len(self.labels)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)

        self.num_classes = len(self.class2idx)
        self.total_size = len(self.sents_and_lables)

        with codecs.open(hparams.merge_cedict, 'r', 'utf-8') as usernames:
            self.merge_cedict = json.load(usernames)
        self.merge_cedict['[UNK]'] = []

    def __len__(self):
        return self.total_size

    def __getitem__(self, index):
        cls_tok = "[CLS]"
        sep_tok = "[SEP]"
        sent = self.sents_and_lables[index][1]
        label = self.sents_and_lables[index][2]

        sent, label = text_and_pinyin_norm(sent, label)
        # print('CHECK sent IN polyTTS_G2PDatasetMask:', sent)
        # # sent = sent.replace(SPLIT_TOKEN, cls_tok)
        # toks = self.tokenizer.tokenize(sent)
        # pinyins = label.strip().split(' ')
        # assert len(toks) == len(pinyins)

        # poly_idx = toks.index(cls_tok) + 1
        # poly_character = toks[poly_idx]

        # input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        # input_ids = torch.tensor(input_ids, dtype=torch.long)
        # label = label.replace('lu:', 'lv')
        # label = label.replace('nu:', 'nv')
        # if label == 'er5':
            # label = 'er2'
        # label_id = self.class2idx[label]


        # output_mask = []
        # output_mask_toks = self.merge_cedict[poly_character]

        # if len(output_mask_toks) >=1:
            # for output_mask_item in output_mask_toks:
                # output_mask.append(self.class2idx[output_mask_item])
            # return input_ids, poly_idx, label_id, output_mask, self.num_classes

        toks = self.tokenizer.tokenize(sent)
        pinyins = label.strip().split(' ')
        # print('CHECK pinyins IN polyTTS_G2PDatasetMask:', pinyins)
        pinyins = ['er5' if i == 'r5' else i for i in pinyins]
        # assert len(toks) == len(pinyins)
        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_idxs = []
        poly_idxs = []
        output_masks = []
        # pinyin_targets = []
        for idx, char in enumerate(sent):
            prons = self.merge_cedict[char]
            if len(prons) >= 1:
                poly_idxs.append(idx)
                label_idxs.append(self.class2idx[pinyins[idx]])
                output_mask = []
                for output_mask_item in prons:
                    output_mask.append(self.class2idx[output_mask_item])
                output_masks.append(output_mask)
                # pinyin_targets.append(self.class2idx[pinyins[idx]])
            else:
                output_mask = []
                output_mask.append(self.class2idx[prons[0]])
                output_masks.append(output_mask)
                # pinyin_targets.append(self.class2idx[pinyins[idx]])

        # print('CHECK input_ids IN polyTTS_G2PDatasetMask:', input_ids)
        # print('CHECK label_idxs IN polyTTS_G2PDatasetMask:', label_idxs)
        # print('CHECK output_masks IN polyTTS_G2PDatasetMask:', output_masks)
        return input_ids, poly_idxs, label_idxs, output_masks, self.num_classes

def polyTTS_collate_fn_mask(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, max(lengths)

    def polyPosition2Bool(poly_ids, label_ids, max_input_len):
        polys_padded = torch.zeros(len(poly_ids), max_input_len).bool()
        labels_padded = torch.LongTensor(len(label_ids), max_input_len)
        labels_padded.zero_()
        for i in range(len(poly_ids)):
            poly_id = poly_ids[i]
            label_id = label_ids[i]
            # labels_padded[i, :label_id.shape[0]] = label_id
            for j in range(len(poly_id)):
                index = torch.LongTensor([[i, poly_id[j]]])

                value_poly = torch.ones(index.shape[0]).bool()
                polys_padded.index_put_(tuple(index.t()), value_poly)

                # value_label = torch.ones(index.shape[0]).bool()
                value_label = torch.LongTensor([label_id[j]])
                labels_padded.index_put_(tuple(index.t()), value_label)

        polys_padded = polys_padded.type(torch.BoolTensor)
        return polys_padded, labels_padded

    def mask(output_mask, max_input_len, num_classes):
        # lengths = [len(seq) for seq in sequences]
        # mask_output = torch.zeros((len(sequences), max(lengths), num_classes)).long()
        # print('CHECK num_classes:', num_classes)
        # print('CHECK num_classes:', num_classes.type())
        # mask_output = torch.FloatTensor(len(sequences), num_classes[0])
        # # mask_output.fill_(-float('inf')) 
        # mask_output.fill_(0.0) 
        # for i in range(len(output_mask)):
            # mask_sequence = output_mask[i]
            # # print('CHECK mask_sequence:', mask_sequence)
            # for j in range(len(mask_sequence)):
                # mask_character = mask_sequence[j]
                # index = torch.LongTensor([[i, mask_character]])
                # value = torch.ones(index.shape[0])
                # mask_output.index_put_(tuple(index.t()), value)

        mask_padded = torch.FloatTensor(len(output_mask), max_input_len, num_classes[0])
        mask_padded.zero_()
        for i in range(len(output_mask)):
            mask_sequence = output_mask[i]
            for j in range(len(mask_sequence)):
                mask_character = mask_sequence[j]
                for k in range(len(mask_character)):
                    index = torch.LongTensor([[i, j, mask_character[k]]])
                    value = torch.ones(index.shape[0])
                    mask_padded.index_put_(tuple(index.t()), value)
        return mask_padded

    data = filter (lambda x:x is not None, data)
    # data = filter (lambda x:len(x) == 5, data)
    # print('CHECK zip(*data):', zip(*data))
    # print('CHECK input:', zip(*data))
    data = [*data]
    # print('CHECK data length:', len(data))
    data_to_check_length = len(data)

    if data_to_check_length != 0:
        all_input_ids, poly_ids, label_ids, output_mask, num_classes = zip(*data)


        all_input_ids, max_input_len = merge(all_input_ids)
        # print('CHECK all_input_ids:', all_input_ids)
        # poly_ids = torch.tensor(poly_ids, dtype=torch.long)
        # label_ids = torch.tensor(label_ids, dtype=torch.long)
        poly_ids, label_ids = polyPosition2Bool(poly_ids, label_ids, max_input_len)
        # print('CHECK poly_ids:', poly_ids)
        # print('CHECK label_ids:', label_ids)
        output_mask = mask(output_mask, max_input_len, num_classes)
        # print('CHECK output_mask:', output_mask)
        return all_input_ids, poly_ids, label_ids, output_mask


def polyTTS_get_dataloader(use_output_mask, audiopaths_and_text, hparams,
                   batch_size, max_length, shuffle=False):

    if use_output_mask:
        dataset = polyTTS_G2PDatasetMask(audiopaths_and_text, hparams, max_length)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=polyTTS_collate_fn_mask,
                                num_workers=1)
    return dataloader


class TextMelLoader(torch.utils.data.Dataset):##继承于torch.utils.data.Dataset,参数是init里的4个参数。
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, polyphone_dict_file, mask_dict_file, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        # with codecs.open(polyphone_dict_file, 'r', 'utf-8') as usernames:
            # self.polyphone_dict = json.load(usernames)
        # with codecs.open(mask_dict_file, 'r', 'utf-8') as usernames:
        #     self.mask_dict = json.load(usernames)
        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)
        print("num classes: {}".format(len(self.class2idx)))
        num_classes = len(self.class2idx)
        with codecs.open(hparams.merge_cedict, 'r', 'utf-8') as usernames:
            self.merge_cedict = json.load(usernames)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text, poly_yinsu = audiopath_and_text[0], audiopath_and_text[1], audiopath_and_text[2]
        # print('CHECK audiopath', audiopath)
        text, poly_yinsu = text_and_pinyin_norm(text, poly_yinsu)
        # print('CHECK text_norm:', text)
        # print('CHECK poly_yinsu_norm:', poly_yinsu)
        input_ids, poly_idxs, label_idxs, output_masks = self.get_poly_label(text, poly_yinsu)
        # input_ids, poly_idxs, label_idxs, output_masks, pinyin_targets = self.get_poly_label(text, poly_yinsu)
        # print('CHECK input_ids:', input_ids)
        # print('CHECK poly_idxs:', poly_idxs)
        # print('CHECK label_idxs:', label_idxs)
        # print('CHECK output_masks:', output_masks)
        mel = self.get_mel(audiopath)

        return (input_ids, poly_idxs, label_idxs, output_masks, mel)
        # return (input_ids, poly_idxs, label_idxs, output_masks, mel, pinyin_targets)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            # print('CHECK load mel')
            melspec = torch.from_numpy(np.load(filename)).transpose(0, 1)
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))
        return melspec

    def get_poly_label(self, text, poly_yinsu):
        toks = self.tokenizer.tokenize(text)
        pinyins = poly_yinsu.strip().split(' ')
        pinyins = ['er5' if i == 'r5' else i for i in pinyins]
        assert len(toks) == len(pinyins)
        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        label_idxs = []
        poly_idxs = []
        output_masks = []
        # pinyin_targets = []
        for idx, char in enumerate(text):
            prons = self.merge_cedict[char]
            if len(prons) >= 1:
                poly_idxs.append(idx)
                label_idxs.append(self.class2idx[pinyins[idx]])
                output_mask = []
                for output_mask_item in prons:
                    output_mask.append(self.class2idx[output_mask_item])
                output_masks.append(output_mask)
                # pinyin_targets.append(self.class2idx[pinyins[idx]])
            else:
                output_mask = []
                label_idxs.append(self.class2idx[pinyins[idx]])
                # print('----------------CHECK PRONS:', prons, 'of', char, 'as', self.class2idx[prons[0]])
                output_mask.append(self.class2idx[prons[0]])
                output_masks.append(output_mask)
                # pinyin_targets.append(self.class2idx[pinyins[idx]])
        label_idxs = torch.tensor(label_idxs, dtype=torch.long)
        return input_ids, poly_idxs, label_idxs, output_masks
        # pinyin_targets = torch.tensor(pinyin_targets, dtype=torch.long)
        # return input_ids, poly_idxs, label_idxs, output_masks, pinyin_targets

    def __len__(self):
        return len(self.audiopaths_and_text)

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])


class TextMelCollate():  ##collate 核对 校对
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step, n_pinyin_symbols):
        self.n_frames_per_step = n_frames_per_step
        self.n_pinyin_symbols = n_pinyin_symbols

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        inputs_padded = torch.LongTensor(len(batch), max_input_len)
        inputs_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            input_id = batch[ids_sorted_decreasing[i]][0]
            inputs_padded[i, :input_id.shape[0]] = input_id
        # print('CHECK inputs_padded IN TextMelCollate:', inputs_padded)

        # pinyin_targets_padded = torch.LongTensor(len(batch), max_input_len)
        # pinyin_targets_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
            # pinyin_target_id = batch[ids_sorted_decreasing[i]][5]
            # pinyin_targets_padded[i, :pinyin_target_id.shape[0]] = pinyin_target_id
        # print('CHECK pinyin_targets_padded IN TextMelCollate:', pinyin_targets_padded)

        # poly_input_lengths = []
        # polys_padded = torch.LongTensor(len(batch), max_input_len)
        # polys_padded.zero_()
        # for i in range(len(ids_sorted_decreasing)):
            # poly_id = batch[ids_sorted_decreasing[i]][1]
            # polys_padded[i, :poly_id.shape[0]] = poly_id
            # poly_input_lengths.append(poly_id.shape[0])
        # print('CHECK polys_padded IN TextMelCollate:', polys_padded)

        # poly_input_lengths = []
        # polys_padded = torch.zeros(len(batch), max_input_len).bool()
        # for i in range(len(ids_sorted_decreasing)):
            # poly_id = batch[ids_sorted_decreasing[i]][1]
            # for j in range(len(poly_id)):
                # index = torch.LongTensor([[i, poly_id[j]]])
                # value = torch.ones(index.shape[0]).bool()
                # polys_padded.index_put_(tuple(index.t()), value)
            # poly_input_lengths.append(len(poly_id))
        # polys_padded = polys_padded.type(torch.BoolTensor)
        # print('CHECK polys_padded IN TextMelCollate:', polys_padded)

        poly_input_lengths = []
        polys_padded = torch.zeros(len(batch), max_input_len).bool()
        # labels_padded = torch.LongTensor(len(batch), max_input_len)
        # labels_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            poly_id = batch[ids_sorted_decreasing[i]][1]
            # label_id = batch[ids_sorted_decreasing[i]][2]
            # labels_padded[i, :label_id.shape[0]] = label_id
            for j in range(len(poly_id)):
                index = torch.LongTensor([[i, poly_id[j]]])

                value_poly = torch.ones(index.shape[0]).bool()
                polys_padded.index_put_(tuple(index.t()), value_poly)

                # value_label = torch.ones(index.shape[0]).bool()
                # value_label = torch.LongTensor([label_id[j]])
                # labels_padded.index_put_(tuple(index.t()), value_label)

            poly_input_lengths.append(len(poly_id))
        polys_padded = polys_padded.type(torch.BoolTensor)
        # print('CHECK polys_padded IN TextMelCollate:', polys_padded)
        # print('CHECK labels_padded IN TextMelCollate:', labels_padded)

        poly_input_lengths = torch.tensor(poly_input_lengths, dtype=torch.long)

        labels_padded = torch.LongTensor(len(batch), max_input_len)
        labels_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            label_id = batch[ids_sorted_decreasing[i]][2]
            labels_padded[i, :label_id.shape[0]] = label_id
        # print('CHECK labels_padded IN TextMelCollate:', labels_padded)

        # # TODO：ids_sorted_decreasing
        # _, poly_ids, label_ids, _, _ = zip(*batch)
        # print('CHECK poly_ids:', poly_ids)
        # print('CHECK label_ids:', label_ids)
        # poly_ids = torch.tensor(poly_ids, dtype=torch.long)
        # label_ids = torch.tensor(label_ids, dtype=torch.long)

        mask_padded = torch.FloatTensor(len(batch), max_input_len, self.n_pinyin_symbols)
        mask_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mask_sequence = batch[ids_sorted_decreasing[i]][3]
            for j in range(len(mask_sequence)):
                mask_character = mask_sequence[j]
                for k in range(len(mask_character)):
                    index = torch.LongTensor([[i, j, mask_character[k]]])
                    value = torch.ones(index.shape[0])
                    mask_padded.index_put_(tuple(index.t()), value)
        # print('CHECK mask_padded IN TextMelCollate:', mask_padded.shape)

        # loss_mask = torch.reshape(mask_padded, [-1, 1663])
        # print('CHECK loss_mask IN TextMelCollate:', loss_mask.shape)
        # loss_mask = torch.reshape(polys_padded, [-1,])
        # print('CHECK loss_mask IN TextMelCollate:', loss_mask.shape)
        # select_pred = mask_padded[loss_mask, :]
        # print('CHECK select_pred IN TextMelCollate:', select_pred.shape)
        select_pred = torch.argmax(mask_padded, 2)
        # print('CHECK select_pred IN TextMelCollate:', select_pred)

        # Right zero-pad mel-spec
        num_mels = batch[0][4].size(0)
        max_target_len = max([x[4].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][4]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return input_lengths, poly_input_lengths, inputs_padded, polys_padded, labels_padded, mask_padded, \
               mel_padded, gate_padded, output_lengths