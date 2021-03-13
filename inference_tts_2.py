# coding:utf-8
import sys
import numpy as np
import torch
import os
import argparse
import json
import codecs

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hparams import create_hparams
from models import Cascaded_Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from scipy.io.wavfile import write
from transformers import BertTokenizer
from distributed import apply_gradient_allreduce


class TextMelLoaderEval(torch.utils.data.Dataset):
    def __init__(self, sentences, hparams):
        self.sentences = sentences

        with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
            self.class2idx = json.load(usernames)
        print("CHECK num classes: {}".format(len(self.class2idx)))
        num_classes = len(self.class2idx)
        with codecs.open(hparams.merge_cedict, 'r', 'utf-8') as usernames:
            self.merge_cedict = json.load(usernames)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # random.seed(hparams.seed)
        # random.shuffle(self.audiopaths_and_text)


    def get_poly_label(self, text):
        toks = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(toks)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        poly_idxs = []
        output_masks = []
        for idx, char in enumerate(text):
            prons = self.merge_cedict[char]
            if len(prons) > 1:
                poly_idxs.append(idx)
                output_mask = []
                for output_mask_item in prons:
                    output_mask.append(self.class2idx[output_mask_item])
                output_masks.append(output_mask)
            else:
                output_mask = []
                output_mask.append(self.class2idx[prons[0]])
                output_masks.append(output_mask)
        return (input_ids, poly_idxs, output_masks)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.get_poly_label(self.sentences[index])


class TextMelCollateEval():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, hparams):
        self.n_frames_per_step = hparams.n_frames_per_step
        self.n_pinyin_symbols = hparams.n_pinyin_symbols

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

        poly_input_lengths = []
        polys_padded = torch.zeros(len(batch), max_input_len).bool()
        for i in range(len(ids_sorted_decreasing)):
            poly_id = batch[ids_sorted_decreasing[i]][1]
            for j in range(len(poly_id)):
                index = torch.LongTensor([[i, poly_id[j]]])
                value_poly = torch.ones(index.shape[0]).bool()
                polys_padded.index_put_(tuple(index.t()), value_poly)
            poly_input_lengths.append(len(poly_id))
        polys_padded = polys_padded.type(torch.BoolTensor)
        poly_input_lengths = torch.tensor(poly_input_lengths, dtype=torch.long)

        mask_padded = torch.FloatTensor(len(batch), max_input_len, self.n_pinyin_symbols)
        mask_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mask_sequence = batch[ids_sorted_decreasing[i]][2]
            for j in range(len(mask_sequence)):
                mask_character = mask_sequence[j]
                for k in range(len(mask_character)):
                    index = torch.LongTensor([[i, j, mask_character[k]]])
                    value = torch.ones(index.shape[0])
                    mask_padded.index_put_(tuple(index.t()), value)
        # print('CHECK mask_padded IN TextMelCollate:', mask_padded.shape)

        return input_lengths, poly_input_lengths, inputs_padded, polys_padded, mask_padded


def poly_yinsu_to_mask_inference(text, mask_dict):
  words = []
  words_id = []
  mask_sequence = []

  for word in text:
    words.append(word)
    words_id.append(__character_symbol_to_id[word])

  for char in words:
    poly_pinyin_list = mask_dict[char]
    # Not fixed mask (to make 1539 mask in model) for every character
    mask_list = []
    for (pinyin, id) in  poly_pinyin_list.items():
      mask_list.append(id)
    mask_sequence.append(mask_list)

  # return words, words_id, mask_sequence
  return words, mask_sequence
  
def get_sentences(args):
    if args.text_file != '':
        with open(args.text_file, 'rb') as f:
            sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
    else:
        sentences = [args.sentences]
    print("Check sentences:", sentences)
    return sentences


def load_model(hparams):
    model = Cascaded_Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def inference(args):
    hparams = create_hparams()

    sentences = get_sentences(args)
    # sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

    model = load_model(hparams)
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.cuda().eval()#.half()

    test_set = TextMelLoaderEval(sentences, hparams)
    test_collate_fn = TextMelCollateEval(hparams)
    test_sampler = DistributedSampler(valset) if hparams.distributed_run else None
    test_loader = DataLoader(test_set, num_workers=0, sampler=test_sampler, batch_size=hparams.synth_batch_size, pin_memory=False, drop_last=True, collate_fn=test_collate_fn)

    taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            print("CHECK batch", i, batch)
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(batch)
            print('synthesize!!!', mel_outputs)
            for j in range(mel_outputs.size(0)):

                mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
                mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
                spec_from_mel_scaling = 1000
                spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
                spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
                spec_from_mel = spec_from_mel * spec_from_mel_scaling

                audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, args.griffin_iters)

                audio = audio.squeeze()
                audio = audio.cpu().numpy()
                #audio = audio.astype('int16')
                # audio_path = os.path.join('samples', "{}_synthesis.wav".format(args.out_filename))
                audio_path = os.path.join(args.out_filename, 'batch_{}_sentence_{}.wav'.format(i, j))
                write(audio_path, hparams.sampling_rate, audio)
                print(audio_path)

    # text = [list(text)]
    # print('CHECK INPUT mask_sequence:', text)
    # mask_padded = torch.FloatTensor(len(sequence), hparams.num_classes)
    # mask_padded.fill_(-float('inf'))
    # mel_outputs, mel_outputs_postnet, _, alignments = model.inference(text)
    # # sequence_id = np.array(sequence_id)[None, :]
    # # mask_sequence = np.array(mask_sequence)[None, :]
    # # sequence_id = torch.autograd.Variable(torch.from_numpy(sequence_id)).cuda().long()

    # # mask_sequence = torch.autograd.Variable(torch.from_numpy(mask_sequence)).cuda().long()

    # # mask_sequence = batch[ids_sorted_decreasing[i]][1]
    # for j in range(len(mask_sequence)):
        # mask_character = mask_sequence[j]
        # for k in range(len(mask_character)):
            # index = torch.LongTensor([[j, mask_character[k]]])
            # value = torch.zeros(index.shape[0])
            # mask_padded.index_put_(tuple(index.t()), value)

    # mel_outputs, mel_outputs_postnet, _, alignments = model.inference([sequence], mask_padded.cuda())
    # # mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, mask_sequence)

    # taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

    # mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    # mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    # spec_from_mel_scaling = 1000
    # spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    # spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    # spec_from_mel = spec_from_mel * spec_from_mel_scaling

    # audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, args.griffin_iters)

    # audio = audio.squeeze()
    # audio = audio.cpu().numpy()
    # #audio = audio.astype('int16')
    # audio_path = os.path.join('samples', "{}_synthesis.wav".format(args.out_filename))
    # write(audio_path, hparams.sampling_rate, audio)
    # print(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--sentences', type=str, help='text to infer', default='南国冬日里寒风冻死人不偿命')
    # ./textToSynthesize.txt
    parser.add_argument('-t', '--text_file', type=str, help='text file to infer', default='')
    parser.add_argument('-s', '--griffin_iters', type=int, help='griffin lim iters', default=60)
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path', default='./tts_cascaded_taco_syntax/checkpoint_28500')
    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='./samples')
    args = parser.parse_args()
    # inference(args.checkpoint, args.steps, args.text, args.out_filename)
    inference(args)