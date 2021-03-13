import sys
import numpy as np
import torch
import os
import argparse
import json
import codecs

from hparams import create_hparams
from models import Poly2Audio, Syntax2Audio
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from scipy.io.wavfile import write
from transformers import BertTokenizer
from distributed import apply_gradient_allreduce


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


def load_model(hparams):
    # model = Poly2Audio(hparams).cuda()
    model = Syntax2Audio(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model

def infer(checkpoint_path, griffin_iters, text, out_filename):
    hparams = create_hparams()

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()#.half()

    with codecs.open(hparams.merge_cedict, 'r', 'utf-8') as usernames:
        mask_dict = json.load(usernames)

    sequence = np.array(poly_yinsu_to_mask_inference(text, mask_dict))[None, :]
    print('CHECK INPUT sequence:', sequence)
    # sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    # sequence, mask_sequence = poly_yinsu_to_mask_inference(text, mask_dict)
    # print('CHECK INPUT sequence:', sequence)

    #tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    #text_seq = tokenizer.convert_tokens_to_ids(text)
    text = [list(text)]
    print('CHECK INPUT mask_sequence:', text)
    mask_padded = torch.FloatTensor(len(sequence), hparams.num_classes)
    mask_padded.fill_(-float('inf'))
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(text)
    # sequence_id = np.array(sequence_id)[None, :]
    # mask_sequence = np.array(mask_sequence)[None, :]
    # sequence_id = torch.autograd.Variable(torch.from_numpy(sequence_id)).cuda().long()

    # mask_sequence = torch.autograd.Variable(torch.from_numpy(mask_sequence)).cuda().long()

    # mask_sequence = batch[ids_sorted_decreasing[i]][1]
    for j in range(len(mask_sequence)):
        mask_character = mask_sequence[j]
        for k in range(len(mask_character)):
            index = torch.LongTensor([[j, mask_character[k]]])
            value = torch.zeros(index.shape[0])
            mask_padded.index_put_(tuple(index.t()), value)

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference([sequence], mask_padded.cuda())
    # mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence, mask_sequence)

    taco_stft = TacotronSTFT(hparams.filter_length, hparams.hop_length, hparams.win_length, sampling_rate=hparams.sampling_rate)

    mel_decompress = taco_stft.spectral_de_normalize(mel_outputs_postnet)
    mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], taco_stft.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling

    audio = griffin_lim(torch.autograd.Variable(spec_from_mel[:, :, :-1]), taco_stft.stft_fn, griffin_iters)

    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    #audio = audio.astype('int16')
    audio_path = os.path.join('samples', "{}_synthesis.wav".format(out_filename))
    write(audio_path, hparams.sampling_rate, audio)
    print(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='text to infer', default='我爱那塞北的雪')
    parser.add_argument('-s', '--steps', type=int, help='griffin lim iters', default=60)
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint path', default='./outdir_e2e_with_encoder/checkpoint_15000')
    parser.add_argument('-o', '--out_filename', type=str, help='output filename', default='sample')
    args = parser.parse_args()
    infer(args.checkpoint, args.steps, args.text, args.out_filename)