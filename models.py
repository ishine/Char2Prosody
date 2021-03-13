import numpy as np
from math import sqrt

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from text import pinyin_to_yinsu, _yinsus_to_sequence, yinsu_symbols

from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel
from sklearn.metrics import accuracy_score

import parse_nk


def torch_load(load_path):    ##?
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


######################
##    Tacotron2     ##
######################
class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_input_dim[i],
                         hparams.encoder_output_dim[i],
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, enforce_sorted=False, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(   ##一个长短期记忆LSTM单元
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(     ##一个长短期记忆LSTM单元
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.tts_use_structure = hparams.tts_use_structure

        self.embedding = nn.Embedding(
            hparams.n_yinsu_symbols, hparams.yinsu_symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_yinsu_symbols + hparams.yinsu_symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)

        if self.tts_use_structure:
            self.linearlayer = nn.Linear(hparams.structure_feature_dim + hparams.yinsu_symbols_embedding_dim, hparams.encoder_embedding_dim)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        return outputs

    def forward(self, inputs, structure_features=None):
        text_inputs, text_lengths, mel_padded, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data
        embedded_inputs = self.embedding(text_inputs)
        if self.tts_use_structure:
            # embedded_inputs = self.linearlayer(torch.cat([embedded_inputs, structure_features], 2))
            embedded_inputs = torch.cat([embedded_inputs, structure_features], 2)

        encoder_outputs = self.encoder(embedded_inputs.transpose(1, 2), text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_padded, memory_lengths=text_lengths)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs, structure_features=None):
        embedded_inputs = self.embedding(inputs)
        if self.tts_use_structure:
            # embedded_inputs = self.linearlayer(torch.cat([embedded_inputs, structure_features], 2))
            embedded_inputs = torch.cat([embedded_inputs, structure_features], 2)

        encoder_outputs = self.encoder.inference(embedded_inputs.transpose(1, 2))
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs


def gather_nd(x,indices):
    newshape=indices.shape[:-1]+x.shape[indices.shape[-1]:]
    indices=indices.view(-1,ishape[-1]).tolist()
    out=torch.cat([x.__getitem__(tuple(i)) for i in indices])
    return out.reshape(newshape)


######################
#  Structure Feature #
######################

class Structure_CNN(nn.Module):

    def __init__(self, hparams, Ks):
        super(Structure_CNN, self).__init__()
        D = 1024
        Co = 100
        convolutions = []
        for K in Ks:
            conv_layer = nn.Sequential(
                ConvNorm(D, Co, kernel_size=K, stride=1, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(Co))
                ## nn.BatchNorm1d(dim),dim等于前一层输出的维度，BatchNorm层输出的维度也是dim。
                ## BatchNorm就是在深度神经网络训练过程中使得每一层网络的输入保持同分布的。
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        ## 区分nn.Sequential() 和 nn.ModuleList()

    def forward(self, x):
        x = x.permute(0, 2, 1)    ##将tensor的维度转换
        cnn_out = [F.dropout(F.relu(conv(x)), 0.5, self.training) for conv in self.convolutions]
        pooling_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]
        result = torch.cat(pooling_out, 1)
        return result
        ## 怎么看F下面的函数，如dropout,max_pool1d, 其中这里的max_pool1d和torch.nn.MaxPool1d()区别？
        ## torch.cat()函数，可以将多个张量在指定的维度进行拼接，得到新的张量。


######################
## Polyphonic model ##
######################

class poly_tonesandhi(nn.Module):   ##多音_连读变调
    def __init__(self, num_classes, hparams):
        super(poly_tonesandhi, self).__init__()
        self.g2ptransformermask = G2PTransformerMask(num_classes, hparams)
        self.poly_use_structure = hparams.poly_use_structure
        if self.poly_use_structure:
            self.g2ptransformermask.load_state_dict(torch.load(hparams.saved_model_path_structure_poly))
        else:
            self.g2ptransformermask.load_state_dict(torch.load(hparams.saved_model_path_poly))
        print('CHECK self.g2ptransformermask Model loaded and locked！')

        self.num_classes = num_classes
        self.embedding_features_dim = 1024
        self.select_model_hidden_dim = 512

        self.linear_pre = nn.Sequential(
            nn.Linear(self.num_classes, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),    ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, self.embedding_features_dim),
            )
            ## nn.Linear(in_features,out_features,bias=True) 
            ## 在PyTorch中的nn.Linear()表示线性变换，全连接层可以看作是nn.Linear()表示线性变换层再加上一个激活函数层所构成的结构。

        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.embedding_features_dim, self.embedding_features_dim, kernel_size=3, padding=1),  ##1d卷积，输入1024维，输出1024维，卷积核大小3，在输入两边进行0填充的数量1
            nn.BatchNorm1d(num_features=self.embedding_features_dim),
            nn.ReLU(True),   ##修正线性单元激活函数
            nn.Dropout(p=0.1)
        )

        self.linear_aft = nn.Sequential(
            nn.Linear(self.embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),      ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, num_classes),
            )
        ## 什么时候加nn.BatchNorm1d(),parse_nk.LayerNormalization(),nn.Dropout()?


    def forward(self, input_ids, attention_mask, poly_ids):
        inputs = {"input_ids": input_ids,
                  "poly_ids": poly_ids,
                  "attention_mask": attention_mask}
        if self.poly_use_structure:
            outputs, structure_features = self.g2ptransformermask(**inputs)
            hidden = self.linear_pre(outputs)
            hidden_cnn = hidden.permute(0, 2, 1)  ## 将tensor的维度转换
            hidden_cnn = self.conv_layers(hidden_cnn)
            hidden = hidden_cnn.permute(0, 2, 1)
            logits = self.linear_aft(hidden)
            return logits, structure_features
        else:
            outputs, _ = self.g2ptransformermask(**inputs)
            hidden = self.linear_pre(outputs)
            hidden_cnn = hidden.permute(0, 2, 1)
            hidden_cnn = self.conv_layers(hidden_cnn)
            hidden = hidden_cnn.permute(0, 2, 1)
            logits = self.linear_aft(hidden)
            return logits, None


    def select_poly(self, target, pred, output_mask, mask):
        target = torch.reshape(target, [-1,])
        pred = torch.reshape(pred, [-1, self.num_classes])
        output_mask = torch.reshape(output_mask, [-1, self.num_classes])
        mask = torch.reshape(mask, [-1,])
        target = torch.masked_select(target, mask)
        pred = pred[mask, :]
        output_mask = output_mask[mask, :]
        ## torch.reshape(input = A ,shape = (2,-1))函数改变输入张量的形状。
        ## torch.masked_select() 会将满足mask的指示，将满足条件的点选出来。

        return target, pred, output_mask

class cxypoly_phoneme_classifier(nn.Module):
    pass

class G2PTransformerMask(nn.Module):
    def __init__(self, num_classes, hparams):
        super(G2PTransformerMask, self).__init__()
        self.bert = BertModel.from_pretrained('./bert/bert-base-chinese')
        self.poly_phoneme_classifier = Poly_Phoneme_Classifier(hparams)
        self.cxypoly_phoneme_classifier = Poly_Phoneme_Classifier(hparams)
        self.linear = nn.Linear(1024, num_classes)
        self.num_classes = num_classes
        self.transformer_embedding_features_dim = 1324
        self.embedding_features_dim = 1024
        self.select_model_hidden_dim = 512
        self.structure_feature_dim = hparams.structure_feature_dim
        self.poly_use_structure = hparams.poly_use_structure
        if self.poly_use_structure:
            info = torch_load(hparams.pretrain_model_path_structure)
            self.tree_encoder = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])
            self.tree_shared_linear = nn.Sequential(
                nn.Linear(1401, 512),
                parse_nk.LayerNormalization(512),
                nn.ReLU(),      ##修正线性单元激活函数
                nn.Linear(512, 1024),
            )
            self.structure_cnn_poly = Structure_CNN(hparams, [1, 3, 5])
            self.structure_cnn_tts = Structure_CNN(hparams, [3, 5, 7])
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.bert_embedding_features_dim = 768 + self.structure_feature_dim
        else:
            self.bert_embedding_features_dim = 768

        self.linear_pre = nn.Sequential(
            nn.Linear(self.bert_embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),    ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, self.transformer_embedding_features_dim),
            )

        self.linear_aft = nn.Sequential(
            nn.Linear(self.embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),    ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, num_classes),
            )

    def forward(self, input_ids, attention_mask, poly_ids):
        batch_size = input_ids.size(0)

        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask}
        outputs = self.bert(**inputs)
        hidden = outputs[0][0]

        if self.poly_use_structure:
            words_sorted = []
            for i in range(batch_size):
                word_sen = self.tokenizer.convert_ids_to_tokens(input_ids[i])
                word_sen = [i for i in word_sen if i != '[PAD]']
                words_sorted.append(word_sen)
            hidden_features, label_scores_charts, embedding_outputs = self.tree_encoder.parse_batch(words_sorted,
                                                                                                return_label_scores_charts=True)
            batch_size_inner = hidden_features.size(0)
            sentence_max_length = hidden_features.size(1)
            structure_features_poly = []
            structure_features_tts = []
            for i, label_scores_chart in enumerate(label_scores_charts):
                sentence_length = label_scores_chart.size(0)
                label_shared_linear_output = self.tree_shared_linear(label_scores_chart)
                label_scores_cnn_poly_output = self.structure_cnn_poly(label_shared_linear_output)
                label_scores_cnn_tts_output = self.structure_cnn_tts(label_shared_linear_output)
                label_scores_cnn_poly_output = label_scores_cnn_poly_output[1:-1, :]
                label_scores_cnn_tts_output = label_scores_cnn_tts_output[1:-1, :]
                label_scores_cnn_poly_output = label_scores_cnn_poly_output.float()
                label_scores_cnn_tts_output = label_scores_cnn_tts_output.float()
                label_scores_cnn_output_padder = torch.zeros([sentence_max_length - sentence_length + 2, self.structure_feature_dim])
                label_scores_cnn_output_padder = to_gpu(label_scores_cnn_output_padder).float()
                label_scores_cnn_poly_output_padded = torch.cat([label_scores_cnn_poly_output, label_scores_cnn_output_padder], 0)
                label_scores_cnn_tts_output_padded = torch.cat([label_scores_cnn_tts_output, label_scores_cnn_output_padder], 0)
                structure_features_poly.append(label_scores_cnn_poly_output_padded)
                structure_features_tts.append(label_scores_cnn_tts_output_padded)
            structure_features_poly_reshape = torch.cat(structure_features_poly, 0)
            structure_features_tts_reshape = torch.cat(structure_features_tts, 0)
            structure_features_poly = torch.reshape(structure_features_poly_reshape, [batch_size_inner, -1, self.structure_feature_dim])
            structure_features_tts = torch.reshape(structure_features_tts_reshape, [batch_size_inner, -1, self.structure_feature_dim])
            hidden = torch.cat([hidden, structure_features_poly], 2)

        hidden = self.linear_pre(hidden)
        transformer_output = self.poly_phoneme_classifier.forward_train_polyphonic(hidden)
        transformer_output = self.cxypoly_phoneme_classifier.forward_train_polyphonic(transformer_output)
        logits = self.linear_aft(transformer_output)

        if self.poly_use_structure:
            return logits, structure_features_tts
        else:
            return logits, None

    def select_acc(self, target, pred, mask):

        target = torch.reshape(target, [-1,])
        pred = torch.reshape(pred, [-1, self.num_classes])
        mask = torch.reshape(mask, [-1,])
        target = torch.masked_select(target, mask)
        pred = pred[mask, :]
        pred = torch.argmax(pred, 1).cpu()
        target = target.cpu()
        accuracy = np.mean((target==pred).numpy())
        return accuracy

    def select_poly(self, target, pred, output_mask, mask):
        target = torch.reshape(target, [-1,])
        pred = torch.reshape(pred, [-1, self.num_classes])
        output_mask = torch.reshape(output_mask, [-1, self.num_classes])
        mask = torch.reshape(mask, [-1,])
        target = torch.masked_select(target, mask)
        pred = pred[mask, :]
        output_mask = output_mask[mask, :]

        return target, pred, output_mask


## Mixed model
def masked_augmax(logits, mask, dim, min_val=-1e7):
    logits = logits.exp()
    logits = logits.mul(mask)
    # one_minus_mask = (1.0 - mask).byte()
    # replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    # max_value, _ = replaced_vector.max(dim=dim)
    max_value = torch.argmax(logits, dim=2)
    return max_value


class Mask_Softmax(nn.Module):
    def __init__(self, plus=1.0):
        super(Mask_Softmax, self).__init__()
        self.plus = plus
    def forward(self, logits, output_mask):
        logits = logits + (output_mask + 1e-45).log()
        return torch.nn.functional.log_softmax(logits, dim=-1)


class Gumbel_Softmax(nn.Module):
    def __init__(self, temperature=1):
        super(Gumbel_Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # initial temperature for gumbel softmax (default: 1)
        self.temperature = temperature
        self.mask_softmax = Mask_Softmax()

    def forward(self, logits, output_mask, hard=False):
        y = self._gumbel_softmax_sample(logits, output_mask, hard)
        return y

    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, output_mask, hard=False):
        sample = Variable(self._sample_gumbel(logits.size()[-1]), requires_grad=True)
        if logits.is_cuda:
            sample = sample.cuda()
        y = logits + sample
        # return self.softmax(y / self.temperature)
        y_soft = self.mask_softmax(y / self.temperature, output_mask)

        if hard:
            # Straight through.
            index = y_soft.exp().mul(output_mask).max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret


class Poly_Phoneme_Classifier(nn.Module):
    
    def __init__(self, hparams,
                       num_layers=1, num_heads=2, d_kv = 32, d_ff=1024,
                       d_positional=None,
                       num_layers_position_only=0,
                       relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super(Poly_Phoneme_Classifier, self).__init__()

        self.num_layers_position_only = num_layers_position_only
        self.embedding_features_dim = 1024
        self.structure_features_dim = 300
        self.select_model_dim = self.embedding_features_dim + self.structure_features_dim
        self.select_model_hidden_dim = 512
        self.n_pinyin_symbols = hparams.n_pinyin_symbols
        d_k = d_v = d_kv

        self.linear_pre = nn.Sequential(
            nn.Linear(self.select_model_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),     ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, self.embedding_features_dim),
            )

        self.stacks = []
        for i in range(num_layers):
            attn = parse_nk.MultiHeadAttention(num_heads, self.embedding_features_dim, d_k, d_v, residual_dropout=residual_dropout, attention_dropout=attention_dropout, d_positional=d_positional)
            if d_positional is None:
                ff = parse_nk.PositionwiseFeedForward(self.embedding_features_dim, d_ff, relu_dropout=relu_dropout, residual_dropout=residual_dropout)
            else:
                ff = parse_nk.PartitionedPositionwiseFeedForward(self.embedding_features_dim, d_ff, d_positional, relu_dropout=relu_dropout, residual_dropout=residual_dropout)

            self.add_module(f"select_attn_{i}", attn)
            self.add_module(f"select_ff_{i}", ff)
            self.stacks.append((attn, ff))

        self.linear_label = nn.Sequential(
            nn.Linear(self.embedding_features_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),     ##修正线性单元激活函数
            nn.Linear(self.select_model_hidden_dim, self.n_pinyin_symbols),
            )

        self.gumbel_softmax = Gumbel_Softmax()
        self.mask_softmax = Mask_Softmax()


    def forward(self, additional_select_features, mask_padded):
        batch_size_inner = additional_select_features.size(0)
        res = self.linear_pre(additional_select_features)
        res = torch.reshape(res, [-1, self.embedding_features_dim])

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn.select_forward(res)
            else:
                res, current_attns = attn.select_forward(res)

        res = torch.reshape(res, [batch_size_inner, -1, self.embedding_features_dim])
        select_without_mask = self.linear_label(res)
        select_with_mask = select_without_mask + mask_padded

        select_pred = self.gumbel_softmax(select_with_mask, True)
        return select_pred

    def forward_train_polyphonic(self, additional_select_features):
        batch_size_inner = additional_select_features.size(0)
        res = self.linear_pre(additional_select_features)
        res = torch.reshape(res, [-1, self.embedding_features_dim])

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn.select_forward(res)
            else:
                res, current_attns = attn.select_forward(res)

        res = torch.reshape(res, [batch_size_inner, -1, self.embedding_features_dim])
        return res

    def inference(self, additional_select_features):
        batch_size_inner = additional_select_features.size(0)
        res = self.linear_pre(additional_select_features)
        res = torch.reshape(res, [-1, self.embedding_features_dim])

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn.select_forward(res)
            else:
                res, current_attns = attn.select_forward(res)

        res = torch.reshape(res, [batch_size_inner, -1, self.embedding_features_dim])
        select_without_mask = self.linear_label(res)
        # select_with_mask = select_without_mask + mask_padded
        select_with_mask = select_without_mask
        select_pred = self.gumbel_softmax(select_with_mask, True)
        return select_pred

    def select_acc(self, target, pred, mask):
        pred = torch.argmax(pred, 2).cpu()
        target = target.cpu()
        mask = mask.cpu()

        accuracy = np.mean((target==pred).numpy())

        pred_nomo = []
        target_nomo = []
        pred_poly = []
        target_poly = []
        pred_pad = []
        target_pad = []
        for i in range(mask.size(0)):
            for j in range(mask.size(1)):
                # clas = sum(mask[i, j, :]).numpy()
                clas = 1665 - sum(mask[i, j, :] == -float('inf')).numpy()
                # print('CHECK clas:', clas)
                if clas == 1:
                    pred_nomo.append(pred[i,j].numpy())
                    target_nomo.append(target[i,j].numpy())
                elif clas > 1:
                    pred_poly.append(pred[i,j].numpy())
                    target_poly.append(target[i,j].numpy())
                else:
                    pred_pad.append(pred[i,j].numpy())
                    target_pad.append(target[i,j].numpy())

        nomo_accuracy = np.mean((np.array(pred_nomo)==np.array(target_nomo)))
        poly_accuracy = np.mean((np.array(pred_poly)==np.array(target_poly)))
        pad_accuracy = np.mean((np.array(pred_pad)==np.array(target_pad)))
        print('CHECK nomo_accuracy:', nomo_accuracy)
        print('CHECK poly_accuracy:', poly_accuracy)
        print('CHECK pad_accuracy:', pad_accuracy)
        return accuracy


class Cascaded_Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Cascaded_Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.structure_feature_dim = hparams.structure_feature_dim
        self.num_classes = hparams.num_classes
        self.tts_use_structure = hparams.tts_use_structure and hparams.poly_use_structure   ## true and true

        self.pinyin_to_yinsu_dict = to_gpu(torch.from_numpy(np.array(pinyin_to_yinsu(hparams.class2idx)))).float()
        ## torch.from_numpy(ndarray) --> Tensor  把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.poly_phoneme_classifier = poly_tonesandhi(hparams.num_classes, hparams)

        if hparams.poly_use_structure:
            self.poly_phoneme_classifier.load_state_dict(torch.load(hparams.saved_model_path_sandhi_structure))
        else:
            self.poly_phoneme_classifier.load_state_dict(torch.load(hparams.saved_model_path_sandhi))

        self.mask_criterion = Gumbel_Softmax()
        self.tacotron2 = Tacotron2(hparams)

    def parse_batch(self, batch):    ## parse (对句子)作语法分析；作句法分析

        input_lengths, poly_input_lengths, inputs_padded, polys_padded, labels_padded, mask_padded, \
               mel_padded, gate_padded, output_lengths = batch

        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        poly_input_lengths = to_gpu(poly_input_lengths).long()

        inputs_padded = to_gpu(inputs_padded).long()
        polys_padded = to_gpu(polys_padded).bool()
        labels_padded = to_gpu(labels_padded).long()
        mask_padded = to_gpu(mask_padded).long()

        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (input_lengths, poly_input_lengths, inputs_padded, polys_padded, labels_padded, mask_padded, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded, labels_padded))

    def parse_output(self, outputs, select_pred, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

            outputs.append(select_pred)
        return outputs

    def forward(self, inputs):
        input_lengths, poly_input_lengths, inputs_padded, polys_padded, labels_padded, mask_padded, mel_padded, max_len, output_lengths = inputs
        input_lengths, poly_input_lengths, output_lengths = input_lengths.data, poly_input_lengths.data, output_lengths.data

        batch_size = input_lengths.size(0)

        attention_mask = torch.sign(inputs_padded)
        muti_inputs = {"input_ids": inputs_padded,
                       "poly_ids": polys_padded,
                       "attention_mask": attention_mask}    ## 字典
        logits, structure_features = self.poly_phoneme_classifier(**muti_inputs)

        labels_acc, logits_acc, output_mask_acc = self.poly_phoneme_classifier.select_poly(labels_padded, logits, mask_padded, polys_padded)

        logits = torch.reshape(logits, [-1, self.num_classes])
        mask_padded = torch.reshape(mask_padded, [-1, self.num_classes])
        logits = self.mask_criterion(logits, mask_padded, True)

        logits = torch.reshape(logits, [batch_size, -1, self.num_classes])
        # whether use labels or logits as pinyin inputs:
        # logits = F.one_hot(labels_padded, num_classes=1665).float()

        yinsu_id_inputs = torch.matmul(logits, self.pinyin_to_yinsu_dict)
        yinsu_id_inputs = torch.reshape(yinsu_id_inputs, [batch_size, -1])
        max_yinsu_len = yinsu_id_inputs.shape[1]

        if self.tts_use_structure:
            structure_features_repeat = torch.repeat_interleave(structure_features, repeats=4, dim=1)
            structure_features_select = []
            yinsu_id_inputs_seq = []
            for i in range(len(yinsu_id_inputs)):
                yinsu_id_inputs[i] = (yinsu_id_inputs[i] + 0.5).long()
                yinsu_mask = yinsu_id_inputs[i] != 70
                yinsu_id_seq_tmp = torch.masked_select(yinsu_id_inputs[i][:(input_lengths[i] * 4)],
                                                       yinsu_mask[:(input_lengths[i] * 4)])
                input_lengths[i] = len(yinsu_id_seq_tmp)
                yinsu_id_inputs_seq.append(F.pad(yinsu_id_seq_tmp, (0, max_yinsu_len - len(yinsu_id_seq_tmp))))
                yinsu_mask_300 = torch.reshape(torch.repeat_interleave(yinsu_mask, repeats=self.structure_feature_dim), [-1, self.structure_feature_dim])
                structure_features_tmp_select = torch.masked_select(structure_features_repeat[i], yinsu_mask_300)
                structure_features_tmp = F.pad(structure_features_tmp_select,
                                               (0, max_yinsu_len * self.structure_feature_dim - structure_features_tmp_select.shape[0]))
                structure_features_select.append(torch.reshape(structure_features_tmp, [-1, self.structure_feature_dim]))
            yinsu_id_inputs_seq = torch.cat(yinsu_id_inputs_seq, 0).long()
            yinsu_id_inputs_seq = torch.reshape(yinsu_id_inputs_seq, [batch_size, -1])
            structure_features_select = torch.cat(structure_features_select, 0)
            structure_features_select = torch.reshape(structure_features_select, [batch_size, -1, self.structure_feature_dim]).float()

            # calculate the acc of predict poly-phoneme
            logits_acc = self.mask_criterion(logits_acc, output_mask_acc, True)
            preds_acc = torch.argmax(logits_acc, dim=1)
            test_acc = accuracy_score(labels_acc.cpu().numpy(), preds_acc.cpu().numpy())
            print("test for phoneme acc in batch: {:.2f}".format(test_acc * 100))

            tacotron2_inputs = (yinsu_id_inputs_seq, input_lengths, mel_padded, max_len, output_lengths)
            tacotron2_outputs = self.tacotron2(tacotron2_inputs, structure_features_select)
            tacotron2_outputs.append(logits)

            return tacotron2_outputs

        else:
            yinsu_id_inputs_seq = []
            for i in range(len(yinsu_id_inputs)):
                yinsu_id_inputs[i] = (yinsu_id_inputs[i] + 0.5).long()
                yinsu_mask = yinsu_id_inputs[i] != 70
                yinsu_id_seq_tmp = torch.masked_select(yinsu_id_inputs[i][:(input_lengths[i] * 4)],
                                                       yinsu_mask[:(input_lengths[i] * 4)])
                input_lengths[i] = len(yinsu_id_seq_tmp)
                yinsu_id_inputs_seq.append(F.pad(yinsu_id_seq_tmp, (0, max_yinsu_len - len(yinsu_id_seq_tmp))))
            yinsu_id_inputs_seq = torch.cat(yinsu_id_inputs_seq, 0).long()
            yinsu_id_inputs_seq = torch.reshape(yinsu_id_inputs_seq, [batch_size, -1])

            # words_sorted = []
            # for i in range(len(inputs_padded)):
            #     word_sen = self.tokenizer.convert_ids_to_tokens(inputs_padded[i])
            #     word_sen = [i for i in word_sen if i != '[PAD]']
            #     words_sorted.append(word_sen)
            # print('CHECK inputs:', words_sorted)
            # for i in range(len(yinsu_id_inputs_seq)):
            #     a = []
            #     for j in range(len(yinsu_id_inputs_seq[i])):
            #         a.append(yinsu_symbols[int(yinsu_id_inputs_seq[i][j])])
            #     print(a)
            # print('CHECK yinsu_input_length:', input_lengths)\

            # calculate the acc of predict poly-phoneme
            logits_acc = self.mask_criterion(logits_acc, output_mask_acc, True)
            preds_acc = torch.argmax(logits_acc, dim=1)
            test_acc = accuracy_score(labels_acc.cpu().numpy(), preds_acc.cpu().numpy())
            print("test for phoneme acc in batch: {:.2f}".format(test_acc * 100))

            tacotron2_inputs = (yinsu_id_inputs_seq, input_lengths, mel_padded, max_len, output_lengths)
            tacotron2_outputs = self.tacotron2(tacotron2_inputs)
            tacotron2_outputs.append(logits)

            return tacotron2_outputs


    def inference(self, inputs):

        input_lengths, poly_input_lengths, inputs_padded, polys_padded, mask_padded = inputs
        input_lengths = input_lengths.data

        input_lengths = to_gpu(input_lengths).long()

        inputs_padded = to_gpu(inputs_padded).long()
        polys_padded = to_gpu(polys_padded).bool()
        mask_padded = to_gpu(mask_padded).long()

        batch_size = input_lengths.size(0)

        attention_mask = torch.sign(inputs_padded)
        muti_inputs = {"input_ids": inputs_padded,
                       "poly_ids": polys_padded,
                       "attention_mask": attention_mask}
        logits, structure_features = self.poly_phoneme_classifier(**muti_inputs)
        logits = torch.reshape(logits, [-1, self.num_classes])
        mask_padded = torch.reshape(mask_padded, [-1, self.num_classes])
        logits = self.mask_criterion(logits, mask_padded, True)
        logits = torch.reshape(logits, [batch_size, -1, self.num_classes])

        yinsu_id_inputs = torch.matmul(logits, self.pinyin_to_yinsu_dict)
        yinsu_id_inputs = torch.reshape(yinsu_id_inputs, [batch_size, -1])
        max_yinsu_len = yinsu_id_inputs.shape[1]

        if self.tts_use_structure:
            structure_features_repeat = torch.repeat_interleave(structure_features, repeats=4, dim=1)
            structure_features_select = []
            yinsu_id_inputs_seq = []
            for i in range(len(yinsu_id_inputs)):
                yinsu_id_inputs[i] = (yinsu_id_inputs[i] + 0.5).long()
                yinsu_mask = yinsu_id_inputs[i] != 70
                yinsu_id_seq_tmp = torch.masked_select(yinsu_id_inputs[i][:(input_lengths[i] * 4)],
                                                       yinsu_mask[:(input_lengths[i] * 4)])
                input_lengths[i] = len(yinsu_id_seq_tmp)
                yinsu_id_inputs_seq.append(F.pad(yinsu_id_seq_tmp, (0, max_yinsu_len - len(yinsu_id_seq_tmp))))
                yinsu_mask_300 = torch.reshape(torch.repeat_interleave(yinsu_mask, repeats=self.structure_feature_dim), [-1, self.structure_feature_dim])
                structure_features_tmp_select = torch.masked_select(structure_features_repeat[i], yinsu_mask_300)
                structure_features_tmp = F.pad(structure_features_tmp_select,
                                               (0, max_yinsu_len * self.structure_feature_dim - structure_features_tmp_select.shape[0]))
                structure_features_select.append(torch.reshape(structure_features_tmp, [-1, self.structure_feature_dim]))
            yinsu_id_inputs_seq = torch.cat(yinsu_id_inputs_seq, 0).long()
            yinsu_id_inputs_seq = torch.reshape(yinsu_id_inputs_seq, [batch_size, -1])
            structure_features_select = torch.cat(structure_features_select, 0)
            structure_features_select = torch.reshape(structure_features_select, [batch_size, -1, self.structure_feature_dim]).float()

            tacotron2_inputs = yinsu_id_inputs_seq
            tacotron2_outputs = self.tacotron2.inference(tacotron2_inputs, structure_features_select)

            return tacotron2_outputs

        else:
            yinsu_id_inputs_seq = []
            for i in range(len(yinsu_id_inputs)):
                yinsu_id_inputs[i] = (yinsu_id_inputs[i] + 0.5).long()
                yinsu_mask = yinsu_id_inputs[i] != 70
                yinsu_id_seq_tmp = torch.masked_select(yinsu_id_inputs[i][:(input_lengths[i] * 4)],
                                                       yinsu_mask[:(input_lengths[i] * 4)])
                input_lengths[i] = len(yinsu_id_seq_tmp)
                yinsu_id_inputs_seq.append(F.pad(yinsu_id_seq_tmp, (0, max_yinsu_len - len(yinsu_id_seq_tmp))))
            yinsu_id_inputs_seq = torch.cat(yinsu_id_inputs_seq, 0).long()
            yinsu_id_inputs_seq = torch.reshape(yinsu_id_inputs_seq, [batch_size, -1])

            tacotron2_inputs = yinsu_id_inputs_seq
            tacotron2_outputs = self.tacotron2.inference(tacotron2_inputs)

            return tacotron2_outputs

