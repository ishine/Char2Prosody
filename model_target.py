from math import sqrt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from text import pinyin_to_yinsu

import parse_nk


def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()   ##每个类的__init__都要加这一句？
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,    ##ConvNorm是layers.py里定义的一个类
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,  ##LinearNorm是layers.py里定义的一个类
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
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
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
            x, input_lengths, batch_first=True)

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

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
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


class Mask_Softmax(nn.Module):
    def __init__(self, plus=1.0):
        super(Mask_Softmax, self).__init__()
        self.plus = plus
    def forward(self, logits):
        logits_exp = logits.exp()
        partition = logits_exp.sum(dim=-1, keepdim=True) + self.plus
        return logits_exp / partition


class Gumbel_Softmax(nn.Module):
    
    def __init__(self, temperature=1):
        super(Gumbel_Softmax, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        # initial temperature for gumbel softmax (default: 1)
        self.temperature = temperature
        self.mask_softmax = Mask_Softmax()

    def forward(self, logits, hard=False):
        y = self._gumbel_softmax_sample(logits, hard)
        return y

    def _sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, hard=False):
        sample = Variable(self._sample_gumbel(logits.size()[-1]), requires_grad=True)
        if logits.is_cuda:
            sample = sample.cuda()
        y = logits + sample
        # return self.softmax(y / self.temperature)
        y_soft = self.mask_softmax(y / self.temperature)

        if hard:
            # Straight through.
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret


class Select_Network(nn.Module):
    
    def __init__(self, hparams):
        super(Select_Network, self).__init__()

        # V = args.embed_num
        phoneme_features_dim = 512
        structure_features_dim = 812
        simility_dim = 256

        self.Wc = nn.Linear(structure_features_dim, simility_dim)
        # self.Wc = nn.Sequential(
            # nn.Linear(hparams.d_model, hparams.d_label_hidden),
            # LayerNormalization(hparams.d_label_hidden),
            # nn.ReLU(),
            # nn.Linear(hparams.d_label_hidden, label_vocab.size - 1),
            # )

        # W to calculate phoneme embedding
        self.Wp = nn.Linear(phoneme_features_dim, simility_dim)

        self.Wp2 = nn.Linear(256 *6, simility_dim)
        # self.Wp = nn.Sequential(
            # nn.Linear(hparams.d_model, hparams.d_label_hidden),
            # LayerNormalization(hparams.d_label_hidden),
            # nn.ReLU(),
            # nn.Linear(hparams.d_label_hidden, label_vocab.size - 1),
            # )

        # self.bias = Parameter(torch.Tensor(simility_dim))

        self.relu = nn.ReLU()

        self.V = nn.Linear(simility_dim, 6)

        self.gumbel_softmax = Gumbel_Softmax()

        # if self.args.static:
            # self.embed.weight.requires_grad = False


class Poly_Phoneme_Classifier(nn.Module):
    
    def __init__(self, hparams,
                       num_layers=1, num_heads=2, d_kv = 32, d_ff=1024,
                       d_positional=None,
                       num_layers_position_only=0,
                       relu_dropout=0.1, residual_dropout=0.1, attention_dropout=0.1):
        super(Poly_Phoneme_Classifier, self).__init__()

        # V = args.embed_num
        self.num_layers_position_only = num_layers_position_only
        self.embedding_features_dim = 1024
        self.structure_features_dim = 300
        self.select_model_dim = self.embedding_features_dim + self.structure_features_dim
        self.select_model_hidden_dim = 512
        self.n_pinyin_symbols = hparams.n_pinyin_symbols

        d_k = d_v = d_kv


        # self.linear_pre = nn.Sequential(
            # nn.Linear(self.embedding_features_dim, self.select_model_hidden_dim),
            # parse_nk.LayerNormalization(self.select_model_hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.select_model_hidden_dim, self.embedding_features_dim),
            # )
        self.linear_pre = nn.Sequential(
            nn.Linear(self.select_model_dim, self.select_model_hidden_dim),
            parse_nk.LayerNormalization(self.select_model_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.select_model_hidden_dim, self.embedding_features_dim),
            )
        # self.linear_pre = nn.Linear(self.select_model_dim, self.select_model_hidden_dim)


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
            nn.ReLU(),
            nn.Linear(self.select_model_hidden_dim, self.n_pinyin_symbols),
            )

        self.gumbel_softmax = Gumbel_Softmax()
        self.mask_softmax = Mask_Softmax()


    def forward(self, additional_select_features, mask_padded):

        batch_size_inner = additional_select_features.size(0)
        # print('CHECK additional_select_features IN Poly_Phoneme_Classifier:', additional_select_features.shape)
        # additional_select_features = torch.reshape(additional_select_features, [-1, self.select_model_dim])
        # print('CHECK additional_select_features IN Poly_Phoneme_Classifier:', additional_select_features.shape)
        res = self.linear_pre(additional_select_features)
        # print('CHECK res IN Poly_Phoneme_Classifier:', res.shape)
        # res = torch.reshape(res, [batch_size_inner, -1, self.embedding_features_dim])
        # print('CHECK res IN Poly_Phoneme_Classifier:', res.shape)
        res = torch.reshape(res, [-1, self.embedding_features_dim])

        for i, (attn, ff) in enumerate(self.stacks):
            if i >= self.num_layers_position_only:
                res, current_attns = attn.select_forward(res)
            else:
                res, current_attns = attn.select_forward(res)
            # res = ff(res, batch_idxs)

        res = torch.reshape(res, [batch_size_inner, -1, self.embedding_features_dim])
        select_without_mask = self.linear_label(res)

        # print('CHECK select_without_mask:', select_without_mask[0, 0, 200:500])
        # print('CHECK select_without_mask:', select_without_mask.shape)

        # _attn_mask = mask_padded == float('-inf')
        # mask_padded_new = mask_padded.data.masked_fill(_attn_mask, 0)

        # select_with_mask = torch.mul(select_without_mask, mask_padded)
        select_with_mask = select_without_mask + mask_padded

        # # print('CHECK mask_padded:', mask_padded)
        # # print('CHECK mask_padded:', mask_padded.shape)

        # print('CHECK select_with_mask:', select_with_mask[0, 0, 200:500])
        # print('CHECK select_with_mask:', select_with_mask.shape)

        select_pred = self.gumbel_softmax(select_with_mask, True)
        # select_pred = self.mask_softmax(select_with_mask)
        # select_pred = F.softmax(select_with_mask, dim=1)

        # select_pred_without_mask = self.gumbel_softmax(select_without_mask)
        # select_pred = torch.mul(select_pred_without_mask, mask_padded)


        # print('CHECK select_pred:', select_pred)
        # print('CHECK select_pred:', select_pred.shape)

        return select_pred

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
        # print('CHECK target IN Poly_Phoneme_Classifier select_acc:', target.shape)
        # print('CHECK pred IN Poly_Phoneme_Classifier select_acc:', pred.shape)
        # print('CHECK pred IN Poly_Phoneme_Classifier select_acc:', pred)
        pred = torch.argmax(pred, 2).cpu()
        target = target.cpu()
        mask = mask.cpu()
        # print('CHECK pred IN Poly_Phoneme_Classifier select_acc:', pred)
        # print('CHECK target IN Poly_Phoneme_Classifier select_acc:', target)
        # print('CHECK mask IN Poly_Phoneme_Classifier select_acc:', mask)

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
                clas = 1539 - sum(mask[i, j, :] == -float('inf')).numpy()
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


        # print('CHECK nomo_accuracy:', pred_nomo)
        # print('CHECK poly_accuracy:', target_nomo)

        nomo_accuracy = np.mean((np.array(pred_nomo)==np.array(target_nomo)))
        poly_accuracy = np.mean((np.array(pred_poly)==np.array(target_poly)))
        pad_accuracy = np.mean((np.array(pred_pad)==np.array(target_pad)))
        # print('CHECK nomo_accuracy:', nomo_accuracy)
        print('CHECK poly_accuracy:', poly_accuracy)
        # print('CHECK pad_accuracy:', pad_accuracy)
        return accuracy


class Structure_CNN(nn.Module):
    
    def __init__(self, hparams):
        super(Structure_CNN, self).__init__()

        # V = args.embed_num
        D = 1401
        # C = args.class_num
        C = 512
        Ci = 1
        Co = 100
        Ks = [3, 5, 7]

        convolutions = []
        for K in Ks:
            conv_layer = nn.Sequential(
                # ConvNorm(D, Co, kernel_size=K, stride=1, padding=int((K - 1) / 2), dilation=1, w_init_gain='relu'),
                ConvNorm(D, Co, kernel_size=K, stride=1, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(Co))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        # self.convs = nn.ModuleList([nn.Conv1d(in_channels=D, out_channels=Co, kernel_size=K) for K in Ks])
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        # self.linear = nn.Linear(len(Ks) * Co, C)

        # if self.args.static:
            # self.embed.weight.requires_grad = False

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = [self.relu(conv(x)) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        # x = torch.cat(x, 1)
        # x = self.dropout(x)  # (N, len(Ks)*Co)

        # print('CHECK self.training IN Structure CNN', self.training)
        cnn_out = [F.dropout(F.relu(conv(x)), 0.5, self.training) for conv in self.convolutions]
        pooling_out = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_out]
        result = torch.cat(pooling_out, 1)
        return result

        # for _ in range(hparams.encoder_n_convolutions):
            # conv_layer = nn.Sequential(
                # ConvNorm(hparams.encoder_embedding_dim,
                         # hparams.encoder_embedding_dim,
                         # kernel_size=hparams.encoder_kernel_size, stride=1,
                         # padding=int((hparams.encoder_kernel_size - 1) / 2),
                         # dilation=1, w_init_gain='relu'),
                # nn.BatchNorm1d(hparams.encoder_embedding_dim))
            # convolutions.append(conv_layer)
        # self.convolutions = nn.ModuleList(convolutions)


class Tacotron2(nn.Module):
    def __init__(self, hparams, pretrain_path):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.pinyin_to_yinsu_dict = to_gpu(torch.from_numpy(np.array(pinyin_to_yinsu()))).float()

        self.yinsu_embedding = nn.Embedding(
            hparams.n_yinsu_symbols, hparams.yinsu_symbols_embedding_dim)
        std_y = sqrt(2.0 / (hparams.n_yinsu_symbols + hparams.yinsu_symbols_embedding_dim))
        val_y = sqrt(3.0) * std_y  # uniform bounds for std
        self.yinsu_embedding.weight.data.uniform_(-val_y, val_y)

        # self.character_embedding = nn.Embedding(
            # hparams.n_character_symbols, hparams.character_symbols_embedding_dim)
        # std_c = sqrt(2.0 / (hparams.n_character_symbols + hparams.character_symbols_embedding_dim))
        # val_c = sqrt(3.0) * std_c  # uniform bounds for std
        # self.character_embedding.weight.data.uniform_(-val_c, val_c)

        # self.select_network = Select_Network(hparams)
        self.poly_phoneme_classifier = Poly_Phoneme_Classifier(hparams)

        print(f"Loading parameters from {pretrain_path}")
        info = torch_load(pretrain_path)
        self.tree_encoder = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

        self.structure_cnn = Structure_CNN(hparams)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        # text_padded, input_lengths, mel_padded, gate_padded, \
            # output_lengths = batch
        input_lengths, mask_padded, words_sorted, \
               select_target_padded, mel_padded, gate_padded, output_lengths = batch

        input_lengths = to_gpu(input_lengths).long()
        mask_padded = to_gpu(mask_padded).float()
        # mask_padded = to_gpu(mask_padded).long()

        select_target_padded = to_gpu(select_target_padded).long()

        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (input_lengths, mask_padded, select_target_padded, words_sorted, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded, select_target_padded))

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
        text_lengths, mask_padded, select_target, words_sorted, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        # batch_size_inner = select_target.size(0)
        # sentence_max_length = select_target.size(1)
        # print('CHECK words_sorted:', words_sorted)
        # predicted_trees, scores = self.tree_encoder.parse_batch(words_sorted)
        # print('CHECK scores:', scores)
        # tree_shows = [p.convert().linearize() for p in predicted_trees]
        # print('CHECK scores:', tree_shows)

        hidden_features, label_scores_charts, embedding_outputs = self.tree_encoder.parse_batch(words_sorted, return_label_scores_charts=True)
        # print('CHECK character_padded:', character_padded.shape)
        # print('CHECK hidden_features:', hidden_features.shape)

        batch_size_inner = hidden_features.size(0)
        sentence_max_length = hidden_features.size(1)

        structure_features = []
        for i, label_scores_chart in enumerate(label_scores_charts):
            sentence_length = label_scores_chart.size(0)
            label_scores_cnn_output = self.structure_cnn(label_scores_chart)
            label_scores_cnn_output = label_scores_cnn_output[1:-1, :]
            # label_scores_cnn_output = to_gpu(label_scores_cnn_output).float()
            label_scores_cnn_output = label_scores_cnn_output.float()
            label_scores_cnn_output_padder = torch.zeros([sentence_max_length - sentence_length + 2, 300])
            label_scores_cnn_output_padder = to_gpu(label_scores_cnn_output_padder).float()
            # label_scores_cnn_output_padder = label_scores_cnn_output_padder.float()
            label_scores_cnn_output_padded = torch.cat([label_scores_cnn_output, label_scores_cnn_output_padder], 0)
            structure_features.append(label_scores_cnn_output_padded)
        structure_features_reshape = torch.cat(structure_features, 0)
        structure_features = structure_features_reshape
        structure_features = torch.reshape(structure_features,  [batch_size_inner, -1, 300])
        # print('CHECK structure_features:', structure_features.shape)

        # select_target_to_loss = torch.reshape(select_target, [-1, 6])
        # print('CHECK select_target:', select_target_to_loss)
        # select_target = select_target_to_loss.unsqueeze(-1)

        # character_embedded_inputs = self.character_embedding(character_padded)
        # poly_yinsu_embedded_inputs = self.yinsu_embedding(poly_yinsu_padded)

        # Pretain to have structure features with character_embedded_inputs [B, L, 512]， actually 300
        # character_embedded_inputs = self.character_embedding(character_padded)

        additional_select_features = torch.cat([embedding_outputs, structure_features], 2)
        # print('CHECK embedding_outputs:', embedding_outputs)
        # print('CHECK mask_padded:', mask_padded)
        # select_pred = self.poly_phoneme_classifier(embedding_outputs, mask_padded)
        select_pred = self.poly_phoneme_classifier(additional_select_features, mask_padded)
        # print('CHECK select_pred:', select_pred)


        # select_pred_to_loss = torch.reshape(select_pred, [-1, 6])
        # print('CHECK select_pred:', select_pred_to_loss)
        # select_pred = select_pred_to_loss.unsqueeze(-1)

        select_accuracy = self.poly_phoneme_classifier.select_acc(select_target, select_pred, mask_padded)
        print('CHECK select_accuracy:', select_accuracy)


        # poly_yinsu_embedded_inputs = torch.reshape(poly_yinsu_embedded_inputs, [-1, 6, 512])
        # poly_yinsu_embedded_inputs = poly_yinsu_embedded_inputs.permute(0, 2, 1)
        # phoneme_selected_inputs = torch.bmm(poly_yinsu_embedded_inputs, select_pred)
        # phoneme_selected_inputs = phoneme_selected_inputs.squeeze(-1)
        # phoneme_selected_inputs = torch.reshape(phoneme_selected_inputs, [batch_size_inner, -1, 512])
        # phoneme_selected_inputs = phoneme_selected_inputs.permute(0, 2, 1)

        # print('CHECK pinyin_to_yinsu_dict:', self.pinyin_to_yinsu_dict)
        # print('CHECK pinyin_to_yinsu_dict:', self.pinyin_to_yinsu_dict.shape)
        # print('CHECK select_pred:', select_pred.shape)
        # yinsu_id_pred = torch.argmax(select_pred, 2)
        yinsu_id_inputs = torch.matmul(select_pred, self.pinyin_to_yinsu_dict)
        yinsu_id_inputs = torch.reshape(yinsu_id_inputs, [batch_size_inner, -1]).long()
        yinsu_embedded_inputs = self.yinsu_embedding(yinsu_id_inputs)
        # print('CHECK yinsu_embedded_inputs:', yinsu_embedded_inputs.shape)
        # print('CHECK yinsu_embedded_inputs:', yinsu_embedded_inputs)

        # Encoder Features Shape = [B, Features length, L]
        hidden_inputs = hidden_features.transpose(1, 2)
        structure_features = structure_features.transpose(1, 2)
        # additional_features = torch.cat([hidden_inputs, phoneme_selected_inputs, structure_features], 1)
        additional_features = torch.cat([hidden_inputs, structure_features], 1)
        # print('CHECK additional_features:', additional_features.shape)
        additional_features = additional_features.permute(0, 2, 1)

        additional_features_repeat = torch.repeat_interleave(additional_features, repeats=4, dim=1)
        # features_for_decoder = torch.cat([additional_features_repeat, yinsu_embedded_inputs], 2)
        features_for_encoder = torch.cat([additional_features_repeat, yinsu_embedded_inputs], 2)
        features_for_encoder = features_for_encoder.permute(0, 2, 1)

        encoder_outputs = self.encoder(features_for_encoder, text_lengths*4)
        # print('CHECK encoder_outputs:', encoder_outputs.shape)

        # mel_outputs, gate_outputs, alignments = self.decoder(features_for_decoder, mels, memory_lengths=text_lengths*4)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths*4)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet


        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], select_pred, output_lengths)

    # def inference(self, inputs):
    def inference(self, inputs, mask_padded):

        hidden_features, label_scores_charts, embedding_outputs = self.tree_encoder.parse_batch(inputs, return_label_scores_charts=True)
        batch_size_inner = hidden_features.size(0)
        sentence_max_length = hidden_features.size(1)
        structure_features = []
        for i, label_scores_chart in enumerate(label_scores_charts):
            sentence_length = label_scores_chart.size(0)
            label_scores_cnn_output = self.structure_cnn(label_scores_chart)
            label_scores_cnn_output = label_scores_cnn_output[1:-1, :]
            # label_scores_cnn_output = to_gpu(label_scores_cnn_output).float()
            label_scores_cnn_output = label_scores_cnn_output.float()
            label_scores_cnn_output_padder = torch.zeros([sentence_max_length - sentence_length + 2, 300])
            label_scores_cnn_output_padder = to_gpu(label_scores_cnn_output_padder).float()
            # label_scores_cnn_output_padder = label_scores_cnn_output_padder.float()
            label_scores_cnn_output_padded = torch.cat([label_scores_cnn_output, label_scores_cnn_output_padder], 0)
            structure_features.append(label_scores_cnn_output_padded)
        structure_features_reshape = torch.cat(structure_features, 0)
        structure_features = structure_features_reshape
        structure_features = torch.reshape(structure_features,  [batch_size_inner, -1, 300])

        additional_select_features = torch.cat([embedding_outputs, structure_features], 2)
        # select_pred = self.poly_phoneme_classifier.inference(additional_select_features)
        select_pred = self.poly_phoneme_classifier(additional_select_features, mask_padded)

        yinsu_id_inputs = torch.matmul(select_pred, self.pinyin_to_yinsu_dict)
        yinsu_id_inputs = torch.reshape(yinsu_id_inputs, [batch_size_inner, -1]).long()
        yinsu_embedded_inputs = self.yinsu_embedding(yinsu_id_inputs)

        hidden_inputs = hidden_features.transpose(1, 2)
        structure_features = structure_features.transpose(1, 2)
        additional_features = torch.cat([hidden_inputs, structure_features], 1)
        additional_features = additional_features.permute(0, 2, 1)

        additional_features_repeat = torch.repeat_interleave(additional_features, repeats=4, dim=1)
        features_for_encoder = torch.cat([additional_features_repeat, yinsu_embedded_inputs], 2)
        features_for_encoder = features_for_encoder.permute(0, 2, 1)
        encoder_outputs = self.encoder.inference(features_for_encoder)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        # hidden_inputs = hidden_features.transpose(1, 2)
        # structure_features = structure_features.transpose(1, 2)
        # additional_features = torch.cat([hidden_inputs, structure_features], 1)
        # additional_features = additional_features.permute(0, 2, 1)
        # additional_features_repeat = torch.repeat_interleave(additional_features, repeats=4, dim=1)
        # features_for_decoder = torch.cat([additional_features_repeat, yinsu_embedded_inputs], 2)
        # mel_outputs, gate_outputs, alignments = self.decoder.inference(features_for_decoder)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        # embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # encoder_outputs = self.encoder.inference(embedded_inputs)
        # mel_outputs, gate_outputs, alignments = self.decoder.inference(
            # encoder_outputs)

        # mel_outputs_postnet = self.postnet(mel_outputs)
        # mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], select_pred)

        return outputs
