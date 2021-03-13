import os
import time
import math
import json
import codecs
import argparse          ##python自带的命令行参数解析包
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
from models import G2PTransformerMask, poly_tonesandhi, Cascaded_Tacotron2
from data_utils import TextMelLoader, TextMelCollate, G2PDatasetMask, get_dataloader, polyTTS_get_dataloader
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams


def reduce_tensor(tensor, n_gpus):   ##？
    rt = tensor.clone()   ## 返回一个张量的副本，其与原张量的尺寸和数据类型相同
    dist.all_reduce(rt, op=dist.reduce_op.SUM)  ## 在所有机器上减少张量数据，通过获得最终的结果。在调用之后张量在所有过程中都是按位相同的。
    rt /= n_gpus   ## /=是除法赋值运算符  rt=rt/n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA." 
    ## cuda是否可用，cuda(compute unified device architecture)是显卡厂商NVIDIA推出的运算平台
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())
    ## torhc.cuda.set_device(device)设置当前设备。不鼓励使用此函数来设置，在大多数情况下，最好使用CUDA_VISIBLE_DEVICES环境变量。参数device(int)-所选设备，如果此参数为负，则此函数是无效操作。
    ## torch.cuda.device_count() 返回可得到的GPU数量

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    ## pytorch分布式训练
    ## backend str/Backend 是通信所用的后端，可以是'nccl''gloo'或者是一个torch.distributed.Backend类（Backend.GLOO）
    ## init_method str 这个URL指定了如何初始化互相通信的进程
    ## world_size init 执行训练的所有进程数
    ## rank int 这个进程的编号，也是其优先级
    ## group_name str 进程所在group的name

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    ## if not 用法：   if true才执行  （即 if not false）
    if not hparams.load_mel_from_disk:
        trainset = TextMelLoader(hparams.training_files, hparams.polyphone_dict_files, hparams.mask_dict_files, hparams)
        valset = TextMelLoader(hparams.validation_files, hparams.polyphone_dict_files, hparams.mask_dict_files, hparams)
    else:
        trainset = TextMelLoader(hparams.mel_training_files, hparams.polyphone_dict_files, hparams.mask_dict_files, hparams)
        valset = TextMelLoader(hparams.mel_validation_files, hparams.polyphone_dict_files, hparams.mask_dict_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step, hparams.num_classes)

    if hparams.distributed_run:   ##False
        train_sampler = DistributedSampler(trainset)
        ## 在多机多卡情况下分布式训练数据的读取，不同的卡读到的数据应该是不同的，利用sampler确保dataloader只会load到整个数据集的一个特定子集
        ## 它为每个子进程划分出一部分数据集，以避免不同进程之间的数据重复。
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    ## 定义一个可迭代的数据加载器
    train_loader = DataLoader(trainset, num_workers=0, shuffle=shuffle,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)
    ## dataset（Dataset类，决定数据从哪里读取及如何读取）  batch_size(每个batch的大小，批大小)  shuffle(是否进行shuffle操作，每个epoch是否乱序)  
    ## num_workers(加载数据时使用几个子进程)  drop_last(当样本数不能被batchsize整除时，是否舍弃最后一批数据)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Cascaded_Tacotron2(hparams).cuda()## 参数是hparams,因为继承了nn.module,所以有.cuda()
    if hparams.fp16_run:   ## False
        model.decoder.attention_layer.score_mask_value = finfo('float16').min ##？

    if hparams.distributed_run:   ## False
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=0,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        val_mel_loss = 0.0
        val_gate_loss = 0.0
        val_select_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            # original_words = x[3]
            # # print('CHECK original_words IN validate:', original_words)

            # _, _, select_target = y
            # select_target = np.array(select_target.cpu())
            # # print('CHECK select_target IN validate:', select_target)
            # np.savetxt('select_target.txt',select_target)

            # _, _, _, _, select_pred = y_pred
            # select_pred = np.array(select_pred.cpu())
            # select_pred = np.argmax(select_pred, axis=2)
            # # print('CHECK select_pred IN validate:', select_pred)
            # np.savetxt('select_pred.txt',select_pred)

            # mask_padded_to_show = np.array(mask_padded.cpu())
            # mask_padded_to_show = np.sum(mask_padded_to_show, axis=2)
            # # print('CHECK mask_padded_to_show IN validate:', mask_padded_to_show)
            # np.savetxt('select_mask.txt',mask_padded_to_show)

            mask_padded = x[3]
            loss, mel_loss, gate_loss, select_loss = criterion(y_pred, y, mask_padded)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_val_mel_loss = reduce_tensor(mel_loss.data, n_gpus).item()
                reduced_val_gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
                reduced_val_select_loss = reduce_tensor(select_loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
                reduced_val_mel_loss = mel_loss.item()
                reduced_val_gate_loss = gate_loss.item()
                reduced_val_select_loss = select_loss.item()
            val_loss += reduced_val_loss
            val_mel_loss += reduced_val_mel_loss
            val_gate_loss += reduced_val_gate_loss
            val_select_loss += reduced_val_select_loss
        val_loss = val_loss / (i + 1)
        val_mel_loss = val_mel_loss / (i + 1)
        val_gate_loss = val_gate_loss / (i + 1)
        val_select_loss = val_select_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
        logger.log_validation(val_loss, val_mel_loss, val_gate_loss, val_select_loss, model, y, y_pred, iteration)


def train_tts(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)  ##设置(CPU)生成随机数的种子，在每次重新运行程序时，同样的随机数生成代码得到的是同样的结果。
    torch.cuda.manual_seed(hparams.seed)## 设置当前GPU的随机数生成种子  torch.cuda.manual_seed_all(seed)设置所有GPU的随机数生成种子
    ## 手动设置种子一般可用于固定随机初始化的权重值，这样就可以让每次重新从头训练网络时的权重的初始值虽然是随机生成的但却是固定的。

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
    #                              weight_decay=hparams.weight_decay)

    for name, param in model.named_parameters():
        # frozen except tts
        # if name.split('.')[0] == 'poly_phoneme_classifier':
        #     param.requires_grad = False

        # frozen poly module except tone sandhi & tts
        # if name.split('.')[0] == 'poly_phoneme_classifier':
        #     if name.split('.')[1] != 'linear_pre' and name.split('.')[1] != 'conv_layers' and name.split('.')[1] != 'linear_aft':
        #         param.requires_grad = False

        # frozen except structure CNN & tonesandhi & tts
        if name.split('.')[0] == 'poly_phoneme_classifier':
            if name.split('.')[1] == 'g2ptransformermask':
                if name.split('.')[2] != 'structure_cnn_tts':
                    param.requires_grad = False
            elif name.split('.')[1] != 'linear_pre' and name.split('.')[1] != 'conv_layers' and name.split('.')[1] != 'linear_aft':
                param.requires_grad = False
            # else:
            #    param.requires_grad = False


    training_parameters_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(training_parameters_list, lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level='O2')
    ## apex是一款由Nvidia开发的基于PyTorch的混合精度训练加速神奇，用短短几行代码就能实现不同程度的混合精度加速，训练时间直接缩小一半。
    ## fp16:半精度浮点数，是一种计算机使用的二进制浮点数数据类型，使用2字节（16位）存储。
    ## fp16优点：减少显存占用；加快训练和推断的计算；张量核心的普及。缺点：量化误差。

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()   ## 返回当前的计算机系统时间
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # print('CHECK batch:', batch)

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            mask_padded = x[3]
            loss, mel_loss, gate_loss, select_loss = criterion(y_pred, y, mask_padded)  ## Tacotron2Loss(model_output,targets,mask_padded)
            ## 区分几种loss

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
                reduced_val_mel_loss = reduce_tensor(mel_loss.data, n_gpus).item()
                reduced_val_gate_loss = reduce_tensor(gate_loss.data, n_gpus).item()
                reduced_val_select_loss = reduce_tensor(select_loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
                reduced_val_mel_loss = mel_loss.item()
                reduced_val_gate_loss = gate_loss.item()
                reduced_val_select_loss = select_loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # print('CHECK  structure_cnn.convs.0.weight IS CHANGE:', model.structure_cnn.convolutions[0][0].conv.weight)

            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()
            ## 在用pytorch训练模型时，通常会在遍历epochs的过程中依次用到optimizer.zero_grad(),loss.backward(),optimizer.step()三个函数，总的来说，这三个函数的作用是先将梯度归零(optimizer.zero_grad()),
            ## 然后反向传播计算得到每个参数的梯度值(loss.backward()),最后通过梯度下降执行一步参数更新(optimizer.step())

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start  ## time.perf_counter()返回当前的计算机系统时间，只有连续两次perf_counter()进行差值才能有意义，一般用于计算程序运行时间。
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, reduced_val_mel_loss, reduced_val_gate_loss, reduced_val_select_loss, grad_norm, learning_rate, duration, iteration)

            if not is_overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1



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
        self.softmax = nn.Softmax(dim=-1)    ## dim=-1 最后一维取softmax
        # initial temperature for gumbel softmax (default: 1)
        self.temperature = temperature
        self.mask_softmax = Mask_Softmax()
        # self.mask_softmax = nn.LogSoftmax()

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
        # y_soft = self.mask_softmax(y / self.temperature)

        if hard:
            # Straight through.
            index = y_soft.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret


def masked_augmax(logits, mask, dim, min_val=-1e7):
    logits = logits.exp()
    logits = logits.mul(mask)
    # one_minus_mask = (1.0 - mask).byte()
    # replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    # max_value, _ = replaced_vector.max(dim=dim)
    max_value = torch.argmax(logits, dim=1)
    return max_value


def train_poly(args, hparams):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    print('CHECK HERE train poly ONLY')
    train_dataloader = get_dataloader(hparams.use_output_mask, hparams.train_file, hparams.train_label,
                                      hparams, hparams.poly_batch_size,
                                      hparams.poly_max_length, shuffle=True)

    val_dataloader = get_dataloader(hparams.use_output_mask, hparams.val_file, hparams.val_label,
                                    hparams, hparams.poly_batch_size,
                                    hparams.poly_max_length, shuffle=True)

    # test_dataloader = get_dataloader(args.use_output_mask, args.test_file, args.test_label,
                                     # args.class2idx, args.merge_cedict, args.poly_batch_size,
                                     # args.max_length, shuffle=True)

    with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
        class2idx = json.load(usernames)
    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    model = G2PTransformerMask(num_classes, hparams)
    device = torch.cuda.current_device()   ## 查看当前使用的gpu序号
    model = model.to(device)   ## 将模型加载到指定设备上
    for name, param in model.named_parameters():
        # frozen syntax module
        if name.split('.')[0] != 'tree_shared_linear' and name.split('.')[0] != 'structure_cnn_poly' \
                and name.split('.')[0] != 'linear_pre' and name.split('.')[0] != 'poly_phoneme_classifier' \
                and name.split('.')[0] != 'linear_aft':
            param.requires_grad = False
    training_parameters_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(training_parameters_list, lr=hparams.poly_lr)

    criterion = nn.NLLLoss()
    # mask_criterion = Mask_Softmax()
    mask_criterion = Gumbel_Softmax()
    model_dir = "./save/poly_only_syntax_frozen"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_acc = 0
    for epoch in range(hparams.poly_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader, start=1):
            # print('CEHCK batch:', batch)
            # if idx > 200:
            #     break
            batch = tuple(t.to(device) for t in batch)
            if hparams.use_output_mask:
                input_ids, poly_ids, labels, output_mask = batch
                mask = torch.sign(input_ids)
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask}
            else:
                input_ids, poly_ids, labels = batch
                mask = torch.sign(input_ids)  ## torch.sign(input,out=None) 符号函数，返回一个新张量，包含输入input张量每个元素的正负（大于0的元素对应1，小于0的元素对应-1，0还是0）
                inputs = {"input_ids": input_ids,
                          "poly_ids": poly_ids,
                          "attention_mask": mask}

            # inputs = {"input_ids": input_ids,
                      # "poly_ids": poly_ids,
                      # "attention_mask": mask}
            logits, _ = model(**inputs)

            batch_size = logits.size(0)
            logits = logits[torch.arange(batch_size), poly_ids]

            # logits = mask_criterion(logits, output_mask, True)
            logits = mask_criterion(logits, output_mask)
            loss = criterion(logits, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

            if idx % 100 == 0:    ## %取余
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
            with torch.no_grad():
                logits, _ = model(**inputs)

            batch_size = logits.size(0)
            logits = logits[torch.arange(batch_size), poly_ids]
            # logits = logits.exp()
            # output_mask_false = 1.0 - output_mask
            # logits = logits - output_mask_false
            # logits = mask_criterion(logits, output_mask, True)
            logits = mask_criterion(logits, output_mask)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            mask_preds = masked_augmax(logits, output_mask, dim=1).cpu().numpy()
            if not (preds == mask_preds).all():
                print('CHECK preds:', preds)
                print('CHECK mask_preds:', mask_preds)
                print('CHECK labels:', labels)
                print('CHECK output_mask:', np.where(output_mask.cpu().numpy()==1.0))

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


def train_poly_tts(args, hparams):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    print('CHECK HERE train TTS poly')
    train_dataloader = polyTTS_get_dataloader(hparams.use_output_mask, hparams.training_files,
                                      hparams, hparams.poly_batch_size,
                                      hparams.poly_max_length, shuffle=True)

    val_dataloader = polyTTS_get_dataloader(hparams.use_output_mask, hparams.validation_files,
                                      hparams, hparams.poly_batch_size,
                                      hparams.poly_max_length, shuffle=True)

    with codecs.open(hparams.class2idx, 'r', 'utf-8') as usernames:
        class2idx = json.load(usernames)
    print("num classes: {}".format(len(class2idx)))
    num_classes = len(class2idx)
    model = poly_tonesandhi(num_classes, hparams)
    device = torch.cuda.current_device()
    model = model.to(device)

    for name, param in model.named_parameters():
        # frozen syntax module
        if name.split('.')[0] == 'g2ptransformermask':
            if name.split('.')[1] != 'tree_shared_linear' and name.split('.')[1] != 'structure_cnn_poly' \
                    and name.split('.')[1] != 'linear_pre' and name.split('.')[1] != 'poly_phoneme_classifier' \
                    and name.split('.')[1] != 'linear_aft':
                param.requires_grad = False

    training_parameters_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(training_parameters_list, lr=hparams.poly_lr)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    # mask_criterion = Mask_Softmax()
    mask_criterion = Gumbel_Softmax()
    model_dir = "./save/poly_tts_CNN_syntax_frozen"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    best_acc = 0
    for epoch in range(hparams.poly_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader, start=1):
            # print('CEHCK batch:', batch)
            # if idx > 200:
            #     break
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

            logits, _ = model(**inputs)
            # logits = mask_criterion(logits, output_mask, True)
            labels, logits, output_mask = model.select_poly(labels, logits, output_mask, poly_ids)

            logits = mask_criterion(logits, output_mask)

            loss = criterion(logits, labels)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.zero_grad()

            if idx % 100 == 0:
                print("loss : {:.4f}".format(loss.item()))



        all_preds = []
        all_labels = []
        model.eval()
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
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
            with torch.no_grad():
                logits, _ = model(**inputs)

            labels, logits, output_mask = model.select_poly(labels, logits, output_mask, poly_ids)
            logits = mask_criterion(logits, output_mask)
            # labels, logits = model.select_poly(labels, logits, poly_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        train_acc = accuracy_score(labels, preds)
        print("epoch :{}, Trainging acc: {:.2f}".format(epoch, train_acc*100))



        all_preds = []
        # all_mask_preds = []
        all_labels = []
        model.eval()
        for batch in tqdm(val_dataloader, total=len(val_dataloader)):
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
            with torch.no_grad():
                logits, _ = model(**inputs)

            labels, logits, output_mask = model.select_poly(labels, logits, output_mask, poly_ids)
            logits = mask_criterion(logits, output_mask)
            # labels, logits = model.select_poly(labels, logits, poly_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        val_acc = accuracy_score(labels, preds)
        print("epoch :{}, acc: {:.2f}".format(epoch, val_acc*100))
        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = model.state_dict()
            save_file = os.path.join(
                model_dir, "{:.2f}_model.pt".format(val_acc*100))
            torch.save(state_dict, save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    parser.add_argument('--train_poly', dest='train_poly', action='store_true', default=False, required=False)
    parser.add_argument('--train_tts', dest='train_tts', action='store_true', default=False, required=False)

    args = parser.parse_args()
    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    if args.train_poly:
        # train_poly(args, hparams)
        train_poly_tts(args, hparams)
    if args.train_tts:
        train_tts(args.output_directory, args.log_directory, args.checkpoint_path,
            args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
