from torch import nn
import torch
from hparams import create_hparams
hparams = create_hparams()
class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets, mask_padded=None):
        mel_target, gate_target, select_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        select_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _, select_pred = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # print('CHECK target IN loss:', select_target)
        # print('CHECK target IN loss:', select_target.shape)
        # print('CHECK pred IN loss:', select_pred)
        # print('CHECK pred IN loss:', select_pred.shape)

        # if mask_padded != None:
            # # print('CHECK mask_loss 1：', mask_padded.shape)
            # loss_mask = torch.reshape(mask_padded, [-1, 1539])
            # # print('CHECK mask_loss 2：', loss_mask.shape)
            # loss_mask = torch.sum(loss_mask, dim=1)
            # # print('CHECK mask_loss 3：', loss_mask)
            # loss_mask = loss_mask.ge(2)
            # # print('CHECK mask_loss：', loss_mask)

        # loss_mask = torch.reshape(mask_padded, [-1, 1675])

        # loss_mask = (loss_mask == -float('inf')).sum(dim=1)
        # loss_mask = loss_mask.le(1537)

        # print('CHECK loss_mask 1', loss_mask)
        # loss_mask = torch.sum(loss_mask, dim=1)
        # loss_mask = loss_mask.ge(2)

        select_target = torch.reshape(select_target, [-1,])
        loss_mask = torch.reshape(mask_padded, [-1,])
        select_target.to(dtype=torch.int64)
        select_pred = torch.reshape(select_pred, [-1, hparams.num_classes])
        # print('CHECK loss_mask IN Tacotron2Loss:', loss_mask)

        # print('CHECK target IN loss:', select_target)
        # print('CHECK target IN loss:', select_target.shape)
        # print('CHECK pred IN loss:', select_pred)
        # print('CHECK pred IN loss:', select_pred.shape)

        # print('CHECK select_pred IN Tacotron2Loss:', select_pred.shape)
        # print('CHECK select_pred IN Tacotron2Loss:', select_pred[0])
        # print('CHECK select_target IN Tacotron2Loss:', select_target.shape)
        # print('CHECK select_target IN Tacotron2Loss:', select_target[0])

        select_target = torch.masked_select(select_target, loss_mask)
        select_pred = select_pred[loss_mask, :]
        # select_pred = torch.gather(select_pred, 1, loss_mask)
        # select_loss = nn.CrossEntropyLoss(reduction='none')(select_pred, select_target)
        # masked_select_loss = torch.mean(select_loss)

        # print('CHECK select_pred IN Tacotron2Loss:', select_pred.shape)
        # print('CHECK select_target IN Tacotron2Loss:', select_target.shape)

        # select_target = torch.reshape(select_target, [-1, 6])
        select_loss = nn.NLLLoss(reduction='none')(select_pred, select_target)
        # print('CHECK select_loss', select_loss.shape)
        # masked_select_loss = torch.masked_select(select_loss, loss_mask)
        # print('CHECK masked_select_loss', masked_select_loss)
        select_loss = torch.mean(select_loss)
        # print('CHECK masked_select_loss', masked_select_loss)

        # return mel_loss + gate_loss
        return mel_loss + gate_loss + 0.1 * select_loss, mel_loss, gate_loss, select_loss
        # return mel_loss + gate_loss + masked_select_loss
