import torch
import torch.nn.functional as F
import torch.nn as nn


def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss

def nn_ce(output_dict, target_dict):
    loss = nn.CrossEntropyLoss()(output_dict['clipwise_output'],target_dict['target'])
    return loss


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll
    if loss_type == 'nn_ce':
        return nn_ce