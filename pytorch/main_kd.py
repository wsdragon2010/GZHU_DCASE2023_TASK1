import os
import random
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.distributions.beta import Beta
 
from utilities import (create_folder, get_filename, create_logging, Mixup, 
    StatisticsContainer)
from all_models_softmax import (Transfer_Cnn14_new_f, Transfer_ResNet38_new_f, BcResNetModel, 
    BcRes2NetModel, BcRes2NetModel_quant_no_feat, mel_model, BcRes2Net_spa_Model, 
    BcRes2NetModel_resnorm_bn_quant, BcRes2NetModel_width_resnorm_bn_quant, BcRes2NetModel_spa_bn_quant, BcRes2NetModel_deep_quant)
from pytorch_utils import (move_data_to_device, count_parameters, 
    mixup, exp_warmup_linear_down)
from data_generator import (DcaseDataset, DcaseDataset_10s, DcaseDataset_device, 
    DcaseDataset_ir, DcaseDataset_ir_device, DcaseDataset_10s_ir,TrainSampler, 
    BalancedTrainSampler, AlternateTrainSampler, EvaluateSampler, collate_fn)
from evaluate_macro_acc import Evaluator
import config
from loss import get_loss_func

logging.disable(logging.DEBUG)


def train(args):
    """Train AudioSet tagging model. 

    Args:
      dataset_dir: str
      workspace: str
      data_type: 'balanced_train' | 'full_train'
      window_size: int
      hop_size: int
      mel_bins: int
      model_type: str
      loss_type: 'clip_bce'
      balanced: 'none' | 'balanced' | 'alternate'
      augmentation: 'none' | 'mixup'
      batch_size: int
      learning_rate: float
      resume_iteration: int
      early_stop: int
      accumulation_steps: int
      cuda: bool
    """

    # Arugments & parameters
    workspace = args.workspace
    data_type = args.data_type
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    tea_window_size = args.tea_window_size
    tea_hop_size = args.tea_hop_size
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    tea_model_type = args.tea_model_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    data_load_way = args.data_load_way
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    model_width = args.model_width
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    stu_pretrained_checkpoint_path = args.stu_pretrained_checkpoint_path
    freeze_base = args.freeze_base
    kd_lambda = args.kd_lambda
    mixup_alpha = args.mixup_alpha
    T = args.T
    tea_mel_bins = args.tea_mel_bins
    mixstyle_alpha = args.mixstyle_alpha
    mixstyle_p = args.mixstyle_p
    lr_strategy = args.lr_strategy
    last_lr_value = args.last_lr_value
    warm_up_len = args.warm_up_len
    ramp_down_len = args.ramp_down_len
    ramp_down_start = args.ramp_down_start

    num_workers = 8

    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    pretrain = True if pretrained_checkpoint_path else False

    stu_pretrain = True if stu_pretrained_checkpoint_path else False
    
    train_indexes_hdf5_path = os.path.join(workspace, 'indexes', 
        '{}.h5'.format(data_type))

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'indexes', 
        'eval.h5')
    
    eval_train_indexes_hdf5_path = os.path.join(workspace, 'indexes', 'train_eval.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 'kd_lambda={}'.format(kd_lambda))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), model_type, 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))

    create_logging(logs_dir, filemode='w')
    logging.info(args)
    
    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'
    
    # Model
    Model = eval(model_type)
    model = Model(model_width=model_width, sample_rate=sample_rate, n_fft=window_size, 
                  window_size=window_size, hop_size=hop_size, 
                  mel_bins=mel_bins, classes_num=classes_num)
    Tea_Model = eval(tea_model_type)
    tea_model = Tea_Model(sample_rate=sample_rate, window_size=tea_window_size, 
        hop_size=tea_hop_size, mel_bins=tea_mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, freeze_base=freeze_base)
    
    if stu_pretrain:
        logging.info('Load stu model from {}'.format(stu_pretrained_checkpoint_path))
        stu_checkpoint = torch.load(stu_pretrained_checkpoint_path)
        model.load_state_dict(stu_checkpoint['model'])
    
    if pretrain:
        logging.info('Load teacher model from {}'.format(pretrained_checkpoint_path))
        checkpoint = torch.load(pretrained_checkpoint_path)
        tea_model.load_state_dict(checkpoint['model'])
     
    params_num = count_parameters(model)
    logging.info('Parameters num: {}'.format(params_num))
    
    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    if data_load_way == '10s':
        train_dataset = DcaseDataset_10s(sample_rate=sample_rate)
    if data_load_way == 'conv_ir':
        train_dataset = DcaseDataset_10s_ir(sample_rate=sample_rate)

    eval_dataset = DcaseDataset(sample_rate=sample_rate)

    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler
     
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_test_loader = torch.utils.data.DataLoader(dataset=eval_dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_train_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_train_indexes_hdf5_path, batch_size=batch_size)

    eval_train_loader = torch.utils.data.DataLoader(dataset=eval_dataset, 
        batch_sampler=eval_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)


    # Evaluator
    evaluator = Evaluator(model=model)
    tea_evaluator = Evaluator(model=tea_model)
            
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    if lr_strategy == 'up_down':
        schedule_lambda = \
            exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
    else:
        pass

    #loss f
    distillation_loss = nn.KLDivLoss(reduction='none', log_target=True)
    # distillation_loss = nn.CrossEntropyLoss()

    train_bgn_time = time.time()

    
    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename, 
            'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
            'data_type={}'.format(data_type), model_type, 
            'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
            'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
            '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        iteration = checkpoint['iteration']

    else:
        iteration = 0
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
        tea_model.to(device)
    
    time1 = time.time()
    

    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        
        # Evaluate
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(eval_train_loader)
            test_statistics = evaluator.evaluate(eval_test_loader)

            train_acc_list = bal_statistics['acc_list']
            train_acc_list = torch.Tensor(train_acc_list)
            train_precision = bal_statistics['precision']
            logging.info(f'train test precision:{train_precision}'
                )
            logging.info('train test mean_precision: {:.4f}, '.format(
                np.mean(train_precision)))
            logging.info('train test macro_AP: {:.4f}, '.format(
                np.mean(bal_statistics['average_precision'])))
            logging.info('train test log loss:{:.4f}'.format(bal_statistics['log_loss']))
            logging.info('train test acc:{:.4f}'.format(bal_statistics['acc']))
            logging.info('train test acc_list:{}'.format(bal_statistics['acc_list']))
            logging.info('train test macro_acc:{:.4f}'.format(bal_statistics['macro_acc']))

            precision = test_statistics['precision']
            logging.info(f'Validate test precision:{precision}'
                )
            logging.info('Validate test mean_precision: {:.4f}, '.format(
                np.mean(precision)))
            logging.info('Validate test macro_AP: {:.4f}, '.format(
                np.mean(test_statistics['average_precision'])))
            logging.info('Validate test acc:{:.4f}'.format(test_statistics['acc']))
            logging.info('Validate test log loss:{:.4f}'.format(test_statistics['log_loss']))
            logging.info('Validate tesst acc_list:{}'.format(test_statistics['acc_list']))
            logging.info('Validate test macro_acc:{:.4f}'.format(test_statistics['macro_acc']))

            
            tea_test = tea_evaluator.evaluate(eval_test_loader)
            tea_precision = tea_test['precision']
            logging.info(f'tea test precision:{precision}'
                )
            logging.info('tea test mean_precision: {:.4f}, '.format(
                np.mean(tea_precision)))
            logging.info('tea test macro_AP: {:.4f}, '.format(
                np.mean(tea_test['average_precision'])))
            logging.info('tea test acc:{:.4f}'.format(tea_test['acc']))
            logging.info('tea test log loss:{:.4f}'.format(tea_test['log_loss']))
            logging.info('tea test acc_list:{}'.format(tea_test['acc_list']))
            logging.info('tea test macro_acc:{:.4f}'.format(tea_test['macro_acc']))

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            if test_statistics['acc'] > 0.57:
                checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict()}
                # 'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            if iteration==0:
                checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict()}
                # 'sampler': train_sampler.state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            train_bgn_time = time.time()
            


        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        model.train()

        # Mixup lambda
        if 'mixup' in augmentation and random.random()>0.3:
            rn_indices, lam = mixup(batch_size, mixup_alpha)
            lam = lam.to(device)
        else:
            rn_indices=None
            lam=None

        # Mixstyle
        if random.random()>0.3:
            spec_aug = True
        else:
            spec_aug = False
        if random.random()<mixstyle_p:
            lmda = Beta(mixstyle_alpha, mixstyle_alpha).sample((batch_size, 1, 1, 1)).to(device)  # sample instance-wise convex weights
            perm = torch.randperm(batch_size).to(device)  # generate shuffling indices
        else:
            lmda=None
            perm=None

        if 'mixup' in augmentation:
            if pretrain:
                with torch.no_grad():
                    tea_model.eval()
                    batch_tea_output = torch.log_softmax((tea_model(batch_data_dict['waveform'])['clipwise_output'])/T,dim=-1)

            batch_output_dict = model(batch_data_dict['waveform'], 
                rn_indices,lam)
            batch_stu_output = torch.log_softmax((batch_output_dict['clipwise_output'])/T,dim=-1)
            if loss_type == 'clip_nll':
                batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
            if loss_type == 'nn_ce':
                pass
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']* lam.reshape(batch_size, 1) + batch_data_dict['target'][rn_indices] * (1. - lam.reshape(batch_size, 1))}
            """{'target': (batch_size, classes_num)}"""
            if pretrain:
                kd_loss = distillation_loss(batch_stu_output, batch_tea_output).mean(dim=1) * lam.reshape(batch_size) + \
                        distillation_loss(batch_stu_output, batch_tea_output[rn_indices]).mean(dim=1) * (1. - lam.reshape(batch_size))
                kd_loss = kd_loss.mean()

        elif 'mixstyle' in augmentation:
            if pretrain:
                with torch.no_grad():
                    tea_model.eval()
                    batch_tea_output = torch.log_softmax((tea_model(batch_data_dict['waveform'],
                                                                    style_lmda=lmda, style_perm=perm, 
                                                                    rn_indices=None, lam=None)['clipwise_output'])/T,dim=-1)
            batch_output_dict = model(batch_data_dict['waveform'], 
                style_lmda=lmda, style_perm=perm, rn_indices=None, lam=None, spec_aug=spec_aug)

            batch_stu_output = torch.log_softmax((batch_output_dict['clipwise_output'])/T,dim=-1)

            if loss_type == 'clip_nll':
                batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
            if loss_type == 'nn_ce':
                pass

            batch_target_dict = {'target': batch_data_dict['target']}
            if pretrain:
                kd_loss = distillation_loss(batch_stu_output, batch_tea_output).mean()

        else:
            if pretrain:
                with torch.no_grad():
                    tea_model.eval()
                    batch_tea_output = torch.log_softmax((tea_model(batch_data_dict['waveform'])['clipwise_output'])/T,dim=-1)

            batch_output_dict = model(batch_data_dict['waveform'], None,None,None,None,spec_aug)
            batch_stu_output = torch.log_softmax((batch_output_dict['clipwise_output'])/T,dim=-1)
            if loss_type == 'clip_nll':
                batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
            if loss_type == 'nn_ce':
                pass

            batch_target_dict = {'target': batch_data_dict['target']}
            if pretrain:
                kd_loss = distillation_loss(batch_stu_output, batch_tea_output).mean()
        # Loss
        label_loss = loss_func(batch_output_dict, batch_target_dict)
        if pretrain:
            loss = label_loss + kd_lambda*kd_loss
        else:
            loss = label_loss

        

        # Backward
        loss.backward()
        print(f'loss:{loss}')
        
        optimizer.step()
        optimizer.zero_grad()
        if lr_strategy == 'up_down' and iteration % 100 == 0:
                scheduler.step()
        
        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---'\
                .format(iteration, time.time() - time1))
            time1 = time.time()
        
        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1


            
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train') 
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--data_type', type=str, default='full_train', choices=['balanced_train', 'full_train'])
    parser_train.add_argument('--sample_rate', type=int, default=32000)
    parser_train.add_argument('--window_size', type=int, default=800)
    parser_train.add_argument('--hop_size', type=int, default=320)
    parser_train.add_argument('--mel_bins', type=int, default=80)
    parser_train.add_argument('--tea_window_size', type=int, default=800)
    parser_train.add_argument('--tea_hop_size', type=int, default=320)
    parser_train.add_argument('--tea_mel_bins', type=int, default=160)
    parser_train.add_argument('--fmin', type=int, default=0)
    parser_train.add_argument('--fmax', type=int, default=32000) 
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--tea_model_type', type=str, required=True)
    parser_train.add_argument('--loss_type', type=str, default='nn_ce', choices=['clip_bce', 'bce', 'nn_ce', 'clip_nll'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup', 'mixstyle'])
    parser_train.add_argument('--data_load_way', type=str, default='10s', choices=['none', '10s', 'conv_ir'])
    parser_train.add_argument('--lr_strategy', type=str, default='none', choices=['none', 'up_down'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=1000000)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--model_width', type=int, default=64)
    parser_train.add_argument('--T', type=int, default=64)
    parser_train.add_argument('--kd_lambda', type=float, default=50)
    parser_train.add_argument('--mixup_alpha', type=float, default=1.0)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--stu_pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--num_epochs', type=int, default=1000)
    parser_train.add_argument('--data', type=str, default='10s', choices=['none', '10s', 'conv_ir'])
    parser_train.add_argument('--mixstyle_alpha', type=float, default=1.0)
    parser_train.add_argument('--mixstyle_p', type=float, default=1.0)
    parser_train.add_argument('--warm_up_len', type=int, default=1000)
    parser_train.add_argument('--ramp_down_len', type=int, default=5000)
    parser_train.add_argument('--ramp_down_start', type=int, default=5000)
    parser_train.add_argument('--last_lr_value', type=float, default=0.001)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')