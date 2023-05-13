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
from pytorch_utils import (move_data_to_device, count_parameters, count_flops, 
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
    tea_mel_bins = args.tea_mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model1_type = args.model1_type
    model2_type = args.model2_type
    model3_type = args.model3_type
    model4_type = args.model4_type
    loss_type = args.loss_type
    balanced = args.balanced
    augmentation = args.augmentation
    data_load_way = args.data_load_way
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    tea_learning_rate = args.tea_learning_rate
    resume_iteration = args.resume_iteration
    early_stop = args.early_stop
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = args.filename
    small_model_width = args.small_model_width
    large_model_width = args.large_model_width
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    lr_strategy = args.lr_strategy
    freeze_base = args.freeze_base
    mixup_alpha = args.mixup_alpha
    mixstyle_alpha = args.mixstyle_alpha
    mixstyle_p = args.mixstyle_p
    last_lr_value = args.last_lr_value
    warm_up_len = args.warm_up_len
    ramp_down_len = args.ramp_down_len
    ramp_down_start = args.ramp_down_start
    T = args.T

    num_workers = 8

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    loss_func = get_loss_func(loss_type)

    
    pretrain = True if pretrained_checkpoint_path else False

    # Paths
    black_list_csv = None
    
    train_indexes_hdf5_path = os.path.join(workspace, 'indexes', 
        '{}.h5'.format(data_type))

    eval_train_indexes_hdf5_path = os.path.join(workspace, 'indexes', 'train_eval.h5')

    eval_test_indexes_hdf5_path = os.path.join(workspace, 'indexes', 
        'eval.h5')

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), 
        'loss_type={}'.format(loss_type), 'balanced={}'.format(balanced), 
        'augmentation={}'.format(augmentation), 'batch_size={}'.format(batch_size), 
        'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename, 
        'sample_rate={},window_size={},hop_size={},mel_bins={},fmin={},fmax={}'.format(
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax), 
        'data_type={}'.format(data_type), 
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
    
    net=[model1_type, model2_type, model3_type, model4_type]
    for i in range(len(net)):
        if net[-1]== 'none':
            net.pop()

    num_net=len(net)
    models=[]
    optimizers=[]
    schedulers=[]
    evaluator=[]

    for i in range(num_net):
        if net[i] == 'Transfer_ResNet38_new_f':
            Model1 = eval('Transfer_ResNet38_new_f')
            model1 = Model1(sample_rate=sample_rate, window_size=tea_window_size, 
                hop_size=tea_hop_size, mel_bins=tea_mel_bins, fmin=fmin, fmax=fmax, 
                classes_num=classes_num, freeze_base=freeze_base)
            if pretrain:
                logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
                model1.load_from_pretrain(pretrained_checkpoint_path)
                            
            params_num = count_parameters(model1)
            logging.info('{} Parameters num: {}'.format(net[i],params_num))
 
            models.append(model1.to(device))
            
        elif net[i] == 'BcRes2NetModel_resnorm_bn_quant':
            Model2 = eval('BcRes2NetModel_resnorm_bn_quant')
            model2 = Model2(model_width=small_model_width, sample_rate=sample_rate, n_fft=window_size, 
                        window_size=window_size, hop_size=hop_size, mel_bins=mel_bins, classes_num=classes_num)
            
            params_num = count_parameters(model2)
            logging.info('{} Parameters num: {}'.format(net[i],params_num))

            models.append(model2.to(device))
        elif net[i] == 'BcRes2NetModel_deep_quant':
            Model3 = eval('BcRes2NetModel_deep_quant')
            model3 = Model3(model_width=small_model_width, sample_rate=sample_rate, n_fft=window_size, 
                        window_size=window_size, hop_size=hop_size, mel_bins=mel_bins, classes_num=classes_num)
                        
            params_num = count_parameters(model3)
            logging.info('{} Parameters num: {}'.format(net[i],params_num))
 
            models.append(model3.to(device))

        elif net[i] == 'BcRes2NetModel_width_resnorm_bn_quant':
            Model4 = eval('BcRes2NetModel_width_resnorm_bn_quant')
            model4 = Model4(model_width=large_model_width, sample_rate=sample_rate, n_fft=window_size, 
                        window_size=window_size, hop_size=hop_size, mel_bins=mel_bins, classes_num=classes_num)
                        
            params_num = count_parameters(model4)
            logging.info('{} Parameters num: {}'.format(net[i],params_num))
 
            models.append(model4.to(device))

        # elif net[i] == 'WRN_28_10' :
        #     models.append(model.Wide_ResNet(num_classes,args.use_weight_init).to(DEVICE))
    for i in range(num_net):
        if args.optim == 'SGD':
            optimizers.append(optim.SGD(models[i].parameters(), lr=learning_rate, weight_decay=0.05))
        elif args.optim == 'AdamW':
            if net[i] == 'Transfer_ResNet38_new_f':
                optimizers.append(optim.AdamW(models[i].parameters(), lr=tea_learning_rate, weight_decay=0.05))
            else:
                optimizers.append(optim.AdamW(models[i].parameters(), lr=learning_rate, weight_decay=0.05))
        if lr_strategy == 'up_down':
            schedule_lambda = \
                exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizers[i], schedule_lambda)
            schedulers.append(scheduler)
        else:
            pass

    for i in range(num_net):
        if net[i] == 'Transfer_ResNet38_new_f':
            evaluator1 = Evaluator(model=model1)
            evaluator.append(evaluator1)
        elif net[i] == 'BcRes2NetModel_resnorm_bn_quant':
            evaluator2 = Evaluator(model=model2)
            evaluator.append(evaluator2)
        elif net[i] == 'BcRes2NetModel_deep_quant':
            evaluator3 = Evaluator(model=model3)
            evaluator.append(evaluator3)
        elif net[i] == 'BcRes2NetModel_width_resnorm_bn_quant':
            evaluator4 = Evaluator(model=model4)
            evaluator.append(evaluator4)
    # Train sampler
    if balanced == 'none':
        Sampler = TrainSampler
    elif balanced == 'balanced':
        Sampler = BalancedTrainSampler
    elif balanced == 'alternate':
        Sampler = AlternateTrainSampler

    if data_load_way == '10s':
        train_dataset = DcaseDataset_10s(sample_rate=sample_rate)
    if data_load_way == 'conv_ir':
        train_dataset = DcaseDataset_10s_ir(sample_rate=sample_rate)
    train_sampler = Sampler(
        indexes_hdf5_path=train_indexes_hdf5_path, 
        batch_size=batch_size,
        black_list_csv=black_list_csv)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    eval_dataset = DcaseDataset(sample_rate=sample_rate)

    eval_test_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_test_indexes_hdf5_path, batch_size=batch_size)

    eval_test_loader = torch.utils.data.DataLoader(dataset=eval_dataset, 
        batch_sampler=eval_test_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)
    
    eval_train_sampler = EvaluateSampler(
        indexes_hdf5_path=eval_train_indexes_hdf5_path, batch_size=batch_size)

    eval_train_loader = torch.utils.data.DataLoader(dataset=eval_dataset, 
        batch_sampler=eval_train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)


    train_bgn_time = time.time()
    

    iteration = 0

    criterion_KLD = nn.KLDivLoss(reduction='batchmean')
    
    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    # model = torch.nn.DataParallel(model)

    # if 'cuda' in str(device):
    #     model.to(device)
    
    time1 = time.time()
    
    
        
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        # print('jinru batch')
        # Evaluate
        if (iteration % 2000 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()
            test_statistics = []
            bal_statistics = []
            for i in range(num_net):
                test_statistics.append(evaluator[i].evaluate(eval_test_loader))
                bal_statistics.append(evaluator[i].evaluate(eval_train_loader))
            # test_statistics = evaluator.evaluate(eval_test_loader)
            for i in range(num_net):

                train_acc_list = bal_statistics[i]['acc_list']
                train_acc_list = torch.Tensor(train_acc_list)
                train_precision = bal_statistics[i]['precision']
                logging.info(f'{net[i]} train test precision:{train_precision}'
                    )
                logging.info('{} train test mean_precision: {:.4f}, '.format(net[i],
                    np.mean(train_precision)))
                logging.info('{} train test macro_AP: {:.4f}, '.format(net[i],
                    np.mean(bal_statistics[i]['average_precision'])))
                logging.info('{} train test log loss:{:.4f}'.format(net[i],bal_statistics[i]['log_loss']))
                logging.info('{} train test acc:{:.4f}'.format(net[i],bal_statistics[i]['acc']))
                logging.info('{} train test acc_list:{}'.format(net[i],bal_statistics[i]['acc_list']))
                logging.info('{} train test macro_acc:{:.4f}'.format(net[i],bal_statistics[i]['macro_acc']))

                test_acc_list = test_statistics[i]['acc_list']
                test_acc_list = torch.Tensor(test_acc_list)
                test_precision = test_statistics[i]['precision']
                logging.info(f'{net[i]} Validate test precision:{test_precision}'
                    )
                logging.info('{} Validate test mean_precision: {:.4f}, '.format(net[i],
                    np.mean(test_precision)))
                logging.info('{} Validate test macro_AP: {:.4f}, '.format(net[i],
                    np.mean(test_statistics[i]['average_precision'])))
                logging.info('{} Validate test log loss:{:.4f}'.format(net[i],test_statistics[i]['log_loss']))
                logging.info('{} Validate test acc:{:.4f}'.format(net[i],test_statistics[i]['acc']))
                logging.info('{} Validate test acc_list:{}'.format(net[i],test_statistics[i]['acc_list']))
                logging.info('{} Validate test macro_acc:{:.4f}'.format(net[i],test_statistics[i]['macro_acc']))

            if test_statistics[0]['acc'] > 0.57:
                checkpoint = {
                'iteration': iteration, 
                'model': models[0].state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_{}_iterations.pth'.format(net[0],iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            if test_statistics[1]['acc'] > 0.57:
                checkpoint = {
                'iteration': iteration, 
                'model': models[1].state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_{}_iterations.pth'.format(net[1],iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            if test_statistics[2]['acc'] > 0.6:
                checkpoint = {
                'iteration': iteration, 
                'model': models[2].state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_{}_iterations.pth'.format(net[2],iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))

            if test_statistics[2]['acc'] > 0.68:
                checkpoint = {
                'iteration': iteration, 
                'model': models[3].state_dict()}

                checkpoint_path = os.path.join(
                    checkpoints_dir, '{}_{}_iterations.pth'.format(net[3],iteration))
                
                torch.save(checkpoint, checkpoint_path)
                logging.info('Model saved to {}'.format(checkpoint_path))


            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                    ''.format(iteration, train_time, validate_time))
            if iteration == 0 :
                for i in range(num_net):
                    checkpoint = {
                        'iteration': iteration, 
                        'model': models[i].state_dict()}

                    checkpoint_path = os.path.join(
                        checkpoints_dir, '{}_{}_iterations.pth'.format(net[i], iteration))
                    
                    torch.save(checkpoint, checkpoint_path)
                    logging.info('Model saved to {}'.format(checkpoint_path))

            logging.info('------------------------------------')

            train_bgn_time = time.time()
        
        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Forward
        # model.train()
        for i in range(num_net):
            models[i].train()

        output=[]
        losses=[]
        KLD_loss=[]
        label_loss=[]
        if 'mixup' in augmentation and random.random()>0.3:
            rn_indices, lam = mixup(batch_size, mixup_alpha)
            lam = lam.to(device)
        else:
            rn_indices=None
            lam=None
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
        for i in range(num_net):
            if 'mixup' in augmentation:
                batch_output_dict = models[i](batch_data_dict['waveform'], 
                    style_lmda=None, style_perm=None,rn_indices=rn_indices, lam=lam,spec_aug=spec_aug)
                """{'clipwise_output': (batch_size, classes_num), ...}"""

                if loss_type == 'clip_nll':
                    batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
                if loss_type == 'nn_ce':
                    pass
                batch_target_dict = {'target': batch_data_dict['target']* lam.reshape(batch_size, 1) + batch_data_dict['target'][rn_indices] * (1. - lam.reshape(batch_size, 1))}
                """{'target': (batch_size, classes_num)}"""

            elif 'mixstyle' in augmentation:
                
                batch_output_dict = models[i](batch_data_dict['waveform'], 
                    style_lmda=lmda, style_perm=perm,rn_indices=None, lam=None,spec_aug=spec_aug)

                if loss_type == 'clip_nll':
                    batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
                if loss_type == 'nn_ce':
                    pass

                batch_target_dict = {'target': batch_data_dict['target']}

            else:
                batch_output_dict = models[i](batch_data_dict['waveform'], None, None, None, None, spec_aug=spec_aug)

                if loss_type == 'clip_nll':
                    batch_output_dict['clipwise_output'] = torch.log_softmax((batch_output_dict['clipwise_output']),dim=-1)
                if loss_type == 'nn_ce':
                    pass

                batch_target_dict = {'target': batch_data_dict['target']}
            output.append(batch_output_dict)
        
        for k in range(num_net):
            label_loss.append(loss_func(output[k], batch_target_dict))
            KLD_loss.append(0)
            for l in range(num_net):
                if l!=k:
                    KLD_loss[k]+=criterion_KLD(F.log_softmax(output[k]['clipwise_output']/T,dim=-1),F.softmax(output[l]['clipwise_output']/T,dim=-1).detach())
            losses.append(label_loss[k]+KLD_loss[k]/(num_net-1))
        
        for i in range(num_net):
            optimizers[i].zero_grad()
            print(f'losses_{i}: {losses[i]}')
            losses[i].backward()
            optimizers[i].step()

            if lr_strategy == 'up_down' and iteration % 100 == 0:
                scheduler[i].step()
        
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
    parser_train.add_argument('--tea_mel_bins', type=int, default=256)
    parser_train.add_argument('--fmin', type=int, default=0)
    parser_train.add_argument('--fmax', type=int, default=32000) 
    parser_train.add_argument('--model1_type', type=str, default='none', required=True)
    parser_train.add_argument('--model2_type', type=str, default='none', required=True)
    parser_train.add_argument('--model3_type', type=str, default='none', required=True)
    parser_train.add_argument('--model4_type', type=str, default='none', required=True)
    parser_train.add_argument('--loss_type', type=str, default='ce', choices=['clip_bce', 'nn_ce', 'clip_nll', 'log_loss'])
    parser_train.add_argument('--balanced', type=str, default='balanced', choices=['none', 'balanced', 'alternate'])
    parser_train.add_argument('--augmentation', type=str, default='mixup', choices=['none', 'mixup', 'mixstyle'])
    parser_train.add_argument('--data_load_way', type=str, default='10s', choices=['none', '10s', 'conv_ir'])
    parser_train.add_argument('--lr_strategy', type=str, default='none', choices=['none', 'up_down'])
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--tea_learning_rate', type=float, default=1e-3)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--early_stop', type=int, default=1000000)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--small_model_width', type=int, default=3)
    parser_train.add_argument('--large_model_width', type=int, default=10)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--mixup_alpha', type=float, default=0.3)
    parser_train.add_argument('--mixstyle_alpha', type=float, default=0.3)
    parser_train.add_argument('--mixstyle_p', type=float, default=1.0)
    parser_train.add_argument('--num_epochs', type=int, default=10000)
    parser_train.add_argument('--warm_up_len', type=int, default=200)
    parser_train.add_argument('--ramp_down_len', type=int, default=1200)
    parser_train.add_argument('--ramp_down_start', type=int, default=200)
    parser_train.add_argument('--last_lr_value', type=float, default=0.)
    parser_train.add_argument('--optim', default='AdamW', choices=['Adam', 'SGD'], type=str)
    parser_train.add_argument('--T', type=int, default=1)
    
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')