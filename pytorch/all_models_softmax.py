import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from pytorch_utils import do_mixup, interpolate, pad_framewise_output,mixstyle,syn_mixstyle
import math
 

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn14_no_f(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(Cnn14_no_f, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input):
        """
        Input: (batch_size, data_length)"""

        x = self.conv_block1(input, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
class Transfer_Cnn14_new_f(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14_new_f, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=25, time_stripes_num=2, 
            freq_drop_width=20, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.base = Cnn14_no_f(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'],strict=False)

    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)
            
        output_dict = self.base(x)
        embedding = output_dict['embedding']

        clipwise_output =  self.fc_transfer(embedding)
        # print(clipwise_output)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict 


def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNet38_no_f(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num):
        
        super(ResNet38_no_f, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None


        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        # init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)


    def forward(self, x):
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
    
class Transfer_ResNet38_new_f(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
        fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_ResNet38_new_f, self).__init__()
        audioset_classes_num = 527

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        
        self.base = ResNet38_no_f(sample_rate, window_size, hop_size, mel_bins, fmin, 
            fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'],strict=False)

    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):
        """Input: (batch_size, data_length)
        """
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2, 3)

        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))

        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)
            
        output_dict = self.base(x)
        embedding = output_dict['embedding']

        clipwise_output =  self.fc_transfer(embedding)
        # print(clipwise_output)
        output_dict['clipwise_output'] = clipwise_output
 
        return output_dict 

DROPOUT = 0.1

def res_norm(input):

    bs,c,f,t = input.size()
    mean_times_and_channels = torch.mean(input, dim=(1,3)).reshape(-1,1,f,1)
    std_times_and_channels = torch.std(input, dim=(1,3)).reshape(-1,1,f,1)
    sita = 1e-5
    x = (input - mean_times_and_channels) / torch.sqrt(std_times_and_channels + sita)

    return x+input

class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels*sub_bands, eps=eps)

    def forward(self, x):

        N, C, F, T = x.size()
        x = x.contiguous().view(N, C * self.sub_bands, F // self.sub_bands, T)
        x = self.bn(x)
        return x.view(N, C, F, T)

class NormalBlock(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 4) if use_subspectral else nn.BatchNorm2d(n_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(3, 1), padding="same", groups=n_chan),
            norm_layer,
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(n_chan, n_chan, kernel_size=(1, 3), padding="same", groups=n_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(n_chan),
            nn.SiLU(),
            nn.Conv2d(n_chan, n_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        x1 = self.f2(x)

        x2 = torch.mean(x1, dim=2, keepdim=True)
        x2 = self.f1(x2)
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        if stride == 1:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), groups=out_chan, padding="same")
        else:
            conv = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 1), stride=(stride, 1), groups=out_chan, padding=(1, 0))

        norm_layer = SubSpectralNorm(out_chan, 4) if use_subspectral else nn.BatchNorm2d(out_chan)
        self.f2 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
            conv,
            norm_layer,
        )

        self.f1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=(1, 3), padding="same", groups=out_chan, dilation=(1, dilation)),
            nn.BatchNorm2d(out_chan),
            nn.SiLU(),
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.f2(x)
        n_freq = x.shape[2]
        x1 = torch.mean(x, dim=2, keepdim=True)
        x1 = self.f1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)


class BcResNetModel(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=25, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock(8*scale, 12*scale, dilation=2, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock(12*scale, dilation=2, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock(12*scale, 16*scale, dilation=4, stride=2, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock(16*scale, dilation=4, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock(16*scale, 20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock(20*scale, dilation=8, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, classes_num, kernel_size=1)
    
def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)
        x = res_norm(x)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = res_norm(x)

        x = nn.MaxPool2d((1,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = res_norm(x)

        x = nn.MaxPool2d((1,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = res_norm(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = res_norm(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        
        x = self.head_conv(x)
        embedding = x
        x = torch.mean(x, dim=(2,3), keepdim=True)

        x = x.squeeze()
        clipwise_output = x

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict


class NormalBlock_res2(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 4) if use_subspectral else nn.BatchNorm2d(n_chan)

        scale = 4
        width       = int(math.floor(n_chan / scale))
        f_convs       = []
        f_ssn         = []
        self.nums   = scale 
        num_pad = math.floor(3/2)*dilation
        for i in range(self.nums):
            f_convs.append(nn.Conv2d(width, width, kernel_size=(3, 1), padding="same"))
            f_ssn.append(SubSpectralNorm(width, 4))
        self.f_convs  = nn.ModuleList(f_convs)
        self.f_ssn    = nn.ModuleList(f_ssn)
        self.width = width

        t_convs       = []
        t_relu        = []
        t_bn          = []
        for i in range(self.nums):
            t_convs.append(nn.Conv2d(width, width, kernel_size=(1, 3), padding="same", dilation=(1, dilation)))
            t_bn.append(nn.BatchNorm2d(width))
            t_relu.append(nn.ReLU())
        self.t_convs  = nn.ModuleList(t_convs)
        self.t_ssn    = nn.ModuleList(t_bn)
        self.t_relu    = nn.ModuleList(t_relu)

        self.activation = nn.ReLU()

    def forward(self, x):
        n_freq = x.shape[2]
        f_spx = torch.split(x, self.width, 1)

        for i in range(self.nums):
            if i==0:
                f_sp = f_spx[i]
            else:
                f_sp = f_sp + f_spx[i]
            f_sp = self.f_convs[i](f_sp)
            f_sp = self.f_ssn[i](f_sp)
            if i==0:
                f_out = f_sp
            else:
                f_out = torch.cat((f_out, f_sp), 1)
        x1 = f_out

        x2 = torch.mean(x1, dim=2, keepdim=True)

        t_spx = torch.split(x2, self.width, 1)
        for i in range(self.nums):
            if i==0:
                t_sp = t_spx[i]
            else:
                t_sp = t_sp + t_spx[i]
            t_sp = self.t_convs[i](t_sp)
            t_sp = self.t_relu[i](t_sp)
            if i==0:
                t_out = t_sp
            else:
                t_out = torch.cat((t_out, t_sp), 1)
        x2 = t_out
        
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1 + x2)


class TransitionBlock_res2(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        self.f2_conv1x1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        
        scale = 4
        width       = int(math.floor(out_chan / scale))
        f_convs       = []
        f_ssn         = []
        self.nums   = scale 
        num_pad = math.floor(3/2)*dilation

        if stride == 1:
            conv = nn.Conv2d(width, width, kernel_size=(3, 1), padding="same")
        else:
            conv = nn.Conv2d(width, width, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0))

        for i in range(self.nums):
            f_convs.append(conv)
            f_ssn.append(SubSpectralNorm(width, 4))
        self.f_convs  = nn.ModuleList(f_convs)
        self.f_ssn    = nn.ModuleList(f_ssn)
        self.width = width

        self.f1_conv1x1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        t_convs       = []
        t_relu        = []
        t_bn          = []
        for i in range(self.nums):
            t_convs.append(nn.Conv2d(width, width, kernel_size=(1, 3), padding="same", dilation=(1, dilation)))
            t_bn.append(nn.BatchNorm2d(width))
            t_relu.append(nn.ReLU())
        self.t_convs  = nn.ModuleList(t_convs)
        self.t_ssn    = nn.ModuleList(t_bn)
        self.t_relu    = nn.ModuleList(t_relu)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor):

        x = self.f2_conv1x1(x)

        # Freq Conv by res2net

        f_spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i==0:
                f_sp = f_spx[i]
            else:
                f_sp = f_sp + f_spx[i]
            f_sp = self.f_convs[i](f_sp)
            f_sp = self.f_ssn[i](f_sp)
            if i==0:
                f_out = f_sp
            else:
                f_out = torch.cat((f_out, f_sp), 1)
        
        x = f_out
        # x = self.f2(x)
        n_freq = x.shape[2]
        # Freq avg pool
        x1 = torch.mean(x, dim=2, keepdim=True)

        t_spx = torch.split(x1, self.width, 1)
        for i in range(self.nums):
            if i==0:
                t_sp = t_spx[i]
            else:
                t_sp = t_sp + t_spx[i]
            t_sp = self.t_convs[i](t_sp)
            t_sp = self.t_relu[i](t_sp)
            if i==0:
                t_out = t_sp
            else:
                t_out = torch.cat((t_out, t_sp), 1)

        x1 = t_out
        x1 = self.f1_conv1x1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(x + x1)

   
class BcRes2NetModel(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock_res2(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2(12*scale, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock_res2(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock_res2(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 96, kernel_size=1)

        self.head_conv = nn.Conv2d(96, classes_num, kernel_size=1)
    
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        # print(f'inpuit:{input.shape}')
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
        
        # mixstyle
        if style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        x = res_norm(x)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = res_norm(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = res_norm(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=(2,3), keepdim=True)
        
        x = self.head_conv(x)

        x = x.squeeze()
        clipwise_output = x
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

class BcRes2NetModel_quant_no_feat(nn.Module):
    def __init__(self, model_width, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        scale = model_width

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.quant = torch.quantization.QuantStub()

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t2 = TransitionBlock_res2(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2(12*scale, dropout=dropout, use_subspectral=use_subspectral) 

        self.t3 = TransitionBlock_res2(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.t4 = TransitionBlock_res2(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 96, kernel_size=1)

        self.head_conv = nn.Conv2d(96, classes_num, kernel_size=1)

        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):

        x = self.quant(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        x = x.transpose(2, 3)
        x = res_norm(x)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = res_norm(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = res_norm(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=(2,3), keepdim=True)
        
        x = self.head_conv(x)
        x = x.squeeze()
        clipwise_output = self.dequant(x)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict

class mel_model(nn.Module):
    def __init__(self,sample_rate, n_fft, window_size, hop_size, mel_bins):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        
    def forward(self, input):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        return x

    
class SPA_2d(nn.Module):
    def __init__(self, channel):
        super(SPA_2d, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Sequential(
            nn.Linear(channel * 5, 48),
            nn.ReLU(),
            nn.Linear(48, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ , _ = x.shape
        y1 = self.avg_pool1(x).reshape((b, -1))
        y2 = self.avg_pool2(x).reshape((b, -1))
        y = torch.cat((y1, y2), 1)
        y = self.fc(y).reshape((b, -1, 1, 1))
        return x * y
      
class BcRes2Net_spa_Model(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2(8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa11 = SPA_2d(8*scale)

        self.t2 = TransitionBlock_res2(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2(12*scale, dropout=dropout, use_subspectral=use_subspectral) 
        self.spa21 = SPA_2d(12*scale)

        self.t3 = TransitionBlock_res2(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa31 = SPA_2d(16*scale)

        self.t4 = TransitionBlock_res2(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa41 = SPA_2d(20*scale)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, classes_num, kernel_size=1)
        
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        x = res_norm(x)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = self.spa11(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = self.spa21(x)
        x = res_norm(x)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = self.spa31(x)
        x = res_norm(x)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = self.spa41(x)
        x = res_norm(x)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = self.head_conv(x)
        
        embedding = x

        x = torch.mean(x, dim=(2,3), keepdim=True)

        x = x.squeeze()
        clipwise_output = x
        output_dict = {'clipwise_output': clipwise_output, 'embedding':embedding}
        return output_dict

class NormalBlock_res2_skip(nn.Module):
    def __init__(self, n_chan: int, *, dilation: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()
        norm_layer = SubSpectralNorm(n_chan, 4) if use_subspectral else nn.BatchNorm2d(n_chan)

        scale = 4
        width       = int(math.floor(n_chan / scale))
        f_convs       = []
        f_ssn         = []
        self.nums   = scale 
        num_pad = math.floor(3/2)*dilation
        for i in range(self.nums):
            f_convs.append(nn.Conv2d(width, width, kernel_size=(3, 1), padding=(1,0)))
            f_ssn.append(SubSpectralNorm(width, 4))
        self.f_convs  = nn.ModuleList(f_convs)
        self.f_ssn    = nn.ModuleList(f_ssn)
        self.width = width

        t_convs       = []
        t_relu        = []
        t_bn          = []
        for i in range(self.nums):
            t_convs.append(nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0,1), dilation=(1, dilation)))
            t_bn.append(nn.BatchNorm2d(width))
            t_relu.append(nn.ReLU())
        self.t_convs  = nn.ModuleList(t_convs)
        self.t_ssn    = nn.ModuleList(t_bn)
        self.t_relu    = nn.ModuleList(t_relu)

        self.activation = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        n_freq = x.shape[2]
        f_spx = torch.split(x, self.width, 1)

        for i in range(self.nums):
            if i==0:
                f_sp = f_spx[i]
            else:
                f_sp = self.skip_add.add(f_sp, f_spx[i])
            f_sp = self.f_convs[i](f_sp)
            f_sp = self.f_ssn[i](f_sp)
            if i==0:
                f_out = f_sp
            else:
                f_out = torch.cat((f_out, f_sp), 1)
        x1 = f_out

        x2 = torch.mean(x1, dim=2, keepdim=True)

        t_spx = torch.split(x2, self.width, 1)
        for i in range(self.nums):
            if i==0:
                t_sp = t_spx[i]
            else:
                t_sp = self.skip_add.add(t_sp, t_spx[i])
            t_sp = self.t_convs[i](t_sp)
            t_sp = self.t_relu[i](t_sp)
            if i==0:
                t_out = t_sp
            else:
                t_out = torch.cat((t_out, t_sp), 1)
        x2 = t_out
        
        x2 = x2.repeat(1, 1, n_freq, 1)

        return self.activation(self.skip_add.add(x, self.skip_add.add(x1, x2)))


class TransitionBlock_res2_skip(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, *, dilation: int = 1, stride: int = 1, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        self.f2_conv1x1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(),
        )
        
        scale = 4
        width       = int(math.floor(out_chan / scale))
        f_convs       = []
        f_ssn         = []
        self.nums   = scale 
        num_pad = math.floor(3/2)*dilation

        for i in range(self.nums):
            f_convs.append(nn.Conv2d(width, width, kernel_size=(3, 1), padding=(1,0)))
            f_ssn.append(SubSpectralNorm(width, 4))
        self.f_convs  = nn.ModuleList(f_convs)
        self.f_ssn    = nn.ModuleList(f_ssn)
        self.width = width

        self.f1_conv1x1 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=1),
            nn.Dropout2d(dropout)
        )

        t_convs       = []
        t_relu        = []
        t_bn          = []
        for i in range(self.nums):
            t_convs.append(nn.Conv2d(width, width, kernel_size=(1, 3), padding=(0,1), dilation=(1, dilation)))
            t_bn.append(nn.BatchNorm2d(width))
            t_relu.append(nn.ReLU())
        self.t_convs  = nn.ModuleList(t_convs)
        self.t_ssn    = nn.ModuleList(t_bn)
        self.t_relu    = nn.ModuleList(t_relu)

        self.activation = nn.ReLU()

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor):

        x = self.f2_conv1x1(x)

        f_spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i==0:
                f_sp = f_spx[i]
            else:
                f_sp = self.skip_add.add(f_sp, f_spx[i])
            f_sp = self.f_convs[i](f_sp)
            f_sp = self.f_ssn[i](f_sp)
            if i==0:
                f_out = f_sp
            else:
                f_out = torch.cat((f_out, f_sp), 1)
        
        x = f_out
        n_freq = x.shape[2]
        # Freq avg pool
        x1 = torch.mean(x, dim=2, keepdim=True)

        t_spx = torch.split(x1, self.width, 1)
        for i in range(self.nums):
            if i==0:
                t_sp = t_spx[i]
            else:
                t_sp = self.skip_add.add(t_sp, t_spx[i])
            t_sp = self.t_convs[i](t_sp)
            t_sp = self.t_relu[i](t_sp)
            if i==0:
                t_out = t_sp
            else:
                t_out = torch.cat((t_out, t_sp), 1)

        x1 = t_out
        x1 = self.f1_conv1x1(x1)
        x1 = x1.repeat(1, 1, n_freq, 1)

        return self.activation(self.skip_add.add(x, x1))
    
class BcRes2NetModel_resnorm_bn_quant(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)
        self.quant = torch.quantization.QuantStub()

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.bn1 = nn.BatchNorm2d(256)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2_skip(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2_skip(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn2 = nn.BatchNorm2d(128)

        self.t2 = TransitionBlock_res2_skip(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2_skip(12*scale, dropout=dropout, use_subspectral=use_subspectral) 

        self.bn3 = nn.BatchNorm2d(64)

        self.t3 = TransitionBlock_res2_skip(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn4 = nn.BatchNorm2d(32)

        self.t4 = TransitionBlock_res2_skip(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn5 = nn.BatchNorm2d(32)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, classes_num, kernel_size=1)

        self.dequant = torch.quantization.DeQuantStub()

        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = self.quant(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        # print(x.shape)
        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        # if self.training and mixstyle_alpha is not None:
        #     x = mixstyle(x, mixstyle_p, mixstyle_alpha)
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = x.transpose(1, 2)
        x2 = self.bn2(x)
        x = self.skip_add.add(x, x2)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)

        x = x.transpose(1, 2)
        x3 = self.bn3(x)
        x = self.skip_add.add(x, x3)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = x.transpose(1, 2)
        x4 = self.bn4(x)
        x = self.skip_add.add(x, x4)
        x = x.transpose(1, 2)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = x.transpose(1, 2)
        x5 = self.bn5(x)
        x = self.skip_add.add(x, x5)
        x = x.transpose(1, 2)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        embedding = x
        x = torch.mean(x, dim=(2,3), keepdim=True)
        x = self.head_conv(x)
        
        x = x.squeeze()
        clipwise_output = self.dequant(x)
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict

class BcRes2NetModel_deep_quant(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.quant = torch.quantization.QuantStub()
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.bn1 = nn.BatchNorm2d(256)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2(8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.bn2 = nn.BatchNorm2d(128)

        self.t12 = TransitionBlock_res2(8*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n121 = NormalBlock_res2(8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.bn3 = nn.BatchNorm2d(128)

        self.t2 = TransitionBlock_res2(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2(12*scale, dropout=dropout, use_subspectral=use_subspectral) 
        self.bn4 = nn.BatchNorm2d(64)

        self.t22 = TransitionBlock_res2(12*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n221 = NormalBlock_res2(12*scale, dropout=dropout, use_subspectral=use_subspectral) 
        self.bn5 = nn.BatchNorm2d(64)

        self.t3 = TransitionBlock_res2(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.bn6 = nn.BatchNorm2d(32)

        self.t32 = TransitionBlock_res2(16*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n321 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n322 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n323 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.bn7 = nn.BatchNorm2d(32)

        self.t4 = TransitionBlock_res2(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.bn8 = nn.BatchNorm2d(32)

        self.t42 = TransitionBlock_res2(20*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n421 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n422 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n423 = NormalBlock_res2(20*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn9 = nn.BatchNorm2d(32)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 96, kernel_size=1)

        self.head_conv = nn.Conv2d(96, classes_num, kernel_size=1)

        self.dequant = torch.quantization.DeQuantStub()

        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = self.quant(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        # print(x.shape)
        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.t12(x)
        x = self.n121(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.t22(x)
        x = self.n221(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)

        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.t32(x)
        x = self.n321(x)
        x = self.n322(x)
        x = self.n323(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)

        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.t42(x)
        x = self.n421(x)
        x = self.n422(x)
        x = self.n423(x)
        
        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        x = torch.mean(x, dim=(2,3), keepdim=True)
        
        x = self.head_conv(x)

        x = x.squeeze()
        clipwise_output = self.dequant(x)
        output_dict = {'clipwise_output': clipwise_output}
        return output_dict
class BcRes2NetModel_width_resnorm_bn_quant(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)
        self.quant = torch.quantization.QuantStub()

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.bn1 = nn.BatchNorm2d(256)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2_skip(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2_skip(8*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn2 = nn.BatchNorm2d(128)

        self.t2 = TransitionBlock_res2_skip(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2_skip(12*scale, dropout=dropout, use_subspectral=use_subspectral) 

        self.bn3 = nn.BatchNorm2d(64)

        self.t3 = TransitionBlock_res2_skip(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn4 = nn.BatchNorm2d(32)

        self.t4 = TransitionBlock_res2_skip(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)

        self.bn5 = nn.BatchNorm2d(32)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, classes_num, kernel_size=1)

        self.dequant = torch.quantization.DeQuantStub()

        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = self.quant(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        

        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = x.transpose(1, 2)
        x2 = self.bn2(x)
        x = self.skip_add.add(x, x2)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = x.transpose(1, 2)
        x3 = self.bn3(x)
        x = self.skip_add.add(x, x3)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = x.transpose(1, 2)
        x4 = self.bn4(x)
        x = self.skip_add.add(x, x4)
        x = x.transpose(1, 2)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = x.transpose(1, 2)
        x5 = self.bn5(x)
        x = self.skip_add.add(x, x5)
        x = x.transpose(1, 2)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        embedding = x
        x = torch.mean(x, dim=(2,3), keepdim=True)
        x = self.head_conv(x)
        
        x = x.squeeze()
        clipwise_output = self.dequant(x)
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict
     
class BcRes2NetModel_spa_bn_quant(nn.Module):
    def __init__(self, model_width, sample_rate, n_fft, window_size, hop_size, mel_bins, classes_num: int = 10, *, dropout: float = DROPOUT, use_subspectral: bool = True):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        scale = model_width

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
            win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, 
            n_mels=mel_bins, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=8, time_stripes_num=2, 
            freq_drop_width=40, freq_stripes_num=2)

        self.quant = torch.quantization.QuantStub()
        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.bn1 = nn.BatchNorm2d(256)

        self.input_conv = nn.Conv2d(1, 16*scale, kernel_size=(5, 5), stride=(2, 1), padding=2)

        self.t1 = TransitionBlock_res2_skip(16*scale, 8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n11 = NormalBlock_res2_skip(8*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa1 = SPA_2d(8*scale)
        self.bn2 = nn.BatchNorm2d(128)

        self.t2 = TransitionBlock_res2_skip(8*scale, 12*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n21 = NormalBlock_res2_skip(12*scale, dropout=dropout, use_subspectral=use_subspectral) 
        self.spa2 = SPA_2d(12*scale)
        self.bn3 = nn.BatchNorm2d(64)

        self.t3 = TransitionBlock_res2_skip(12*scale, 16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n31 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n32 = NormalBlock_res2(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n33 = NormalBlock_res2_skip(16*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa3 = SPA_2d(16*scale)
        self.bn4 = nn.BatchNorm2d(32)

        self.t4 = TransitionBlock_res2_skip(16*scale, 20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n41 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n42 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.n43 = NormalBlock_res2_skip(20*scale, dropout=dropout, use_subspectral=use_subspectral)
        self.spa4 = SPA_2d(20*scale)
        self.bn5 = nn.BatchNorm2d(32)

        self.dw_conv = nn.Conv2d(20*scale, 20*scale, kernel_size=(5, 5), groups=20)
        self.onexone_conv = nn.Conv2d(20*scale, 32*scale, kernel_size=1)

        self.head_conv = nn.Conv2d(32*scale, classes_num, kernel_size=1)

        self.dequant = torch.quantization.DeQuantStub()

        self.skip_add = nn.quantized.FloatFunctional()
    
    def forward(self, input, style_lmda=None, style_perm=None, rn_indices=None, lam=None, spec_aug=False):

        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)   #(64,1,101,80)
        x = x.transpose(1, 3)
        

        if self.training and spec_aug:
            x = self.spec_augmenter(x)
        x = x.transpose(2,3)
        
        # Mixup on spectrogram
        if self.training and lam is not None:
            x = x * lam.reshape(x.size()[0], 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(x.size()[0], 1, 1, 1))
            
        # mixstyle
        if self.training and style_lmda is not None:
            x = syn_mixstyle(x, lmda=style_lmda, perm=style_perm)

        x = x.transpose(1, 2)
        x1 = self.bn1(x)
        x = self.skip_add.add(x, x1)
        x = x.transpose(1, 2)

        x = self.input_conv(x)
        x = self.t1(x)
        x = self.n11(x)
        x = self.spa1(x)
        x = x.transpose(1, 2)
        x2 = self.bn2(x)
        x = self.skip_add.add(x, x2)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t2(x)
        x = self.n21(x)
        x = self.spa2(x)
        x = x.transpose(1, 2)
        x3 = self.bn3(x)
        x = self.skip_add.add(x, x3)
        x = x.transpose(1, 2)

        x = nn.MaxPool2d((2,2))(x)
        x = self.t3(x)
        x = self.n31(x)
        x = self.n32(x)
        x = self.n33(x)
        x = self.spa3(x)
        x = x.transpose(1, 2)
        x4 = self.bn4(x)
        x = self.skip_add.add(x, x4)
        x = x.transpose(1, 2)

        x = self.t4(x)
        x = self.n41(x)
        x = self.n42(x)
        x = self.n43(x)
        x = self.spa4(x)
        x = x.transpose(1, 2)
        x5 = self.bn5(x)
        x = self.skip_add.add(x, x5)
        x = x.transpose(1, 2)

        x = self.dw_conv(x)
        x = self.onexone_conv(x)

        embedding = x
        x = torch.mean(x, dim=(2,3), keepdim=True)
        x = self.head_conv(x)
        

        x = x.squeeze()
        clipwise_output = x
        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        return output_dict