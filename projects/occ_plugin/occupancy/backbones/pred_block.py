# Developed by Junyi Ma
# Cam4DOcc: Benchmark for Camera-Only 4D Occupancy Forecasting in Autonomous Driving Applications
# https://github.com/haomo-ai/Cam4DOcc

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import BACKBONES
from collections import OrderedDict
from mmcv.cnn import build_norm_layer


class Residual(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3,3,1),
        dilation=1,
        norm_cfg=None
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        # padding_size = ((kernel_size - 1) * dilation + 1) // 2
        padding_size = [0,0,0]
        if dilation!=0:
            padding_size[0] = ((kernel_size[0] - 1) * dilation + 1) // 2
            padding_size[1] = ((kernel_size[1] - 1) * dilation + 1) // 2
            padding_size[2] = ((kernel_size[2] - 1) * dilation + 1) // 2
        padding_size = tuple(padding_size)

        conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, bias=False, dilation=dilation, padding=padding_size)
        self.layers = nn.Sequential(conv, build_norm_layer(norm_cfg, out_channels)[1], nn.LeakyReLU(inplace=True))


        if out_channels == in_channels :
            self.projection = None
        else:
            projection = OrderedDict()
            projection.update(
                {
                    'conv_skip_proj': nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                    'bn_skip_proj': build_norm_layer(norm_cfg, out_channels)[1],
                }
            )
            self.projection = nn.Sequential(projection)


    def forward(self, x):
        x_residual = self.layers(x)
        if self.projection is not None:
            x_projected = self.projection(x)
            return x_residual + x_projected
        return x_residual + x


@BACKBONES.register_module()
class Predictor(nn.Module):
    def __init__(
        self,
        n_input_channels=None,
        in_timesteps=None,
        out_timesteps=None,
        norm_cfg=None,
    ):
        super(Predictor, self).__init__()
        
        self.predictor = nn.ModuleList()
        for nf in n_input_channels:
            self.predictor.append(nn.Sequential(
                Residual(nf * in_timesteps, nf * in_timesteps, norm_cfg=norm_cfg),
                Residual(nf * in_timesteps, nf * in_timesteps, norm_cfg=norm_cfg),
                Residual(nf * in_timesteps, nf * out_timesteps, norm_cfg=norm_cfg),
                Residual(nf * out_timesteps, nf * out_timesteps, norm_cfg=norm_cfg),
                Residual(nf * out_timesteps, nf * out_timesteps, norm_cfg=norm_cfg),
            ))

    def forward(self, x):
        assert len(x) == len(self.predictor), f'The number of input feature tensors ({len(x)}) must be the same as the number of STPredictor blocks {len(self.predictor)}.'
        
        y = []
        
        for i in range(len(x)):
            b, c, _, _, _ = x[i].shape
            y.append(self.predictor[i](x[i]))
                
        return y