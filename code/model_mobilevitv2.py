# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

import torch
import torch.nn as nn
import math

# 上采样块
class Up_Block(nn.Module):
    def __init__(self, in_channel_down, in_channel_up, out_channel):
        super(Up_Block, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel_down+in_channel_up, in_channel_down+in_channel_up, 3, 1, 1, groups=in_channel_down+in_channel_up, bias=False),
                nn.BatchNorm2d(in_channel_down+in_channel_up),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True),
                #nn.Dropout(0.2),
                nn.Conv2d(in_channel_down+in_channel_up, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs2, inputs1):
        results2 = self.up(inputs2)  # 上采样
        padding = (results2.size()[-1] - inputs1.size()[-1]) // 2  # shape(batch, channel, width, height)
        results1 = F.pad(inputs1, 2 * [padding, padding])
        results = torch.cat([results1, results2], 1)  # 合并
        return self.conv(results)  # 卷积
    
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope = 0.2, inplace=True),
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope = 0.2, inplace=True)
    )

class LinearSelfAttention(nn.Module):

    def __init__(self, embed_dim, attn_dropout = 0.0, bias = False, *args, **kwargs):
        super(LinearSelfAttention, self).__init__()

        self.qkv_proj = nn.Conv2d(in_channels = embed_dim, out_channels = 1 + (2 * embed_dim), bias = bias, kernel_size = 1)

        self.attn_dropout = nn.Dropout(p = attn_dropout)
        self.out_proj = nn.Conv2d(in_channels = embed_dim, out_channels = embed_dim,           bias = bias, kernel_size = 1)
        self.embed_dim = embed_dim

    def forward(self, x):
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

class LinearAttnFFN(nn.Module):

    def __init__(self, embed_dim, ffn_latent_dim, patch_size, n_patches, attn_dropout = 0.0, dropout = 0.1, ffn_dropout = 0.0, *args, **kwargs, ):
        super().__init__()
        attn_unit = LinearSelfAttention(embed_dim = embed_dim, attn_dropout = attn_dropout, bias = True)

        self.pre_norm_attn = nn.Sequential(
            nn.LayerNorm((int(embed_dim), int(patch_size), int(n_patches))),
            attn_unit,
            nn.Dropout(p = dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            nn.LayerNorm((int(embed_dim), int(patch_size), int(n_patches))),
            nn.Conv2d(in_channels = embed_dim, out_channels = ffn_latent_dim, kernel_size = 1, stride = 1, bias = True),
            nn.GELU(),
            nn.Dropout(p = ffn_dropout),
            nn.Conv2d(in_channels = ffn_latent_dim, out_channels = embed_dim, kernel_size = 1, stride = 1, bias = True),
            nn.Dropout(p = dropout),
        )

    def forward(self, x):
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x
    
class MobileViTBlockv2(nn.Module):
    def __init__(self, inp, stride, transformer_channels, ffn_dim, patch_h, patch_w, n_patches, num_layers, dropout = .0, attn_dropout = .0, ffn_dropout =.0):
        super(MobileViTBlockv2, self).__init__()
        self.stride = stride
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_size = patch_h*patch_w
        assert stride in [1, 2]
        
        self.local_rep = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True),
                nn.Conv2d(inp, transformer_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(transformer_channels),
            )
        
        global_rep = [
            LinearAttnFFN(
                embed_dim = transformer_channels,
                ffn_latent_dim = ffn_dim,
                patch_size = self.patch_size,
                n_patches  = n_patches+1,
                attn_dropout = attn_dropout,
                dropout = dropout,
                ffn_dropout = ffn_dropout,
            )
            for block_idx in range(num_layers)
        ]
        global_rep.append(nn.LayerNorm((int(transformer_channels), int(self.patch_size), int(n_patches+1))))
        self.global_rep = nn.Sequential(*global_rep)
        
        self.conv_proj = nn.Sequential(
                nn.Conv2d(transformer_channels, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
            )
        self.rain_vector = nn.Linear(9, transformer_channels*self.patch_size)

    def unfolding_pytorch(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches, output_size):
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def forward(self, x, rv):
        fm = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm)
        # learn global representations on all patches
        B, C, P, N = patches.shape
        rv = self.rain_vector(rv).reshape(B, C, P, 1)
        patches = torch.cat([patches, rv], dim = 3)
        patches = self.global_rep(patches)[:, :, :, :N]
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm
        
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, *args, **kwargs):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        assert stride in [1, 2]

        #hidden_dim = min(round(inp * expand_ratio), 1024)
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的调整
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                #-----------------------------------#
                #   利用1x1卷积进行通道数的上升
                #-----------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True),
                #--------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                #--------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(negative_slope = 0.2, inplace=True),
                #-----------------------------------#
                #   利用1x1卷积进行通道数的下降
                #-----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetViT2(nn.Module):
    def __init__(self, input_size=256, final_channel = 1, width_mult=1., dropout = .2, attn_dropout = .0, ffn_dropout = .2):
        super(MobileNetViT2, self).__init__()
        
        input_channel = 64
        interverted_residual_setting = [
            # t, c, n, s
            [4, 128, 2, 2,   0,   0, 0,],
            [4, 256, 3, 2,   0,   0, 0,],
            [4, 512, 5, 2, 512,1024, 2,],
            [4, 512, 5, 2, 512,1024, 2,],
            [4, 512, 5, 2, 512,1024, 2,],
        ]
        input_channel = int(input_channel * width_mult)
        self.input_channel = input_channel
        self.final_channel = final_channel
        self.interverted_residual_setting = interverted_residual_setting
        
        assert input_size % 32 == 0
        self.features = [conv_bn(6, self.input_channel, 1)]
        self.decoders = []
        

        self.depth = 0
        self.outs_depth = [0]
        self.is_mit = [0]
        cnt = 0
        for t, c, n, s, tf_c, ffn_d, ph in interverted_residual_setting:
            output_channel = int(c * width_mult)
            if self.depth == 0:
                self.decoders.append(Up_Block(in_channel_down = output_channel, in_channel_up = input_channel, out_channel = self.final_channel))
            else:
                self.decoders.append(Up_Block(in_channel_down = output_channel, in_channel_up = input_channel, out_channel = input_channel))
            self.depth += 1
            
            for i in range(n):  
                cnt += 1
                if i == 0:
                    self.features.append(InvertedResidual(inp = input_channel, oup = output_channel, stride = s, expand_ratio = t))
                    self.is_mit.append(0)
                elif tf_c == 0:
                    self.features.append(InvertedResidual(inp = input_channel, oup = output_channel, stride = 1, expand_ratio = t))
                    self.is_mit.append(0)
                else:
                    self.features.append(MobileViTBlockv2(inp = input_channel                      , stride = 1, 
                                                          transformer_channels = tf_c, ffn_dim = ffn_d, patch_h = ph, patch_w = ph, n_patches = (input_size/(2**self.depth)/ph)**2,
                                                          num_layers = n-1, dropout = dropout, attn_dropout = attn_dropout, ffn_dropout = ffn_dropout))
                    self.is_mit.append(1)
                    break
                input_channel = output_channel
            self.outs_depth.append(cnt)
        
        self.outs_depth = self.outs_depth[:-1]
        self.features = nn.Sequential(*self.features)
        self.decoders = nn.Sequential(*self.decoders)

        self.MSELoss = nn.MSELoss()
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, inputs, rain_p, labels = None, mask = None, ssl_train = False):
        x = inputs
        outs = []
        cnt = 0
        for blk in self.features:
            if self.is_mit[cnt] == 1:
                x = blk(x, rain_p)
            else:
                x = blk(x)
            if cnt in self.outs_depth:
                outs.append(x)
            cnt += 1
            
        
        for i in range(self.depth):
            x = self.decoders[self.depth-1-i](x, outs[self.depth-1-i])
        
        cls = None
        
        x = torch.squeeze(x, 1) 
        if labels is not None:#supervised
            labels = torch.squeeze(labels, 3)
            mseloss  = self.MSELoss(x, labels)
        else:
            mseloss = None  
                
        output = {
            "loss": mseloss,
            "results": x,
            "cls": cls
        }
        return output
