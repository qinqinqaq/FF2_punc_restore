#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'fkb'
__email__ = 'fkb@zjuici.com'

import torch
import torch.nn as nn
from transformers import BartModel, ElectraModel, BartConfig


class ParallelEndecoderGraph(nn.Module):
    """docstring for LanModelGraph"""
    def __init__(self, config):
        super(ParallelEndecoderGraph, self).__init__()
        conf = BartConfig()
        conf.vocab_size = 30522
        conf.encoder_layers = 4
        conf.decoder_layers = 2
        conf.encoder_attention_heads = 12
        conf.decoder_attention_heads = 12
        conf.max_position_embeddings = 512
        conf.d_model = 768
        conf.encoder_ffn_dim = 4 * conf.d_model
        conf.decoder_ffn_dim = 4 * conf.d_model
        # [100, 102, 0, 101, 103] ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        conf.eos_token_id = 102  # 句子结束的token_id [SEP]
        conf.bos_token_id = 101  # 句子开头的token_id [CLS]
        conf.pad_token_id = 0  # 用于pad填充的token_id
        conf.decoder_start_token_id = 102
        self.config = config
        self.decode_layer = BartModel(conf)

        self.encode_layer = ElectraModel.from_pretrained(config.lan_model, hidden_dropout_prob=config.xfmr_hidden_dropout_prob)

        fusion_in_features = config.lan_hidden_size + conf.d_model

        self.fusion_layer = nn.TransformerEncoderLayer(
             d_model=fusion_in_features
           , nhead=config.xfmr_num_attention_heads
           , dim_feedforward=config.xfmr_intermediate_size
           , dropout=config.xfmr_hidden_dropout_prob
           , activation='gelu'
              )

        self.dropout_layer = nn.Dropout(config.xfmr_hidden_dropout_prob)
        self.out_layer = nn.Linear(
                in_features=fusion_in_features
                , out_features=len(config.label2idx_dict))

        # freeze the Pre-trained Language Model
        if config.freeze_lan_model:
            for param in self.lan_layer.base_model.parameters():
                param.requires_grad = False

    def forward(
            self,
            xs,
            x_masks,
            y_tags=None,
            y_mask=None
    ):
        xs_encode = self.encode_layer(xs, attention_mask=x_masks)[0]
        xs_decode = self.decode_layer(xs, attention_mask=x_masks)[0]

        xs = torch.cat((xs_encode, xs_decode), dim=-1)
        xs = self.fusion_layer(xs)
        x = self.dropout_layer(xs)
        ys = self.out_layer(x)

        return ys



