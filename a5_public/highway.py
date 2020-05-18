#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn


class Highway(nn.Module):
    def __init__(self, embed_size):
        """
        Instantiates nn modules and assigns them as member variables
        """
        super(Highway, self).__init__()
        self.linear_proj = nn.Linear(embed_size, embed_size)
        self.linear_gate = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """

        @param x: tensor of shape (batch_size, word_embed_size)
        @return: xhighway: tensor of shape (batch_size, word_embed_size)
        """
        xproj = self.linear_proj(x)
        xproj_relu = self.relu(xproj)

        xgate = self.linear_gate(x)
        xgate_sigmoid = self.sigmoid(xgate)

        xhighway = xgate_sigmoid * xproj_relu + (1 - xgate_sigmoid) * x
        # x_word_emb = self.dropout(xhighway)
        return xhighway

### END YOUR CODE
