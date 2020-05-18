#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5):
        """
        defines layers in CNN
        @param filter_num: number of channels in the output
        @param input_size: number of word embeddings
        """
        super(CNN, self).__init__()
        self.convolution = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size,
                                     kernel_size=kernel_size)
    def forward(self, x_reshaped):
        """
        runs x through conv1d layer, relu activation and maxpool
        @param x: input of shape (N,char_embed, word_embed)
        @return: x_conv_out of shape (N, char_embed - 4, 1)
        """
        x_conv = self.convolution(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
        return x_conv_out
### END YOUR CODE

