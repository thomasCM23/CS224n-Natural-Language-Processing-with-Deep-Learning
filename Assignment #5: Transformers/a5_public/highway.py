#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):

    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, word_embedding_size):
        """
        :param word_embedding_size:
        """
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(in_features=word_embedding_size, out_features=word_embedding_size, bias=True)
        self.gate_layer = nn.Linear(in_features=word_embedding_size, out_features=word_embedding_size, bias=True)

    def forward(self, x_conv: torch.Tensor) -> torch.Tensor:
        """
        take batch of conv output and compute highway forward
        :param x_conv: shape batch_size x word_embed_size
        :return: word_embedding (torch.Tensor), shape batch_size x word_embed_size
        """
        x_proj = F.relu(self.proj_layer(x_conv))
        x_gate = torch.sigmoid(self.gate_layer(x_conv))

        return x_gate * x_proj + (1 - x_gate) * x_conv  # x_highway

    ### END YOUR CODE
