#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5):
        super(CNN, self).__init__()

        self.conv1d = nn.Conv1d(in_channels=char_embed_size,
                                out_channels=num_filters,
                                kernel_size=kernel_size)
        # self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(max_word_length - kernel_size + 1)

    def forward(self, X_reshaped: torch.Tensor) -> torch.Tensor:
        # print(X_reshaped.shape)
        X_conv = self.conv1d(X_reshaped)
        # print("good")
        X_conv_out = self.maxpool(F.relu(X_conv))

        return torch.squeeze(X_conv_out, -1)

### END YOUR CODE

