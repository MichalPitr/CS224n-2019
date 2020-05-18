#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from cnn import CNN

N, input_size, filters, embedding = 10, 3, 6, 10
x = torch.randn(N, input_size, embedding)
model = CNN(filter_num=filters, input_size=input_size)

y_pred = model(x)
print(x)
print(y_pred)
print(x.shape)
print(y_pred.shape)


