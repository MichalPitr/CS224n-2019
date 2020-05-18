#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from highway import Highway

N, embed_size, out_size = 2, 2, 2
x = torch.randn(N, embed_size)
model = Highway(embed_size=embed_size, out_size=out_size, drop_prob=0.5)

y_pred = model(x)
print(x)
print(y_pred)
print(x.shape)
print(y_pred.shape)


